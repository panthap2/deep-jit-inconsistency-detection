import enum
from enum import Enum
import json
import numpy as np
import re
import torch
from typing import List, NamedTuple

from external_cache import get_node_features

@enum.unique
class CommentCategory(Enum):
    Return = 0
    Param = 1
    Summary = 2

@enum.unique
class DiffEdgeType(Enum):
    PARENT = 0
    CHILD = 1
    SUBTOKEN_CHILD = 2
    SUBTOKEN_PARENT = 3
    PREV_SUBTOKEN = 4
    NEXT_SUBTOKEN = 5
    ALIGNED_NEIGHBOR = 6

@enum.unique
class SrcType(Enum):
    KEEP = 0
    INSERT = 1
    DELETE = 2
    REPLACE_OLD = 3
    REPLACE_NEW = 4
    MOVE = 5

class DiffTreeNode:
    def __init__(self, value, attribute, src, is_leaf):
        self.value = value
        self.node_id = -1
        self.parents = []
        self.attribute = attribute
        self.src = src
        self.is_leaf = is_leaf
        self.children = []
        self.prev_siblings = []
        self.next_siblings = []
        self.aligned_neighbors = []
        self.action_type = None
        self.prev_tokens = []
        self.next_tokens = []
        self.subtokens = []

        self.subtoken_children = []
        self.subtoken_parents = []
        self.prev_subtokens = []
        self.next_subtokens = []
    
    def to_json(self):
        return {
            'value': self.value,
            'node_id': self.node_id,
            'parent_ids': [p.node_id for p in self.parents],
            'attribute': self.attribute,
            'src': self.src,
            'is_leaf': self.is_leaf,
            'children_ids': [c.node_id for c in self.children],
            'prev_sibling_ids': [p.node_id for p in self.prev_siblings],
            'next_sibling_ids': [n.node_id for n in self.next_siblings],
            'aligned_neighbor_ids': [n.node_id for n in self.aligned_neighbors],
            'action_type': self.action_type,
        }
    
    @property
    def is_identifier(self):
        return self.is_leaf and self.attribute == 'SimpleName'

class DiffAST:
    def __init__(self, ast_root):
        self.node_cache = set()
        self.root = ast_root
        self.nodes = []
        self.traverse(self.root)
    
    def traverse(self, curr_node):
        if curr_node not in self.node_cache:
            self.node_cache.add(curr_node)
            curr_node.node_id = len(self.nodes)
            self.nodes.append(curr_node)
        for child in curr_node.subtoken_children:
            self.traverse(child)
        for child in curr_node.children:
            self.traverse(child)
    
    def to_json(self):
        return [n.to_json() for n in self.nodes]
        
    @property
    def leaves(self):
        return [n for n in self.nodes if n.is_leaf]
    
    @classmethod
    def from_json(cls, obj):
        nodes = []
        for node_obj in obj:
            node = DiffTreeNode(node_obj['value'], node_obj['attribute'], node_obj['src'], False)
            if 'action_type' in node_obj:
                node.action_type = node_obj['action_type']
            nodes.append(node)

        new_nodes = []

        for n, node_obj in enumerate(obj):
            nodes[n].parents = [nodes[i] for i in node_obj['parent_ids']]
            nodes[n].children = [nodes[i] for i in node_obj['children_ids']]
            nodes[n].prev_siblings = [nodes[i] for i in node_obj['prev_sibling_ids']]
            nodes[n].next_siblings = [nodes[i] for i in node_obj['next_sibling_ids']]
            nodes[n].aligned_neighbors = [nodes[i] for i in node_obj['aligned_neighbor_ids']]
            new_nodes.append(nodes[n])

            if len(nodes[n].children) == 0:
                nodes[n].is_leaf = True
                curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', nodes[n].value).split()
                new_curr = []
                for c in curr:
                    by_symbol = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip())
                    new_curr = new_curr + by_symbol
                nodes[n].subtokens = [s.lower() for s in new_curr]

                if len(nodes[n].subtokens) > 1:
                    for s in nodes[n].subtokens:
                        sub_node = DiffTreeNode(s, '', nodes[n].src, True)
                        sub_node.action_type = nodes[n].action_type
                        sub_node.subtoken_parents.append(nodes[n])
                        
                        if len(nodes[n].subtoken_children) > 0:
                            nodes[n].subtoken_children[-1].next_subtokens.append(sub_node)
                            sub_node.prev_subtokens.append(nodes[n].subtoken_children[-1])
                            
                        nodes[n].subtoken_children.append(sub_node)
                        new_nodes.append(sub_node)
            
                nodes[n].value = nodes[n].value.lower()
        
        return cls(new_nodes[0])

def insert_graph(batch, ex, ast, vocabulary, use_features, max_ast_length):
    batch.root_ids.append(batch.num_nodes)
    graph_node_positions = []
    for n, node in enumerate(ast.nodes):
        batch.graph_ids.append(batch.num_graphs)
        batch.is_internal.append(not node.is_leaf)
        batch.value_lookup_ids.append(vocabulary.get_id_or_unk(node.value))

        if node.action_type == 'Insert':
            src_type = SrcType.INSERT
        elif node.action_type == 'Delete':
            src_type = SrcType.DELETE
        elif node.action_type == 'Move':
            src_type = SrcType.MOVE
        elif node.src == 'old' and node.action_type == 'Update':
            src_type = SrcType.REPLACE_OLD
        elif node.src == 'new' and node.action_type == 'Update':
            src_type = SrcType.REPLACE_NEW
        else:
            src_type = SrcType.KEEP

        batch.src_type_ids.append(src_type.value)
        graph_node_positions.append(batch.num_nodes + node.node_id)
        
        for parent in node.parents:
            if parent.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.PARENT.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + parent.node_id))
        
        for child in node.children:
            if child.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.CHILD.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + child.node_id))
        
        for subtoken_parent in node.subtoken_parents:
            if subtoken_parent.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.SUBTOKEN_PARENT.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + subtoken_parent.node_id))
        
        for subtoken_child in node.subtoken_children:
            if subtoken_child.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.SUBTOKEN_CHILD.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + subtoken_child.node_id))
        
        for next_subtoken in node.next_subtokens:
            if next_subtoken.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.NEXT_SUBTOKEN.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + next_subtoken.node_id))
        
        for prev_subtoken in node.prev_subtokens:
            if prev_subtoken.node_id < len(ast.nodes):
                batch.edges[DiffEdgeType.PREV_SUBTOKEN.value].append(
                    (batch.num_nodes + node.node_id, batch.num_nodes + prev_subtoken.node_id))

        if len(batch.edges) == len(DiffEdgeType):
            for aligned_neighbor in node.aligned_neighbors:
                if aligned_neighbor.node_id < len(ast.nodes):
                    batch.edges[DiffEdgeType.ALIGNED_NEIGHBOR.value].append(
                        (batch.num_nodes + node.node_id, batch.num_nodes + aligned_neighbor.node_id))
        
    if use_features:
        node_features = get_node_features(ast.nodes, ex, max_ast_length)
        batch.node_features.extend(node_features)
    
    batch.node_positions.append(graph_node_positions)
    batch.num_nodes_per_graph.append(len(ast.nodes))
    batch.num_nodes += len(ast.nodes)
    batch.num_graphs += 1
    return batch


class GraphMethodBatch:
    def __init__(self, graph_ids, value_lookup_ids, src_type_ids, root_ids, is_internal,
                 edges, num_graphs, num_nodes, node_features, node_positions, num_nodes_per_graph):
        self.graph_ids = graph_ids
        self.value_lookup_ids = value_lookup_ids
        self.src_type_ids = src_type_ids
        self.root_ids = root_ids
        self.is_internal = is_internal
        self.edges = edges
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.node_positions = node_positions
        self.num_nodes_per_graph = num_nodes_per_graph

def initialize_graph_method_batch(num_edges):
    return GraphMethodBatch(
        graph_ids = [],
        value_lookup_ids = [],
        src_type_ids = [],
        root_ids = [],
        is_internal = [],
        edges = [[] for _ in range(num_edges)],
        num_graphs = 0,
        num_nodes = 0,
        node_features = [],
        node_positions = [],
        num_nodes_per_graph = []
    )

def tensorize_graph_method_batch(batch, device, max_num_nodes_per_graph):
    node_positions = np.zeros([batch.num_graphs, max_num_nodes_per_graph], dtype=np.int64)
    for g in range(batch.num_graphs):
        graph_node_positions = batch.node_positions[g]
        node_positions[g,:len(graph_node_positions)] = graph_node_positions
        node_positions[g,len(graph_node_positions):] = batch.root_ids[g]

    return GraphMethodBatch(
        torch.tensor(batch.graph_ids, dtype=torch.int64, device=device),
        torch.tensor(batch.value_lookup_ids, dtype=torch.int64, device=device),
        torch.tensor(batch.src_type_ids, dtype=torch.int64, device=device),
        torch.tensor(batch.root_ids, dtype=torch.int64, device=device),
        torch.tensor(batch.is_internal, dtype=torch.uint8, device=device),
        batch.edges, batch.num_graphs, batch.num_nodes,
        torch.tensor(batch.node_features, dtype=torch.float32, device=device),
        torch.tensor(node_positions, dtype=torch.int64, device=device),
        torch.tensor(batch.num_nodes_per_graph, dtype=torch.int64, device=device))

class GenerationBatchData(NamedTuple):
    """Stores tensorized batch used in generation model."""
    code_ids: torch.Tensor
    code_lengths: torch.Tensor
    trg_nl_ids: torch.Tensor
    trg_extended_nl_ids: torch.Tensor
    trg_nl_lengths: torch.Tensor
    invalid_copy_positions: torch.Tensor
    input_str_reps: List[List[str]]
    input_ids: List[List[str]]

class UpdateBatchData(NamedTuple):
    """Stores tensorized batch used in edit model."""
    code_ids: torch.Tensor
    code_lengths: torch.Tensor
    old_nl_ids: torch.Tensor
    old_nl_lengths: torch.Tensor
    trg_nl_ids: torch.Tensor
    trg_extended_nl_ids: torch.Tensor
    trg_nl_lengths: torch.Tensor
    invalid_copy_positions: torch.Tensor
    input_str_reps: List[List[str]]
    input_ids: List[List[str]]
    code_features: torch.Tensor
    nl_features: torch.Tensor
    labels: torch.Tensor
    graph_batch: GraphMethodBatch

class EncoderOutputs(NamedTuple):
    """Stores tensorized batch used in edit model."""
    encoder_hidden_states: torch.Tensor
    masks: torch.Tensor
    encoder_final_state: torch.Tensor
    code_hidden_states: torch.Tensor
    code_masks: torch.Tensor
    old_nl_hidden_states: torch.Tensor
    old_nl_masks: torch.Tensor
    old_nl_final_state: torch.Tensor
    attended_old_nl_final_state: torch.Tensor

class Example(NamedTuple):
    """Data format for examples used in generation model."""
    id: str
    old_comment: str
    old_comment_tokens: List[str]
    new_comment: str
    new_comment_tokens: List[str]
    old_code: str
    old_code_tokens: List[str]
    new_code: str
    new_code_tokens: List[str]

class DiffExample(NamedTuple):
    id: str
    label: int
    comment_type: str
    old_comment_raw: str
    old_comment_subtokens: List[str]
    new_comment_raw: str
    new_comment_subtokens: List[str]
    span_minimal_diff_comment_subtokens: List[str]
    old_code_raw: str
    old_code_subtokens: List[str]
    new_code_raw: str
    new_code_subtokens: List[str]
    span_diff_code_subtokens: List[str]
    token_diff_code_subtokens: List[str]

class DiffASTExample(NamedTuple):
    id: str
    label: int
    comment_type: str
    old_comment_raw: str
    old_comment_subtokens: List[str]
    new_comment_raw: str
    new_comment_subtokens: List[str]
    span_minimal_diff_comment_subtokens: List[str]
    old_code_raw: str
    old_code_subtokens: List[str]
    new_code_raw: str
    new_code_subtokens: List[str]
    span_diff_code_subtokens: List[str]
    token_diff_code_subtokens: List[str]
    old_ast: DiffAST
    new_ast: DiffAST
    diff_ast: DiffAST

def get_processed_comment_sequence(comment_subtokens):
    """Returns sequence without tag string. Tag strings are excluded for evaluation purposes."""
    if len(comment_subtokens) > 0 and comment_subtokens[0] in ['@param', '@return']:
        return comment_subtokens[1:]
    
    return comment_subtokens

def get_processed_comment_str(comment_subtokens):
    """Returns string without tag string. Tag strings are excluded for evaluation purposes."""
    return ' '.join(get_processed_comment_sequence(comment_subtokens))

def read_full_examples_from_file(filename):
    """Reads in data in the format used for generation model."""
    with open(filename) as f:
        data = json.load(f)
    return [Example(**d) for d in data]