import argparse
from collections import Counter
import numpy as np
import os
import random
import sys
import torch
from torch import nn

from dpu_utils.mlutils import Vocabulary

from ast_graph_encoder import ASTGraphEncoder
from constants import *
from data_utils import *
import diff_utils
from embedding_store import EmbeddingStore
from encoder import Encoder
from external_cache import get_code_features, get_nl_features, get_num_code_features, get_num_nl_features
from tensor_utils import *


class ModuleManager(nn.Module):
    """Utility class which helps manage related attributes of the update and detection tasks."""
    def __init__(self, attend_code_sequence_states, attend_code_graph_states, features, posthoc, task):
        super(ModuleManager, self).__init__()
        self.attend_code_sequence_states = attend_code_sequence_states
        self.attend_code_graph_states = attend_code_graph_states
        self.features = features
        self.posthoc = posthoc
        self.task = task

        self.num_encoders = 0
        self.num_seq_encoders = 0
        self.out_dim = 0
        self.attention_state_size = 0
        self.update_encoder_state_size = 0
        self.max_ast_length = 0
        self.max_code_length = 0
        self.max_nl_length = 0
        self.generate = task in ['update', 'dual']
        self.classify = task in ['detect', 'dual']

        self.encode_code_sequence = self.generate or self.attend_code_sequence_states

        print('Attend code sequence states: {}'.format(self.attend_code_sequence_states))
        print('Attend code graph states: {}'.format(self.attend_code_graph_states))
        print('Features: {}'.format(self.features))
        print('Task: {}'.format(self.task))
        sys.stdout.flush()
    
    def get_code_representation(self, ex, data_type):
        if self.posthoc:
            if data_type == 'sequence':
                return ex.new_code_subtokens
            else:
                return ex.new_ast
        else:
            if data_type == 'sequence':
                return ex.span_diff_code_subtokens
            else:
                return ex.diff_ast

    def initialize(self, train_data):
        """Initializes model parameters from pre-defined hyperparameters and other hyperparameters
           that are computed based on statistics over the training data."""
        nl_lengths = []
        code_lengths = []
        ast_lengths = []

        nl_token_counter = Counter()
        code_token_counter = Counter()

        for ex in train_data:
            if self.generate: 
                trg_sequence = [START] + ex.span_minimal_diff_comment_subtokens + [END]
                nl_token_counter.update(trg_sequence)
                nl_lengths.append(len(trg_sequence))

            old_nl_sequence = ex.old_comment_subtokens
            nl_token_counter.update(old_nl_sequence)
            nl_lengths.append(len(old_nl_sequence))

            if self.encode_code_sequence:
                code_sequence = self.get_code_representation(ex, 'sequence')
                code_token_counter.update(code_sequence)
                code_lengths.append(len(code_sequence))
            
            if self.attend_code_graph_states:
                code_sequence = [n.value for n in self.get_code_representation(ex, 'graph').nodes]
                code_token_counter.update(code_sequence)
                ast_lengths.append(len(code_sequence))
        
        self.max_nl_length = int(np.percentile(np.asarray(sorted(nl_lengths)),
            LENGTH_CUTOFF_PCT))
        self.max_vocab_extension = self.max_nl_length

        if self.encode_code_sequence:
            self.max_code_length = int(np.percentile(np.asarray(sorted(code_lengths)),
                LENGTH_CUTOFF_PCT))
            self.max_vocab_extension += self.max_code_length
        
        if self.attend_code_graph_states:
            self.max_ast_length = int(np.percentile(np.asarray(sorted(ast_lengths)),
                LENGTH_CUTOFF_PCT))
    
        nl_counts = np.asarray(sorted(nl_token_counter.values()))
        nl_threshold = int(np.percentile(nl_counts, VOCAB_CUTOFF_PCT)) + 1
        code_counts = np.asarray(sorted(code_token_counter.values()))
        code_threshold = int(np.percentile(nl_counts, VOCAB_CUTOFF_PCT)) + 1

        self.embedding_store = EmbeddingStore(nl_threshold, NL_EMBEDDING_SIZE, nl_token_counter,
            code_threshold, CODE_EMBEDDING_SIZE, code_token_counter,
            DROPOUT_RATE, len(SrcType), SRC_EMBEDDING_SIZE, CODE_EMBEDDING_SIZE, True)
        
        self.out_dim = 2*HIDDEN_SIZE
        
        # Accounting for the old NL encoder
        self.num_encoders = 1
        self.num_seq_encoders += 1
        self.attention_state_size += 2*HIDDEN_SIZE
        self.nl_encoder = Encoder(NL_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
        self.nl_attention_transform_matrix = nn.Parameter(torch.randn(
            self.out_dim, self.out_dim, dtype=torch.float, requires_grad=True))
        self.self_attention = nn.MultiheadAttention(self.out_dim, MULTI_HEADS, DROPOUT_RATE)

        if self.encode_code_sequence:
            self.sequence_code_encoder = Encoder(CODE_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
            self.num_encoders += 1
            self.num_seq_encoders += 1
                
            if self.attend_code_sequence_states:
                self.attention_state_size += 2*HIDDEN_SIZE
                self.sequence_attention_transform_matrix = nn.Parameter(torch.randn(
                    self.out_dim, self.out_dim, dtype=torch.float, requires_grad=True))
                self.code_sequence_multihead_attention = nn.MultiheadAttention(self.out_dim, MULTI_HEADS, DROPOUT_RATE)
        
        if self.attend_code_graph_states:
            self.graph_code_encoder = ASTGraphEncoder(CODE_EMBEDDING_SIZE, len(DiffEdgeType))
            self.num_encoders += 1
            self.attention_state_size += 2*HIDDEN_SIZE
            self.graph_attention_transform_matrix = nn.Parameter(torch.randn(
                CODE_EMBEDDING_SIZE, self.out_dim, dtype=torch.float, requires_grad=True))
            self.graph_multihead_attention = nn.MultiheadAttention(self.out_dim, MULTI_HEADS, DROPOUT_RATE)

        if self.features:
            self.code_features_to_embedding = nn.Linear(CODE_EMBEDDING_SIZE + get_num_code_features(),
                CODE_EMBEDDING_SIZE, bias=False)
            self.nl_features_to_embedding = nn.Linear(
                NL_EMBEDDING_SIZE + get_num_nl_features(),
                NL_EMBEDDING_SIZE, bias=False)
        
        if self.generate:
            self.update_encoder_state_size = self.num_seq_encoders*self.out_dim
            self.encoder_final_to_decoder_initial = nn.Parameter(torch.randn(self.update_encoder_state_size,
                self.out_dim, dtype=torch.float, requires_grad=True))
        
        if self.classify:
            self.attended_nl_encoder = Encoder(self.out_dim, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
            self.attended_nl_encoder_output_layer = nn.Linear(self.attention_state_size, self.out_dim, bias=False)

    def get_batches(self, dataset, device, shuffle=False):
        """Divides the dataset into batches based on pre-defined BATCH_SIZE hyperparameter.
           Each batch is tensorized so that it can be directly passed into the network."""
        batches = []
        if shuffle:
            random.shuffle(dataset)
        
        curr_idx = 0
        while curr_idx < len(dataset):
            start_idx = curr_idx
            end_idx = min(start_idx + BATCH_SIZE, len(dataset))
            
            code_token_ids = []
            code_lengths = []
            old_nl_token_ids = []
            old_nl_lengths = []
            trg_token_ids = []
            trg_extended_token_ids = []
            trg_lengths = []
            invalid_copy_positions = []
            inp_str_reps = []
            inp_ids = []
            code_features = []
            nl_features = []
            labels = []

            graph_batch = initialize_graph_method_batch(len(DiffEdgeType))

            for i in range(start_idx, end_idx):
                if self.encode_code_sequence:
                    code_sequence = self.get_code_representation(dataset[i], 'sequence')
                    code_sequence_ids = self.embedding_store.get_padded_code_ids(
                        code_sequence, self.max_code_length)
                    code_length = min(len(code_sequence), self.max_code_length)
                    code_token_ids.append(code_sequence_ids)
                    code_lengths.append(code_length)
                
                if self.attend_code_graph_states:
                    ast = self.get_code_representation(dataset[i], 'graph')
                    ast_sequence = [n.value for n in ast.nodes]
                    ast_length = min(len(ast_sequence), self.max_ast_length)
                    ast.nodes = ast.nodes[:ast_length]
                    graph_batch = insert_graph(graph_batch, dataset[i], ast,
                        self.embedding_store.code_vocabulary, self.features, self.max_ast_length)

                old_nl_sequence = dataset[i].old_comment_subtokens
                old_nl_length = min(len(old_nl_sequence), self.max_nl_length)
                old_nl_sequence_ids = self.embedding_store.get_padded_nl_ids(
                    old_nl_sequence, self.max_nl_length)
                
                old_nl_token_ids.append(old_nl_sequence_ids)
                old_nl_lengths.append(old_nl_length)
                
                if self.generate:
                    ex_inp_str_reps = []
                    ex_inp_ids = []
                    
                    extra_counter = len(self.embedding_store.nl_vocabulary)
                    max_limit = len(self.embedding_store.nl_vocabulary) + self.max_vocab_extension
                    out_ids = set()

                    copy_inputs = []
                    copy_inputs += code_sequence[:code_length]
                    
                    copy_inputs += old_nl_sequence[:old_nl_length]
                    for c in copy_inputs:
                        nl_id = self.embedding_store.get_nl_id(c)
                        if self.embedding_store.is_nl_unk(nl_id) and extra_counter < max_limit:
                            if c in ex_inp_str_reps:
                                nl_id = ex_inp_ids[ex_inp_str_reps.index(c)]
                            else:
                                nl_id = extra_counter
                                extra_counter += 1

                        out_ids.add(nl_id)
                        ex_inp_str_reps.append(c)
                        ex_inp_ids.append(nl_id)
                
                    trg_sequence = trg_sequence = [START] + dataset[i].span_minimal_diff_comment_subtokens + [END]
                    trg_sequence_ids = self.embedding_store.get_padded_nl_ids(
                        trg_sequence, self.max_nl_length)
                    trg_extended_sequence_ids = self.embedding_store.get_extended_padded_nl_ids(
                        trg_sequence, self.max_nl_length, ex_inp_ids, ex_inp_str_reps)
                    
                    trg_token_ids.append(trg_sequence_ids)
                    trg_extended_token_ids.append(trg_extended_sequence_ids)
                    trg_lengths.append(min(len(trg_sequence), self.max_nl_length))
                    inp_str_reps.append(ex_inp_str_reps)
                    inp_ids.append(ex_inp_ids)

                    invalid_copy_positions.append(get_invalid_copy_locations(ex_inp_str_reps, self.max_vocab_extension,
                        trg_sequence, self.max_nl_length))

                labels.append(dataset[i].label)

                if self.features:
                    if self.encode_code_sequence:
                        code_features.append(get_code_features(code_sequence, dataset[i], self.max_code_length))
                    nl_features.append(get_nl_features(old_nl_sequence, dataset[i], self.max_nl_length))
                
            batches.append(UpdateBatchData(torch.tensor(code_token_ids, dtype=torch.int64, device=device),
                                           torch.tensor(code_lengths, dtype=torch.int64, device=device),
                                           torch.tensor(old_nl_token_ids, dtype=torch.int64, device=device),
                                           torch.tensor(old_nl_lengths, dtype=torch.int64, device=device),
                                           torch.tensor(trg_token_ids, dtype=torch.int64, device=device),
                                           torch.tensor(trg_extended_token_ids, dtype=torch.int64, device=device),
                                           torch.tensor(trg_lengths, dtype=torch.int64, device=device),
                                           torch.tensor(invalid_copy_positions, dtype=torch.uint8, device=device),
                                           inp_str_reps, inp_ids,
                                           torch.tensor(code_features, dtype=torch.float32, device=device),
                                           torch.tensor(nl_features, dtype=torch.float32, device=device),
                                           torch.tensor(labels, dtype=torch.int64, device=device),
                                           tensorize_graph_method_batch(graph_batch, device, self.max_ast_length)))
            curr_idx = end_idx
        return batches
    
    def get_encoder_output(self, batch_data, device):
        """Gets hidden states, final state, and a length masks corresponding to each encoder."""
        encoder_hidden_states = None
        input_lengths = None
        final_states = None
        mask = None

        # Encode old NL
        old_nl_embedded_subtokens = self.embedding_store.get_nl_embeddings(batch_data.old_nl_ids)
        if self.features:
            old_nl_embedded_subtokens = self.nl_features_to_embedding(torch.cat(
                [old_nl_embedded_subtokens, batch_data.nl_features], dim=-1))
        old_nl_hidden_states, old_nl_final_state = self.nl_encoder.forward(old_nl_embedded_subtokens,
            batch_data.old_nl_lengths, device)
        old_nl_masks = (torch.arange(
            old_nl_hidden_states.shape[1], device=device).view(1, -1) >= batch_data.old_nl_lengths.view(-1, 1)).unsqueeze(1)
        attention_states = compute_attention_states(old_nl_hidden_states, old_nl_masks,
            old_nl_hidden_states, transformation_matrix=self.nl_attention_transform_matrix, multihead_attention=self.self_attention)

        # Encode code
        code_hidden_states = None
        code_masks = None
        code_final_state = None

        if self.encode_code_sequence:
            code_embedded_subtokens = self.embedding_store.get_code_embeddings(batch_data.code_ids)
            if self.features:
                code_embedded_subtokens = self.code_features_to_embedding(torch.cat(
                    [code_embedded_subtokens, batch_data.code_features], dim=-1))
            code_hidden_states, code_final_state = self.sequence_code_encoder.forward(code_embedded_subtokens,
                batch_data.code_lengths, device)
            code_masks = (torch.arange(
                code_hidden_states.shape[1], device=device).view(1, -1) >= batch_data.code_lengths.view(-1, 1)).unsqueeze(1)
            encoder_hidden_states = code_hidden_states
            input_lengths = batch_data.code_lengths
            final_states = code_final_state

            if self.attend_code_sequence_states:
                attention_states = torch.cat([attention_states, compute_attention_states(
                    code_hidden_states, code_masks, old_nl_hidden_states,
                    transformation_matrix=self.sequence_attention_transform_matrix,
                    multihead_attention=self.code_sequence_multihead_attention)], dim=-1)

        if self.attend_code_graph_states:
            embedded_nodes = self.embedding_store.get_node_embeddings(
                batch_data.graph_batch.value_lookup_ids, batch_data.graph_batch.src_type_ids)
            
            if self.features:
                embedded_nodes = self.code_features_to_embedding(torch.cat(
                    [embedded_nodes, batch_data.graph_batch.node_features], dim=-1))
            
            graph_states = self.graph_code_encoder.forward(embedded_nodes, batch_data.graph_batch, device)
            graph_lengths = batch_data.graph_batch.num_nodes_per_graph
            graph_masks = (torch.arange(
                graph_states.shape[1], device=device).view(1, -1) >= graph_lengths.view(-1, 1)).unsqueeze(1)

            transformed_graph_states = torch.einsum('ijk,km->ijm', graph_states, self.graph_attention_transform_matrix)
            graph_attention_states = compute_attention_states(transformed_graph_states, graph_masks,
                old_nl_hidden_states, multihead_attention=self.graph_multihead_attention)
            attention_states = torch.cat([attention_states, graph_attention_states], dim=-1)      
                
        if self.classify:
            nl_attended_states = torch.tanh(self.attended_nl_encoder_output_layer(attention_states))
            _, attended_old_nl_final_state = self.attended_nl_encoder.forward(nl_attended_states,
                batch_data.old_nl_lengths, device)     
        else:
            attended_old_nl_final_state = None
        
        if self.generate:
            encoder_final_state = torch.einsum('ij,jk->ik',
                torch.cat([final_states, old_nl_final_state], dim=-1),
                self.encoder_final_to_decoder_initial)
            encoder_hidden_states, input_lengths = merge_encoder_outputs(encoder_hidden_states,
                input_lengths, old_nl_hidden_states, batch_data.old_nl_lengths, device)
            mask = (torch.arange(
                encoder_hidden_states.shape[1], device=device).view(1, -1) >= input_lengths.view(-1, 1)).unsqueeze(1)
        else:
            encoder_final_state = None
        
        return EncoderOutputs(encoder_hidden_states, mask, encoder_final_state, code_hidden_states, code_masks,
                              old_nl_hidden_states, old_nl_masks, old_nl_final_state, attended_old_nl_final_state)