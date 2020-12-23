import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import numpy as np
import os
import re

from constants import *
from diff_utils import *

method_details = dict()
tokenization_features = dict()
for d in os.listdir(RESOURCES_PATH):
    try:
        with open(os.path.join(RESOURCES_PATH, d, 'high_level_details.json')) as f:
            method_details.update(json.load(f))
        with open(os.path.join(RESOURCES_PATH, d, 'tokenization_features.json')) as f:
            tokenization_features.update(json.load(f))
    except:
        print('Failed parsing: {}'.format(d))

stop_words = set(stopwords.words('english'))
java_keywords = set(['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class',
         'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally',
         'float', 'for', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long',
         'native', 'new', 'null', 'package', 'private', 'protected', 'public', 'return', 'short',
         'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient',
         'try', 'void', 'volatile', 'while'])

tags = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT',
'POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB',
'OTHER']

NUM_CODE_FEATURES = 19
NUM_NL_FEATURES = 17 + len(tags)

def get_num_code_features():
    return NUM_CODE_FEATURES

def get_num_nl_features():
    return NUM_NL_FEATURES

def is_java_keyword(token):
    return token in java_keywords

def is_operator(token):
    for s in token:
        if s.isalnum():
            return False
    return True

def get_return_type_subtokens(example):
    return method_details[example.id]['new']['subtoken']['return_type']

def get_old_return_type_subtokens(example):
    return method_details[example.id]['old']['subtoken']['return_type']

def get_method_name_subtokens(example):
    return method_details[example.id]['new']['subtoken']['method_name']

def get_new_return_sequence(example):
    return method_details[example.id]['new']['subtoken']['return_statement']

def get_old_return_sequence(example):
    return method_details[example.id]['old']['subtoken']['return_statement']

def get_old_argument_type_subtokens(example):
    return method_details[example.id]['old']['subtoken']['argument_type']

def get_new_argument_type_subtokens(example):
    return method_details[example.id]['new']['subtoken']['argument_type']

def get_old_argument_name_subtokens(example):
    return method_details[example.id]['old']['subtoken']['argument_name']

def get_new_argument_name_subtokens(example):
    return method_details[example.id]['new']['subtoken']['argument_name']

def get_old_code(example):
    return example.old_code_raw

def get_new_code(example):
    return example.new_code_raw

def get_edit_span_subtoken_tokenization_labels(example):
    return tokenization_features[example.id]['edit_span_subtoken_labels']

def get_edit_span_subtoken_tokenization_indices(example):
    return tokenization_features[example.id]['edit_span_subtoken_indices']

def get_nl_subtoken_tokenization_labels(example):
    return tokenization_features[example.id]['old_nl_subtoken_labels']

def get_nl_subtoken_tokenization_indices(example):
    return tokenization_features[example.id]['old_nl_subtoken_indices']

def get_node_features(nodes, example, max_ast_length):
    old_return_type_subtokens = get_old_return_type_subtokens(example)
    new_return_type_subtokens = get_return_type_subtokens(example)
    method_name_subtokens = get_method_name_subtokens(example)

    old_return_sequence = get_old_return_sequence(example)
    new_return_sequence = get_new_return_sequence(example)

    old_return_line_terms = set([t for t in old_return_sequence if not is_java_keyword(t) and not is_operator(t)])
    new_return_line_terms = set([t for t in new_return_sequence if not is_java_keyword(t) and not is_operator(t)])
    return_line_intersection = old_return_line_terms.intersection(new_return_line_terms)

    old_set = set(old_return_type_subtokens)
    new_set = set(new_return_type_subtokens)

    intersection = old_set.intersection(new_set)

    features = np.zeros((len(nodes), get_num_code_features()), dtype=np.int64)

    old_nl_tokens = set(example.old_comment_subtokens)
    last_command = None

    for i, node in enumerate(nodes):
        if not node.is_leaf:
            continue
        
        token = node.value

        if token in intersection:
            features[i][0] = True
        elif token in old_set:
            features[i][1] = True
        elif token in new_set:
            features[i][2] = True
        else:
            features[i][3] = True

        if token in return_line_intersection:
            features[i][4] = True
        elif token in old_return_line_terms:
            features[i][5] = True
        elif token in new_return_line_terms:
            features[i][6] = True
        else:
            features[i][7] = True
        
        if is_edit_keyword(token):
            features[i][8] = True
        if is_java_keyword(token):
            features[i][9] = True
        if is_operator(token):
            features[i][10] = True
        if token in old_nl_tokens:
            features[i][11] = True
        
        if not is_edit_keyword(token):
            if last_command == KEEP:
                features[i][12] = 1
            elif last_command == INSERT:
                features[i][13] = 1
            elif last_command == DELETE:
                features[i][14] = 1
            elif last_command == REPLACE_NEW:
                features[i][15] = 1
            else:
                features[i][16] = 1
        else:
            last_command = token
        
        if len(node.subtoken_children) > 0 or len(node.subtoken_parents) > 0:
            features[i][17] = True
        
        if len(node.subtoken_parents) == 1:
            features[i][18] = node.subtoken_parents[0].subtoken_children.index(node)

    return features.astype(np.float32)

def get_code_features(code_sequence, example, max_code_length):
    old_return_type_subtokens = get_old_return_type_subtokens(example)
    new_return_type_subtokens = get_return_type_subtokens(example)
    method_name_subtokens = get_method_name_subtokens(example)

    old_return_sequence = get_old_return_sequence(example)
    new_return_sequence = get_new_return_sequence(example)

    old_return_line_terms = set([t for t in old_return_sequence if not is_java_keyword(t) and not is_operator(t)])
    new_return_line_terms = set([t for t in new_return_sequence if not is_java_keyword(t) and not is_operator(t)])
    return_line_intersection = old_return_line_terms.intersection(new_return_line_terms)

    old_set = set(old_return_type_subtokens)
    new_set = set(new_return_type_subtokens)

    intersection = old_set.intersection(new_set)

    features = np.zeros((max_code_length, get_num_code_features()), dtype=np.int64)

    old_nl_tokens = set(example.old_comment_subtokens)
    last_command = None

    subtoken_labels = get_edit_span_subtoken_tokenization_labels(example)
    subtoken_indices = get_edit_span_subtoken_tokenization_indices(example)

    for i, token in enumerate(code_sequence):
        if i >= max_code_length:
            break
        if token in intersection:
            features[i][0] = True
        elif token in old_set:
            features[i][1] = True
        elif token in new_set:
            features[i][2] = True
        else:
            features[i][3] = True

        if token in return_line_intersection:
            features[i][4] = True
        elif token in old_return_line_terms:
            features[i][5] = True
        elif token in new_return_line_terms:
            features[i][6] = True
        else:
            features[i][7] = True
        
        if is_edit_keyword(token):
            features[i][8] = True
        if is_java_keyword(token):
            features[i][9] = True
        if is_operator(token):
            features[i][10] = True
        if token in old_nl_tokens:
            features[i][11] = True
        
        if not is_edit_keyword(token):
            if last_command == KEEP:
                features[i][12] = 1
            elif last_command == INSERT:
                features[i][13] = 1
            elif last_command == DELETE:
                features[i][14] = 1
            elif last_command == REPLACE_NEW:
                features[i][15] = 1
            else:
                features[i][16] = 1
        else:
            last_command = token
        
        features[i][17] = subtoken_labels[i]
        features[i][18] = subtoken_indices[i]

    return features.astype(np.float32)

def get_nl_features(old_nl_sequence, example, max_nl_length):
    insert_code_tokens = set()
    keep_code_tokens = set()
    delete_code_tokens = set()
    replace_old_code_tokens = set()
    replace_new_code_tokens = set()

    frequency_map = dict()
    for tok in old_nl_sequence:
        if tok not in frequency_map:
            frequency_map[tok] = 0
        frequency_map[tok] += 1
    
    pos_tags = pos_tag(word_tokenize(' '.join(old_nl_sequence)))
    pos_tag_indices = []
    for _, t in pos_tags:
        if t in tags:
            pos_tag_indices.append(tags.index(t))
        else:
            pos_tag_indices.append(tags.index('OTHER'))

    i = 0
    code_tokens = example.token_diff_code_subtokens

    while i < len(code_tokens):
        if code_tokens[i] == INSERT:
            insert_code_tokens.add(code_tokens[i+1].lower())
            i += 2
        elif code_tokens[i] == KEEP:
            keep_code_tokens.add(code_tokens[i+1].lower())
            i += 2
        elif code_tokens[i] == DELETE:
            delete_code_tokens.add(code_tokens[i+1].lower())
            i += 2
        elif code_tokens[i] == REPLACE_OLD:
            replace_old_code_tokens.add(code_tokens[i+1].lower())
            i += 2
        elif code_tokens[i] == REPLACE_NEW:
            replace_new_code_tokens.add(code_tokens[i+1].lower())
            i += 2
    
    old_return_type_subtokens = get_old_return_type_subtokens(example)
    new_return_type_subtokens = get_return_type_subtokens(example)

    old_return_sequence = get_old_return_sequence(example)
    new_return_sequence = get_new_return_sequence(example)

    old_return_line_terms = set([t for t in old_return_sequence if not is_java_keyword(t) and not is_operator(t)])
    new_return_line_terms = set([t for t in new_return_sequence if not is_java_keyword(t) and not is_operator(t)])
    return_line_intersection = old_return_line_terms.intersection(new_return_line_terms)

    old_set = set(old_return_type_subtokens)
    new_set = set(new_return_type_subtokens)

    intersection = old_set.intersection(new_set)

    method_name_subtokens = method_name_subtokens = get_method_name_subtokens(example)

    nl_subtoken_labels = get_nl_subtoken_tokenization_labels(example)
    nl_subtoken_indices = get_nl_subtoken_tokenization_indices(example)
        
    features = np.zeros((max_nl_length, get_num_nl_features()), dtype=np.int64)
    for i in range(len(old_nl_sequence)):
        if i >= max_nl_length:
            break
        token = old_nl_sequence[i].lower()
        if token in intersection:
            features[i][0] = True
        elif token in old_set:
            features[i][1] = True
        elif token in new_set:
            features[i][2] = True
        else:
            features[i][3] = True

        if token in return_line_intersection:
            features[i][4] = True
        elif token in old_return_line_terms:
            features[i][5] = True
        elif token in new_return_line_terms:
            features[i][6] = True
        else:
            features[i][7] = True
        
        features[i][8] = token in insert_code_tokens
        features[i][9] = token in keep_code_tokens
        features[i][10] = token in delete_code_tokens
        features[i][11] = token in replace_old_code_tokens
        features[i][12] = token in replace_new_code_tokens
        features[i][13] = token in stop_words
        features[i][14] = frequency_map[token] > 1

        features[i][15] = nl_subtoken_labels[i]
        features[i][16] = nl_subtoken_indices[i]
        features[i][17 + pos_tag_indices[i]] = 1

    return features.astype(np.float32)
