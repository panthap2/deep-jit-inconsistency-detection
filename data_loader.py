import json
import os

from constants import DATA_PATH
from data_utils import DiffAST, DiffExample, DiffASTExample, CommentCategory

PARTITIONS = ['train', 'valid', 'test']

def get_data_splits(comment_type_str=None, ignore_ast=False):
    """Retrieves train/validation/test sets for the given comment_type_str.
       comment_type_str -- Return, Param, Summary, or None (if None, uses all comment types)
       ignore_ast -- Skip loading ASTs (they take a long time)"""
    dataset, high_level_details = load_processed_data(comment_type_str, ignore_ast)
    train_examples = dataset['train']
    valid_examples = dataset['valid']
    test_examples = dataset['test']
    return train_examples, valid_examples, test_examples, high_level_details

def load_cleaned_test_set(comment_type_str=None):
    """Retrieves the ids corresponding to clean examples, for the given comment_type_str.
       comment_type_str -- Return, Param, Summary, or None (if None, uses all comment types)"""
    if not comment_type_str:
        comment_types = [CommentCategory(category).name for category in CommentCategory]
    else:
        comment_types = [comment_type_str]
    
    test_ids = []
    for comment_type in comment_types:
        resources_path  = os.path.join(DATA_PATH, 'resources', comment_type, 'clean_test_ids.json')
        with open(resources_path) as f:
            test_ids.extend(json.load(f))
    return test_ids

def load_processed_data(comment_type_str, ignore_ast):
    """Processes saved data for the given comment_type_str.
       comment_type_str -- Return, Param, Summary, or None (if None, uses all comment types)
       ignore_ast -- Skip loading ASTs (they take a long time)"""
    if not comment_type_str:
        comment_types = [CommentCategory(category).name for category in CommentCategory]
    else:
        comment_types = [comment_type_str]
    
    print('Loading data from: {}'.format(comment_types))
    
    dataset = dict()
    high_level_details = dict()
    for comment_type in comment_types:
        path = os.path.join(DATA_PATH, comment_type)
        loaded = load_raw_data_from_path(path)
        category_high_level_details_path = os.path.join(DATA_PATH, 'resources', comment_type, 'high_level_details.json')

        with open(category_high_level_details_path) as f:
            category_high_level_details = json.load(f)
        high_level_details.update(category_high_level_details)

        if not ignore_ast:
            ast_path  = os.path.join(DATA_PATH, 'resources', comment_type, 'ast_objs.json')
            with open(ast_path) as f:
                ast_details = json.load(f)

        for partition, examples in loaded.items():
            if partition not in dataset:
                dataset[partition] = []
            
            if ignore_ast:
                dataset[partition].extend(examples)
            else:
                for ex in examples:
                    ex_ast_info = ast_details[ex.id]
                    old_ast = DiffAST.from_json(ex_ast_info['old_ast'])
                    new_ast = DiffAST.from_json(ex_ast_info['new_ast'])
                    diff_ast = DiffAST.from_json(ex_ast_info['diff_ast'])

                    ast_ex = DiffASTExample(ex.id, ex.label, ex.comment_type, ex.old_comment_raw,
                        ex.old_comment_subtokens, ex.new_comment_raw, ex.new_comment_subtokens, ex.span_minimal_diff_comment_subtokens,
                        ex.old_code_raw, ex.old_code_subtokens, ex.new_code_raw, ex.new_code_subtokens,
                        ex.span_diff_code_subtokens, ex.token_diff_code_subtokens, old_ast, new_ast, diff_ast)
                    
                    dataset[partition].append(ast_ex)

    return dataset, high_level_details

def load_raw_data_from_path(path):
    """Reads saved partition-level data from a directory path"""
    dataset = dict()

    for partition in PARTITIONS:
        dataset[partition] = []
        dataset[partition].extend(read_diff_examples_from_file(os.path.join(path, '{}.json'.format(partition))))

    return dataset

def read_diff_examples_from_file(filename):
    """Reads saved data from filename"""
    with open(filename) as f:
        data = json.load(f)
    return [DiffExample(**d) for d in data]