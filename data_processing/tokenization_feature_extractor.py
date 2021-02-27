import difflib
import javalang
import json
import os
import re
import sys

from build_example import build_test_example
from data_formatting_utils import subtokenize_code, tokenize_clean_code, get_clean_code,\
subtokenize_comment, tokenize_comment

sys.path.append('../')
from diff_utils import is_edit_keyword, KEEP, KEEP_END, REPLACE_OLD, REPLACE_NEW,\
REPLACE_END, INSERT, INSERT_END, DELETE, DELETE_END, compute_code_diffs

def subtokenize_token(token, parse_comment=False):
    if parse_comment and token in ['@return', '@param', '@throws']:
        return [token]
    if is_edit_keyword(token):
        return [token]
    curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
    
    try:
        new_curr = []
        for t in curr:
            new_curr.extend([c for c in re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", t.encode('ascii', errors='ignore').decode().strip()) if len(c) > 0])
        curr = new_curr
    except:
        pass
    try:
        new_curr = []
        for c in curr:
            by_symbol = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip())
            new_curr = new_curr + by_symbol

        curr = new_curr
    except:
        curr = []
    subtokens = [c.lower() for c in curr]

    return subtokens

def get_subtoken_labels(gold_subtokens, tokens, parse_comment=False):
    labels = []
    indices = []
    all_subtokens = []

    token_map = []
    subtoken_map = []

    gold_idx = 0

    for token in tokens:
        subtokens = subtokenize_token(token, parse_comment)
        all_subtokens.extend(subtokens)
        token_map.append(subtokens)
        if len(subtokens) == 1:
            label = 0
            labels.append(label)
            indices.append(0)
            subtoken_map.append([token])
        else:
            label = 1
            for s, subtoken in enumerate(subtokens):
                labels.append(label)
                indices.append(s)
                subtoken_map.append([token])
    try:
        assert len(labels) == len(gold_subtokens)
        assert len(indices) == len(gold_subtokens)
        assert len(token_map) == len(tokens)
        assert len(subtoken_map) == len(gold_subtokens)
    except:
        print(tokens)
        print('\n')
        print(gold_subtokens)
        print('\n')
        for s, subtoken in enumerate(all_subtokens):
            print('Parsed: {}'.format(subtoken))
            print('True: {}'.format(gold_subtokens[s]))
            print('---------------------------------')
            if subtoken != gold_subtokens[s]:
                break
        print(len(labels))
        print(len(gold_subtokens))
        raise ValueError('stop')
    return labels, indices, token_map, subtoken_map

def get_code_subtoken_labels(gold_subtokens, tokens, raw_code):
    labels = []
    indices = []
    all_subtokens = []

    token_map = []
    subtoken_map = []

    for token in tokens:
        if is_edit_keyword(token):
            token_map.append([token])
        else:
            curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
            new_curr = []
            for c in curr:
                by_symbol = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip())
                new_curr = new_curr + by_symbol
            token_map.append([s.lower() for s in new_curr])

    try:
        parsed_tokens = get_clean_code(list(javalang.tokenizer.tokenize(raw_code)))
    except:
        parsed_tokens = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", raw_code.strip())
    
    subtokens = []
    for t, token in enumerate(parsed_tokens):
        curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
        subtokens = [c.lower() for c in curr]
        all_subtokens.extend(subtokens)
        if len(subtokens) == 1:
            label = 0
            labels.append(label)
            indices.append(0)
            subtoken_map.append([token])
        else:
            label = 1
            for s, subtoken in enumerate(subtokens):
                labels.append(label)
                indices.append(s)
                subtoken_map.append([token])
    try:
        assert len(labels) == len(gold_subtokens)
        assert len(indices) == len(gold_subtokens)
        assert len(token_map) == len(tokens)
        assert len(subtoken_map) == len(gold_subtokens)
    except:
        print(tokens)
        print('\n')
        print(gold_subtokens)
        print('\n')
        for s, subtoken in enumerate(all_subtokens):
            print('Parsed: {}'.format(subtoken))
            print('True: {}'.format(gold_subtokens[s]))
            print('---------------------------------')
            if subtoken != gold_subtokens[s]:
                break
        print(len(labels))
        print(len(gold_subtokens))
        raise ValueError('stop')
    return labels, indices, token_map, subtoken_map

def get_diff_subtoken_labels(diff_subtokens, old_subtokens, old_tokens, new_subtokens, new_tokens, diff_tokens, old_code_raw, new_code_raw):
    old_labels, old_indices, old_token_map, old_subtoken_map = get_code_subtoken_labels(old_subtokens, old_tokens, old_code_raw)
    new_labels, new_indices, new_token_map, new_subtoken_map = get_code_subtoken_labels(new_subtokens, new_tokens, new_code_raw)

    diff_labels = []
    diff_indices = []

    diff_token_map = []
    diff_subtoken_map = []

    for token in diff_tokens:
        if is_edit_keyword(token):
            diff_token_map.append([token])
        else:
            curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
            new_curr = []
            for c in curr:
                by_symbol = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip())
                new_curr = new_curr + by_symbol
            diff_token_map.append([s.lower() for s in new_curr])

    for edit_type, o_start, o_end, n_start, n_end in difflib.SequenceMatcher(None, old_subtokens, new_subtokens).get_opcodes():
        if edit_type == 'equal':
            diff_labels.extend([0] + old_labels[o_start:o_end] + [0])
            diff_indices.extend([0] + old_indices[o_start:o_end] + [0])
            diff_subtoken_map.append([KEEP])
            diff_subtoken_map.extend(old_subtoken_map[o_start:o_end])
            diff_subtoken_map.append([KEEP_END])
        elif edit_type == 'replace':
            diff_labels.extend([0] + old_labels[o_start:o_end] + [0] + new_labels[n_start:n_end] + [0])
            diff_indices.extend([0] + old_indices[o_start:o_end] + [0] + new_indices[n_start:n_end] + [0])
            diff_subtoken_map.append([REPLACE_OLD])
            diff_subtoken_map.extend(old_subtoken_map[o_start:o_end])
            diff_subtoken_map.append([REPLACE_NEW])
            diff_subtoken_map.extend(new_subtoken_map[n_start:n_end])
            diff_subtoken_map.append([REPLACE_END])
        elif edit_type == 'insert':
            diff_labels.extend([0] + new_labels[n_start:n_end] + [0])
            diff_indices.extend([0] + new_indices[n_start:n_end] + [0])
            diff_subtoken_map.append([INSERT])
            diff_subtoken_map.extend(new_subtoken_map[n_start:n_end])
            diff_subtoken_map.append([INSERT_END])
        else:
            diff_labels.extend([0] + old_labels[o_start:o_end] + [0])
            diff_indices.extend([0] + old_indices[o_start:o_end] + [0])
            diff_subtoken_map.append([DELETE])
            diff_subtoken_map.extend(old_subtoken_map[o_start:o_end])
            diff_subtoken_map.append([DELETE_END])
    
    assert len(diff_labels) == len(diff_subtokens)
    assert len(diff_indices) == len(diff_subtokens)
    assert len(diff_subtoken_map) == len(diff_subtokens)
    assert len(diff_token_map) == len(diff_tokens)
    return diff_labels, diff_indices, diff_token_map, diff_subtoken_map

if __name__ == "__main__":
    # Demo for extracting tokenization features for one example
    # Corresponds to what is written in tokenization_features.json files
    ex = build_test_example()

    old_code_tokens = tokenize_clean_code(ex.old_code_raw).split()
    new_code_tokens = tokenize_clean_code(ex.new_code_raw).split()
    span_diff_code_tokens, _, _ = compute_code_diffs(old_code_tokens, new_code_tokens)

    edit_span_subtoken_labels, edit_span_subtoken_indices, edit_span_token_map, edit_span_subtoken_map = get_diff_subtoken_labels(
        ex.span_diff_code_subtokens, ex.old_code_subtokens, old_code_tokens, ex.new_code_subtokens, new_code_tokens,
        span_diff_code_tokens, ex.old_code_raw, ex.new_code_raw)
        
    old_comment_tokens = tokenize_comment(ex.old_comment_raw).split()

    prefix = []
    if ex.comment_type == 'Return':
        prefix = ['@return']
    elif ex.comment_type == 'Param':
        prefix = ['@param']
    
    old_nl_subtoken_labels, old_nl_subtoken_indices, old_nl_token_map, old_nl_subtoken_map = get_subtoken_labels(
        prefix + ex.old_comment_subtokens, prefix + old_comment_tokens, parse_comment=True)

    cache = dict()
    cache[ex.id] = {
        'old_nl_subtoken_labels': old_nl_subtoken_labels,
        'old_nl_subtoken_indices': old_nl_subtoken_indices,
        'edit_span_subtoken_labels': edit_span_subtoken_labels,
        'edit_span_subtoken_indices': edit_span_subtoken_indices,
        'old_nl_token_map': old_nl_token_map,
        'old_nl_subtoken_map': old_nl_subtoken_map,
        'edit_span_token_map': edit_span_token_map,
        'edit_span_subtoken_map': edit_span_subtoken_map
    }