import json
import os
import re
import sys

from build_example import build_test_example
from data_formatting_utils import tokenize_clean_code, subtokenize_code

sys.path.append('../')
from diff_utils import is_edit_keyword, KEEP, DELETE, INSERT, REPLACE_OLD, REPLACE_NEW

EDIT_INDICES = [KEEP, DELETE, INSERT, REPLACE_OLD, REPLACE_NEW]

def extract_arguments(code_block):
    i = 0
    while i < len(code_block):
        line = code_block[i].strip()
        if len(line.strip()) == 0:
            i += 1
            continue
        if line[0] == '@' and ' ' not in line:
            i += 1
            continue
        if '//' in line or '*' in line:
            i += 1
            continue
        else:
            break

    argument_string =  line[line.index('(')+1:]

    if argument_string.count('(') + 1 == argument_string.count(')'):
        argument_string = argument_string[:argument_string.rfind(')')]
    else:
        curr_open_count = argument_string.count('(') + 1
        curr_close_count = argument_string.count(')')
        i += 1
        extension = ''
        while i < len(code_block):
            for w in code_block[i].strip():
                extension += w
                if w == '(':
                    curr_open_count += 1
                elif w == ')':
                    curr_close_count += 1
                if curr_open_count == curr_close_count:
                    break
            if curr_open_count == curr_close_count:
                break
            i += 1

        if curr_open_count != curr_close_count:  
            raise ValueError('Invalid arguments')

        argument_string = argument_string + extension[:-1]

    argument_types = []
    argument_names = []

    argument_string = ' '.join([a for a in argument_string.split() if '@' not in a])
    terms = []
    a = 0
    curr_term = []
    
    open_count = 0
    close_count = 0

    while a < len(argument_string):
        t = argument_string[a]
        if t == ' ' and open_count == close_count:
            terms.append(''.join(curr_term).strip())
            curr_term = []
            a += 1
            continue
        if t == ',' and open_count == close_count:
            curr_term.append(t)
            terms.append(''.join(curr_term).strip())
            curr_term = []
            a += 1
            continue
        
        if t == ',' and open_count != close_count:
            a += 1
            continue
        
        if t == '<':
            open_count += 1
        
        if t == '>':
            close_count += 1
        
        curr_term.append(t)
        a += 1
    
    if len(curr_term) > 0:
        terms.append(''.join(curr_term).strip())

    terms = [t for t in terms if t not in ['private', 'protected', 'public', 'final', 'static']]
    arguments = ' '.join(terms).split(',')
    arguments = [a.strip() for a in arguments if len(a.strip()) > 0]
    for argument in arguments:
        argument_tokens = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", argument.strip())
        argument_types.append(argument_tokens[0])
        argument_names.append(argument_tokens[-1])

    return argument_names, argument_types

def strip_comment(s):
    """Checks whether a single line follows the structure of a comment."""
    new_s = re.sub(r'\"(.+?)\"', '', s)
    matched_obj = re.findall("(?:/\\*(?:[^*]|(?:\\*+[^*/]))*\\*+/)|(?://.*)", new_s)
    url_match = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', new_s)
    file_match = re.findall('^(.*/)?(?:$|(.+?)(?:(\.[^.]*$)|$))', new_s)

    if matched_obj and not url_match:
        for m in matched_obj:
            s = s.replace(m, ' ')
    return s.strip()

def extract_return_statements(code_block):
    cleaned_lines = []
    for l in code_block:
        cleaned_l = strip_comment(l)
        if len(cleaned_l) > 0:
            cleaned_lines.append(cleaned_l)
 
    combined_block = ' '.join(cleaned_lines)
    if 'return' not in combined_block:
        return []
    indices = [m.start() for m in re.finditer('return ', combined_block)]
    return_statements = []
    for idx in indices:
        s_idx = idx + len('return ')
        e_idx = s_idx + combined_block[s_idx:].index(';')
        statement = combined_block[s_idx:e_idx].strip()
        if len(statement) > 0:
            return_statements.append(statement)

    return return_statements


def is_operator(token):
    for s in token:
        if s.isalnum():
            return False
    return True

def extract_method_name(code_block):
    i = 0
    while i < len(code_block):
        line = code_block[i].strip()
        if len(line.strip()) == 0:
            i += 1
            continue
        if line[0] == '@' and ' ' not in line:
            i += 1
            continue
        if '//' in line or '*' in line:
            i += 1
            continue
        else:
            break
    
    try:
        method_components = line.strip().split('(')[0].split(' ')
        method_components = [m for m in method_components if len(m) > 0]
        method_name = method_components[-1].strip()
    except:
        method_name = ''

    return method_name

def extract_return_type(code_block):
    i = 0
    while i < len(code_block):
        line = code_block[i].strip()
        if len(line.strip()) == 0:
            i += 1
            continue
        if line[0] == '@':
            i += 1
            continue
        if '//' in line or '*' in line:
            i += 1
            continue
        else:
            break
    
    before_method_name_tokens = line.split('(')[0].split(' ')[:-1]
    return_type_tokens = []
    for tok in before_method_name_tokens:
        if tok not in ['private', 'protected', 'public', 'final', 'static']:
            return_type_tokens.append(tok)
    return ' '.join(return_type_tokens)

def get_change_labels(tokens):
    cache = dict()
    for label in EDIT_INDICES:
        cache[label] = set()
    
    label = None
    for t in tokens:
        if is_edit_keyword(t):
            label = t
        elif is_operator(t):
            continue
        else:
            cache[label].add(t)

    for label, label_set in cache.items():
        cache[label] = list(label_set)
    return cache

def extract_throwable_exceptions(code_block):
    i = 0
    while i < len(code_block):
        line = code_block[i].strip()
        if 'throws' in line:
            break
        i += 1

    if 'throws' not in line:
        return []
    
    throws_string = line[line.index('throws') + len('throws'):]
    if '{' in throws_string:
        throws_string = throws_string[:throws_string.index('{')]
    else:
        extension = ''
        i += 1
        while i < len(code_block):
            line = code_block[i].strip()
            if len(line) == 0:
                i += 1
                continue
            for w in line:
                if w == '{':
                    break
                else:
                    extension += w
            if w == '{':
                break
            else:
                i += 1
        
        throws_string += extension
    
    exception_tokens = [t for t in tokenize_clean_code(throws_string).split() if not is_operator(t)]
    return exception_tokens

def extract_throw_statements(code_block):
    cleaned_lines = []
    for l in code_block:
        cleaned_l = strip_comment(l)
        if len(cleaned_l) > 0:
            cleaned_lines.append(cleaned_l)
    
    combined_block = ' '.join(cleaned_lines)
    if 'throw' not in combined_block:
        return []
    indices = [m.start() for m in re.finditer('throw ', combined_block)]
    throw_statements = []
    for idx in indices:
        s_idx = idx + len('throw ')
        e_idx = s_idx + combined_block[s_idx:].index(';')
        statement = combined_block[s_idx:e_idx].strip()
        if len(statement) > 0:
            throw_statements.append(statement)

    return throw_statements

def get_method_elements(code_block):
    argument_names, argument_types = extract_arguments(code_block)
    return_statements = extract_return_statements(code_block)
    return_type = extract_return_type(code_block)

    throwable_exception_tokens = extract_throwable_exceptions(code_block)
    throwable_exception_subtokens = []
    for throwable_exception in throwable_exception_tokens:
        throwable_exception_subtokens.extend(subtokenize_code(throwable_exception).split())

    throw_statements = extract_throw_statements(code_block)
    throw_statement_tokens = []
    throw_statement_subtokens = []
    for throw_statement in throw_statements:
        throw_statement_tokens.extend([t for t in tokenize_clean_code(throw_statement).split() if not is_operator(t)])
        throw_statement_subtokens.extend([t for t in subtokenize_code(throw_statement).split() if not is_operator(t)])

    argument_name_tokens = []
    argument_name_subtokens = []
    argument_type_tokens = []
    argument_type_subtokens = []

    for argument_name in argument_names:
        argument_name_tokens.extend([t for t in tokenize_clean_code(argument_name).split() if not is_operator(t)])
        argument_name_subtokens.extend([t for t in subtokenize_code(argument_name).split() if not is_operator(t)])

    for argument_type in argument_types:
        argument_type_tokens.extend([t for t in tokenize_clean_code(argument_type).split() if not is_operator(t)])
        argument_type_subtokens.extend([t for t in subtokenize_code(argument_type).split() if not is_operator(t)])

    return_statement_tokens = []
    return_statement_subtokens = []
    for return_statement in return_statements:
        return_statement_tokens.extend([t for t in tokenize_clean_code(return_statement).split() if not is_operator(t)])
        return_statement_subtokens.extend([t for t in subtokenize_code(return_statement).split() if not is_operator(t)])

    return_type_tokens = [t for t in tokenize_clean_code(return_type).split() if not is_operator(t)]
    return_type_subtokens = [t for t in subtokenize_code(return_type).split() if not is_operator(t)]

    method_name = extract_method_name(code_block)
    method_name_tokens = [method_name]
    method_name_subtokens = subtokenize_code(method_name).split()

    token_elements = {
        'argument_name': argument_name_tokens,
        'argument_type': argument_type_tokens,
        'return_type': return_type_tokens,
        'return_statement': return_statement_tokens,
        'throwable_exception': throwable_exception_tokens,
        'throw_statement': throw_statement_tokens,
        'method_name': method_name_tokens
    }

    subtoken_elements = {
        'argument_name': argument_name_subtokens,
        'argument_type': argument_type_subtokens,
        'return_type': return_type_subtokens,
        'return_statement': return_statement_subtokens,
        'throwable_exception': throwable_exception_subtokens,
        'throw_statement': throw_statement_subtokens,
        'method_name': method_name_subtokens
    }

    return {
        'token': token_elements,
        'subtoken': subtoken_elements
    }

if __name__ == "__main__":
    # Demo for extracting high level features for one example
    # Corresponds to what is written in high_level_features.json files
    
    ex = build_test_example()
    cache = dict()
    cache[ex.id] = {
        'old': get_method_elements(ex.old_code_raw.split('\n')),
        'new': get_method_elements(ex.new_code_raw.split('\n')),
        'code_change_labels': {'subtoken': get_change_labels(ex.token_diff_code_subtokens)}
    }