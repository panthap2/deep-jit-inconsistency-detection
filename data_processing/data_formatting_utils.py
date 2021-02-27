import argparse
import javalang
import json
import numpy as np
import os
import random
import re
import string

SPECIAL_TAGS = ['{', '}', '@code', '@docRoot', '@inheritDoc', '@link', '@linkplain', '@value']

def remove_html_tag(line):
    clean = re.compile('<.*?>')
    line = re.sub(clean, '', line)

    for tag in SPECIAL_TAGS:
        line = line.replace(tag, '')

    return line

def remove_tag_string(line):
    search_strings = ['@return', '@ return', '@param', '@ param', '@throws', '@ throws']
    for s in search_strings:
        line = line.replace(s, '').strip()
    return line

def tokenize_comment(comment_line, remove_tag=True):
    if remove_tag:
        comment_line = remove_tag_string(comment_line)
    comment_line = remove_html_tag(comment_line)
    comment_line = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", comment_line.strip())
    comment_line = ' '.join(comment_line)
    comment_line = comment_line.replace('\n', ' ').strip()

    return comment_line

def subtokenize_comment(comment_line, remove_tag=True):
    if remove_tag:
        comment_line = remove_tag_string(comment_line)
    comment_line = remove_html_tag(comment_line.replace('/**', '').replace('**/', '').replace('/*', '').replace('*/', '').replace('*', '').strip())
    comment_line = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", comment_line.strip())
    comment_line = ' '.join(comment_line)
    comment_line = comment_line.replace('\n', ' ').strip()

    tokens = comment_line.split(' ')
    subtokens = []
    for token in tokens:
        curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
        try:
            new_curr = []
            for c in curr:
                by_symbol = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip())
                new_curr = new_curr + by_symbol

            curr = new_curr
        except:
            curr = []
        subtokens = subtokens + [c.lower() for c in curr]
    
    comment_line = ' '.join(subtokens)
    return comment_line.lower()

def subtokenize_code(line):
    try:
        tokens = get_clean_code(list(javalang.tokenizer.tokenize(line)))
    except:
        tokens = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", line.strip())
    subtokens = []
    for token in tokens:
        curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
        subtokens = subtokens + [c.lower() for c in curr]
    
    return ' '.join(subtokens)

def tokenize_code(line):
    try:
        tokens = [t.value for t in list(javalang.tokenizer.tokenize(line))]
        return ' '.join(tokens)
    except:
        return tokenize_clean_code(line)

def tokenize_clean_code(line):
    try:
        return ' '.join(get_clean_code(list(javalang.tokenizer.tokenize(line))))
    except:
        return ' '.join(re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", line.strip()))

def get_clean_code(tokenized_code):
    token_vals = [t.value for t in tokenized_code]
    new_token_vals = []
    for t in token_vals:
        n = [c for c in re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", t.encode('ascii', errors='ignore').decode().strip()) if len(c) > 0]
        new_token_vals = new_token_vals + n

    token_vals = new_token_vals
    cleaned_code_tokens = []

    for c in token_vals:
        try:
            cleaned_code_tokens.append(str(c))
        except:
            pass

    return cleaned_code_tokens