import sys

from data_formatting_utils import subtokenize_code, subtokenize_comment

sys.path.append('../')
sys.path.append('../comment_update')
from data_utils import DiffASTExample
from diff_utils import compute_minimal_comment_diffs, compute_code_diffs

# NOTE: Javalang will need to installed for this
def build_test_example():
    example_id = 'test-id-0'
    label = 1
    comment_type = 'Return'
    old_comment_raw = '@return the highest score'
    old_comment_subtokens = subtokenize_comment(old_comment_raw).split()
    new_comment_raw = '@return the lowest score'
    new_comment_subtokens = subtokenize_comment(new_comment_raw).split()
    span_minimal_diff_comment_subtokens, _, _ = compute_minimal_comment_diffs(
        old_comment_subtokens, new_comment_subtokens)
    old_code_raw = 'public int getBestScore()\n{\n\treturn Collections.max(scores);\n}'
    old_code_subtokens = subtokenize_code(old_code_raw).split()
    new_code_raw = 'public int getBestScore()\n{\n\treturn Collections.min(scores);\n}'
    new_code_subtokens = subtokenize_code(new_code_raw).split()
    span_diff_code_subtokens, token_diff_code_subtokens, _ = compute_code_diffs(old_code_subtokens, new_code_subtokens)

    # TODO: Add code for parsing ASTs
    old_ast = None
    new_ast = None
    diff_ast = None

    return DiffASTExample(example_id, label, comment_type, old_comment_raw, old_comment_subtokens, new_comment_raw,
        new_comment_subtokens, span_minimal_diff_comment_subtokens, old_code_raw, old_code_subtokens, new_code_raw,
        new_code_subtokens, span_diff_code_subtokens, token_diff_code_subtokens, old_ast, new_ast, diff_ast)