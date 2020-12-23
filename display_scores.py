import argparse
import os
import sys

sys.path.append('comment_update')
from data_loader import get_data_splits, load_cleaned_test_set
from data_utils import get_processed_comment_str
from detection_evaluation_utils import compute_score
from update_evaluation_utils import write_predictions, compute_accuracy, compute_bleu,\
    compute_meteor, compute_sari, compute_gleu

"""Script for printing update or detection metrics for output, on full and clean test sets."""

def load_predicted_detection_labels(filepath, selected_positions):
    with open(filepath) as f:
        lines = f.readlines()
    
    selected_labels = []
    for s in selected_positions:
        selected_labels.append(int(lines[s].strip().split()[-1]))
    return selected_labels

def load_predicted_generation_sequences(filepath, selected_positions):
    with open(filepath) as f:
        lines = f.readlines()
    
    selected_sequences = []
    for s in selected_positions:
        selected_sequences.append(lines[s].strip())
    return selected_sequences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_output_file', help='path to detection output file')
    parser.add_argument('--update_output_file', help='path to update output file')
    args = parser.parse_args()

    # NOTE: To evaluate the pretrained approach, detection_output_file and
    # update_output_file must be both specified. For all other approaches,
    # only one should be specified.

    _, _, test_examples, _ = get_data_splits(ignore_ast=True)
    positions = list(range(len(test_examples)))

    clean_ids = load_cleaned_test_set()
    clean_positions = []
    for e, example in enumerate(test_examples):
        if example.id in clean_ids:
            clean_positions.append(e)
    clean_test_examples = [test_examples[pos] for pos in clean_positions]

    eval_tuples = [(test_examples, positions, 'full'), (clean_test_examples, clean_positions, 'clean')]

    for (examples, indices, test_type) in eval_tuples:
        if args.detection_output_file:
            predicted_labels = load_predicted_detection_labels(args.detection_output_file, indices)
            gold_labels = [ex.label for ex in examples]

            precision, recall, f1 = compute_score(predicted_labels, gold_labels, verbose=False)

            num_correct = 0
            for p, p_label in enumerate(predicted_labels):
                if p_label == gold_labels[p]:
                    num_correct += 1
            
            print('Detection Precision: {}'.format(precision))
            print('Detection Recall: {}'.format(recall))
            print('Detection F1: {}'.format(f1))
            print('Detection Accuracy: {}\n'.format(float(num_correct)/len(predicted_labels)))
        
        if args.update_output_file:
            update_strs = load_predicted_generation_sequences(args.update_output_file, indices)

            references = []
            pred_instances = []
            src_strs = []
            gold_strs = []
            pred_strs = []
            
            for i in range(len(examples)):
                src_str = get_processed_comment_str(examples[i].old_comment_subtokens)
                src_strs.append(src_str)
                
                gold_str = get_processed_comment_str(examples[i].new_comment_subtokens)
                gold_strs.append(gold_str)
                references.append([gold_str.split()])

                if args.detection_output_file and predicted_labels[i] == 0:
                    pred_instances.append(src_str.split())
                    pred_strs.append(src_str)
                else:
                    pred_instances.append(update_strs[i].split())
                    pred_strs.append(update_strs[i])
                
            prediction_file = os.path.join(os.getcwd(), 'pred.txt')
            src_file = os.path.join(os.getcwd(), 'src.txt')
            ref_file = os.path.join(os.getcwd(), 'ref.txt')

            write_predictions(pred_strs, prediction_file)
            write_predictions(src_strs, src_file)
            write_predictions(gold_strs, ref_file)

            predicted_accuracy = compute_accuracy(gold_strs, pred_strs)
            predicted_bleu = compute_bleu(references, pred_instances)
            predicted_meteor = compute_meteor(references, pred_instances)
            predicted_sari = compute_sari(examples, pred_instances)
            predicted_gleu = compute_gleu(examples, src_file, ref_file, prediction_file)

            print('Update Accuracy: {}'.format(predicted_accuracy))
            print('Update BLEU: {}'.format(predicted_bleu))
            print('Update Meteor: {}'.format(predicted_meteor))
            print('Update SARI: {}'.format(predicted_sari))
            print('Update GLEU: {}\n'.format(predicted_gleu))
        
        print('Test type: {}'.format(test_type))
        print('Detection file: {}'.format(args.detection_output_file))
        print('Update file: {}'.format(args.update_output_file))
        print('Total: {}'.format(len(examples)))
        print('--------------------------------------')




