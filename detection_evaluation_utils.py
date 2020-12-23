import logging

from sklearn.metrics import precision_recall_fscore_support

def compute_average(values):
    return sum(values)/float(len(values))

def compute_score(predicted_labels, gold_labels, verbose=True):
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    assert(len(predicted_labels) == len(gold_labels))

    for i in range(len(gold_labels)):
        if gold_labels[i]:
            if predicted_labels[i]:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted_labels[i]:
                false_positives += 1
            else:
                true_negatives += 1
    
    if verbose:
        print('True positives: {}'.format(true_positives))
        print('False positives: {}'.format(false_positives))
        print('True negatives: {}'.format(true_negatives))
        print('False negatives: {}'.format(false_negatives))
    
    try:
        precision = true_positives/(true_positives + false_positives)
    except:
        precision = 0.0
    try:
        recall = true_positives/(true_positives + false_negatives)
    except:
        recall = 0.0
    try:
        f1 = 2*((precision * recall)/(precision + recall))
    except:
        f1 = 0.0
    return precision, recall, f1