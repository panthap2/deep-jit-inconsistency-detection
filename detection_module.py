import argparse
from collections import Counter
import numpy as np
import os
import random
import sys
import torch
from torch import nn

from constants import *
from detection_evaluation_utils import compute_score


class DetectionModule(nn.Module):
    """Binary classification model for detecting inconsistent comments."""
    def __init__(self, model_path, manager):
        super(DetectionModule, self).__init__()

        self.model_path = model_path
        self.manager = manager
        feature_input_dimension = self.manager.out_dim

        self.output_layer = nn.Linear(feature_input_dimension, NUM_CLASSES)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)

    def get_logprobs(self, encoder_outputs):
        """Computes the class-level log probabilities corresponding to the examples in the batch."""
        logits = self.output_layer(encoder_outputs.attended_old_nl_final_state)
        return torch.nn.functional.log_softmax(logits, dim=-1)
    
    def compute_detection_loss(self, encoder_outputs, batch_data):
        """Computes the negative log likelihood loss against the gold labels corresponding to the examples in the batch."""
        logprobs = self.get_logprobs(encoder_outputs)
        return torch.nn.functional.nll_loss(logprobs, batch_data.labels), logprobs
    
    def forward(self, batch_data):
        """Computes prediction loss for given batch."""
        encoder_outputs = self.manager.get_encoder_output(batch_data, self.get_device())
        loss, logprobs = self.compute_detection_loss(encoder_outputs, batch_data)
        return loss, logprobs
    
    def run_train(self, train_examples, valid_examples):
        """Runs training over the entire training set across several epochs. Following each epoch,
           F1 on the validation data is computed. If the validation F1 has improved, save the model.
           Early-stopping is employed to stop training if validation hasn't improved for a certain number
           of epochs."""
        valid_batches = self.manager.get_batches(valid_examples, self.get_device())
        best_loss = float('inf')
        best_f1 = 0.0
        patience_tally = 0

        for epoch in range(MAX_EPOCHS):
            if patience_tally > PATIENCE:
                print('Terminating: {}'.format(epoch))
                break
            
            self.train()
            train_batches = self.manager.get_batches(train_examples, self.get_device(), shuffle=True)
            
            train_loss = 0
            for batch_data in train_batches:
                train_loss += self.run_gradient_step(batch_data)
        
            self.eval()
            validation_loss = 0
            validation_predicted_labels = []
            validation_gold_labels = []
            with torch.no_grad():
                for batch_data in valid_batches:
                    b_loss, b_logprobs = self.forward(batch_data)
                    validation_loss += float(b_loss.cpu())
                    validation_predicted_labels.extend(b_logprobs.argmax(-1).tolist())
                    validation_gold_labels.extend(batch_data.labels.tolist())

            validation_loss = validation_loss/len(valid_batches)
            validation_precision, validation_recall, validation_f1 = compute_score(
                validation_predicted_labels, validation_gold_labels, verbose=False)
            
            if validation_f1 >= best_f1:
                best_f1 = validation_f1
                torch.save(self, self.model_path)
                saved = True
                patience_tally = 0
            else:
                saved = False
                patience_tally += 1
            
            print('Epoch: {}'.format(epoch))
            print('Training loss: {:.3f}'.format(train_loss/len(train_batches)))
            print('Validation loss: {:.3f}'.format(validation_loss))
            print('Validation precision: {:.3f}'.format(validation_precision))
            print('Validation recall: {:.3f}'.format(validation_recall))
            print('Validation f1: {:.3f}'.format(validation_f1))
            if saved:
                print('Saved')
            print('-----------------------------------')
            sys.stdout.flush()
    
    def get_device(self):
        """Returns the proper device."""
        if self.torch_device_name == 'gpu':
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def run_gradient_step(self, batch_data):
        """Performs gradient step."""
        self.optimizer.zero_grad()
        loss, _ = self.forward(batch_data)
        loss.backward()
        self.optimizer.step()
        return float(loss.cpu())
    
    def run_evaluation(self, test_examples, model_name):
        """Predicts labels for all comments in the test set and computes evaluation metrics."""
        self.eval()

        test_batches = self.manager.get_batches(test_examples, self.get_device())
        test_predictions = []

        with torch.no_grad():
            for b, batch in enumerate(test_batches):
                print('Testing batch {}/{}'.format(b, len(test_batches)))
                sys.stdout.flush()
                encoder_outputs = self.manager.get_encoder_output(batch, self.get_device())
                batch_logprobs = self.get_logprobs(encoder_outputs)
                test_predictions.extend(batch_logprobs.argmax(dim=-1).tolist())

        self.compute_metrics(test_predictions, test_examples, model_name)
    
    def compute_metrics(self, predicted_labels, test_examples, model_name):
        """Computes evaluation metrics."""
        gold_labels = []
        correct = 0
        for e, ex in enumerate(test_examples):
            if ex.label == predicted_labels[e]:
                correct += 1
            gold_labels.append(ex.label)
        
        accuracy = float(correct)/len(test_examples)
        precision, recall, f1 = compute_score(predicted_labels, gold_labels)
        
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1: {}'.format(f1))
        print('Accuracy: {}\n'.format(accuracy))

        write_file = os.path.join(DETECTION_DIR, '{}_detection.txt'.format(model_name))
        with open(write_file, 'w+') as f:
            for e, ex in enumerate(test_examples):
                f.write('{} {}\n'.format(ex.id, predicted_labels[e]))
