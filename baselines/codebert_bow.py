import argparse
from datetime import datetime
import numpy as np
import os
import random
import torch
from torch import nn
from transformers import *
import sys
import json

sys.path.append('../')
sys.path.append('../comment_update')
from constants import *
from data_loader import get_data_splits
from detection_evaluation_utils import compute_score

BERT_HIDDEN_SIZE = 768
DROPOUT_RATE = 0.6
BATCH_SIZE = 100
CLASSIFICATION_HIDDEN_SIZE = 256
# TRANSFORMERS_CACHE='' # TODO: Fill in

class BERTBatch():
    def __init__(self, old_comment_ids, old_comment_lengths,
                 new_code_ids, new_code_lengths, diff_code_ids, diff_code_lengths, labels):
        self.old_comment_ids = old_comment_ids
        self.old_comment_lengths = old_comment_lengths
        self.new_code_ids = new_code_ids
        self.new_code_lengths = new_code_lengths
        self.diff_code_ids = diff_code_ids
        self.diff_code_lengths = diff_code_lengths
        self.labels = labels

class BERTClassifier(nn.Module):
    def __init__(self, model_path, new_code, diff_code):
        super(BERTClassifier, self).__init__()
        self.model_path = model_path
        self.new_code = new_code
        self.diff_code = diff_code

        self.code_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base", cache_dir=TRANSFORMERS_CACHE)
        self.code_model = RobertaModel.from_pretrained("microsoft/codebert-base", cache_dir=TRANSFORMERS_CACHE)
        self.comment_tokenizer = self.code_tokenizer
        self.comment_model = self.code_model

        self.torch_device_name = 'cpu'
        self.max_nl_length = 0
        self.max_code_length = 0

        print('Model path: {}'.format(self.model_path))
        print('New code: {}'.format(self.new_code))
        print('Diff code: {}'.format(self.diff_code))
        sys.stdout.flush()
    
    def initialize(self, train_examples):
        self.max_nl_length = 200
        self.max_code_length = 200

        output_size = BERT_HIDDEN_SIZE

        if self.new_code:
            output_size += BERT_HIDDEN_SIZE
        if self.diff_code:
            output_size += BERT_HIDDEN_SIZE

        self.classification_dropout_layer = nn.Dropout(p=DROPOUT_RATE)
        self.fc1 = nn.Linear(output_size, CLASSIFICATION_HIDDEN_SIZE)
        self.fc2 = nn.Linear(CLASSIFICATION_HIDDEN_SIZE, CLASSIFICATION_HIDDEN_SIZE)
        self.output_layer = nn.Linear(CLASSIFICATION_HIDDEN_SIZE, NUM_CLASSES)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)
    
    def get_code_inputs(self, input_text, max_length):
        tokens = self.code_tokenizer.tokenize(input_text)
        length = min(len(tokens), max_length)
        tokens = tokens[:length]
        token_ids = self.code_tokenizer.convert_tokens_to_ids(tokens)

        padding_length = max_length - len(tokens)
        token_ids += [self.code_tokenizer.pad_token_id]*padding_length
        return token_ids, length
    
    def get_comment_inputs(self, input_text, max_length):
        tokens = self.comment_tokenizer.tokenize(input_text)
        length = min(len(tokens), max_length)
        tokens = tokens[:length]
        token_ids = self.comment_tokenizer.convert_tokens_to_ids(tokens)

        padding_length = max_length - len(tokens)
        token_ids += [self.comment_tokenizer.pad_token_id]*padding_length
        return token_ids, length
    
    def get_batches(self, dataset, shuffle=False):
        batches = []
        if shuffle:
            random.shuffle(dataset)
        
        curr_idx = 0
        while curr_idx < len(dataset):
            batch_idx = 0
            
            start_idx = curr_idx
            end_idx = min(start_idx + BATCH_SIZE, len(dataset))
            labels = []
            old_comment_ids = []
            old_comment_lengths = []
            new_code_ids = []
            new_code_lengths = []
            diff_code_ids = []
            diff_code_lengths = []

            for i in range(start_idx, end_idx):
                comment_ids, comment_length = self.get_comment_inputs(dataset[i].old_comment_raw, self.max_nl_length)
                old_comment_ids.append(comment_ids)
                old_comment_lengths.append(comment_length)

                if self.new_code:
                    code_ids, code_length = self.get_code_inputs(dataset[i].new_code_raw, self.max_code_length)
                    new_code_ids.append(code_ids)
                    new_code_lengths.append(code_length)
                
                if self.diff_code:
                    code_ids, code_length = self.get_code_inputs(' '.join(dataset[i].span_diff_code_tokens), self.max_code_length)
                    diff_code_ids.append(code_ids)
                    diff_code_lengths.append(code_length)

                labels.append(dataset[i].label)
            
            curr_idx = end_idx 
            batches.append(BERTBatch(
                torch.tensor(old_comment_ids, dtype=torch.int64, device=self.get_device()),
                torch.tensor(old_comment_lengths, dtype=torch.int64, device=self.get_device()),
                torch.tensor(new_code_ids, dtype=torch.int64, device=self.get_device()),
                torch.tensor(new_code_lengths, dtype=torch.int64, device=self.get_device()),
                torch.tensor(diff_code_ids, dtype=torch.int64, device=self.get_device()),
                torch.tensor(diff_code_lengths, dtype=torch.int64, device=self.get_device()),
                torch.tensor(labels, dtype=torch.int64, device=self.get_device())
            ))

        return batches
    
    def get_code_representation(self, input_ids, masks):
        embeddings = self.code_model.embeddings(input_ids)
        if self.torch_device_name == 'cpu':
            factor = masks.type(torch.FloatTensor).unsqueeze(-1)
        else:
            factor = masks.type(torch.FloatTensor).cuda(self.get_device()).unsqueeze(-1)
        embeddings = embeddings * factor
        vector = torch.sum(embeddings, dim=1)/torch.sum(factor, dim=1)
        return embeddings, vector

    def get_comment_representation(self, input_ids, masks):
        embeddings = self.comment_model.embeddings(input_ids)
        if self.torch_device_name == 'cpu':
            factor = masks.type(torch.FloatTensor).unsqueeze(-1)
        else:
            factor = masks.type(torch.FloatTensor).cuda(self.get_device()).unsqueeze(-1)
        embeddings = embeddings * factor
        vector = torch.sum(embeddings, dim=1)/torch.sum(factor, dim=1)
        return embeddings, vector

    def get_input_features(self, batch_data):
        old_comment_masks = (torch.arange(
            batch_data.old_comment_ids.shape[1], device=self.get_device()).view(1, -1) < batch_data.old_comment_lengths.view(-1, 1))
        old_comment_hidden_states, old_comment_final_state = self.get_comment_representation(batch_data.old_comment_ids, old_comment_masks)
        final_state = old_comment_final_state

        if self.new_code:
            new_code_masks = (torch.arange(
                batch_data.new_code_ids.shape[1], device=self.get_device()).view(1, -1) < batch_data.new_code_lengths.view(-1, 1))
            new_code_hidden_states, new_code_final_state = self.get_code_representation(batch_data.new_code_ids, new_code_masks)
            final_state = torch.cat([final_state, new_code_final_state], dim=-1)
        
        if self.diff_code:
            diff_code_masks = (torch.arange(
                batch_data.diff_code_ids.shape[1], device=self.get_device()).view(1, -1) < batch_data.diff_code_lengths.view(-1, 1))
            diff_code_hidden_states, diff_code_final_state = self.get_code_representation(batch_data.diff_code_ids, diff_code_masks)
            final_state = torch.cat([final_state, diff_code_final_state], dim=-1)

        return final_state

    def get_logits(self, batch_data):
        all_features = self.get_input_features(batch_data)
        all_features = self.classification_dropout_layer(torch.nn.functional.relu(self.fc1(all_features)))
        all_features = self.classification_dropout_layer(torch.nn.functional.relu(self.fc2(all_features)))
        
        return self.output_layer(all_features)
    
    def get_logprobs(self, batch_data):
        logits = self.get_logits(batch_data)
        return torch.nn.functional.log_softmax(logits, dim=-1)
    
    def forward(self, batch_data, is_training=True):
        logprobs = self.get_logprobs(batch_data)
        loss = torch.nn.functional.nll_loss(logprobs, batch_data.labels)
        return loss, logprobs
        
    def run_train(self, train_examples, valid_examples):
        best_loss = float('inf')
        best_f1 = 0.0
        patience_tally = 0
        valid_batches = self.get_batches(valid_examples)

        for epoch in range(MAX_EPOCHS):
            if patience_tally > PATIENCE:
                print('Terminating')
                break
            
            self.train()
            train_batches = self.get_batches(train_examples, shuffle=True)
            
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
    
    def run_evaluation(self, test_examples, write_file):
        self.eval()

        test_batches = self.get_batches(test_examples)
        test_predictions = []

        with torch.no_grad():
            for b, batch in enumerate(test_batches):
                print('Testing batch {}/{}'.format(b, len(test_batches)))
                sys.stdout.flush()
                batch_logprobs = self.get_logprobs(batch)
                test_predictions.extend(batch_logprobs.argmax(dim=-1).tolist())

        self.compute_metrics(test_predictions, test_examples, write_file)
    
    def compute_metrics(self, predicted_labels, test_examples, write_file):
        gold_labels = []
        correct = 0

        print('Writing to: {}'.format(write_file))
        with open(write_file, 'w+') as f:
            for e, ex in enumerate(test_examples):
                f.write('{} {}\n'.format(ex.id, predicted_labels[e]))
                gold_label = ex.label
                if gold_label == predicted_labels[e]:
                    correct += 1
                gold_labels.append(gold_label)

        accuracy = float(correct)/len(test_examples)
        precision, recall, f1 = compute_score(predicted_labels, gold_labels, False)

        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_code', action='store_true')
    parser.add_argument('--diff_code', action='store_true')
    parser.add_argument('--comment_type')
    parser.add_argument('--trial')
    parser.add_argument('--test_mode', action='store_true')
    args = parser.parse_args()

    print('Starting')
    sys.stdout.flush()
    
    train_examples, valid_examples, test_examples, high_level_details = get_data_splits()

    print('Train: {}'.format(len(train_examples)))
    print('Valid: {}'.format(len(valid_examples)))
    print('Test: {}'.format(len(test_examples)))
    sys.stdout.flush()

    model_name = 'bert'

    if args.new_code:
        model_name += '-new_code'
    if args.diff_code:
        model_name += '-diff_code'

    if args.comment_type:
        model_name += '-{}'.format(args.comment_type)
    if args.trial:
        model_name += '-{}'.format(args.trial)

    # Assumes that saved_bert_models directory exists
    model_path = 'saved_bert_models/{}.pkl.gz'.format(model_name)
    sys.stdout.flush()  

    if args.test_mode:
        print('Loading model from: {}'.format(model_path))
        print('Starting evaluation: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
        sys.stdout.flush()
        model = torch.load(model_path)
        if torch.cuda.is_available():
            model.torch_device_name = 'gpu'
            model.cuda()
            for c in model.children():
                c.cuda()
        else:
            model.torch_device_name = 'cpu'
            model.cpu()
            for c in model.children():
                c.cpu()

        # Assumes that bert_predictions directory exists
        write_file = os.path.join('bert_predictions', '{}.txt'.format(model_name))
        model.run_evaluation(test_examples, write_file)
        print('Terminating evaluation: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
    else:
        print('Starting training: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
        sys.stdout.flush()
        model = BERTClassifier(model_path, args.new_code, args.diff_code)
        model.initialize(train_examples)
        
        if torch.cuda.is_available():
            model.torch_device_name = 'gpu'
            model.cuda()
            for c in model.children():
                c.cuda()
        else:
            model.torch_device_name = 'cpu'
            model.cpu()
            for c in model.children():
                c.cpu()
        
        model.run_train(train_examples, valid_examples)
        print('Terminating training: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))


