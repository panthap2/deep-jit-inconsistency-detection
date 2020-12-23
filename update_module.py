from datetime import datetime
import numpy as np
import os
import random
import sys
import torch
from torch import nn

from constants import *
from data_utils import get_processed_comment_sequence, get_processed_comment_str, Example
from detection_evaluation_utils import compute_score
import diff_utils
from encoder import Encoder
from external_cache import get_old_code, get_new_code
from update_evaluation_utils import compute_accuracy, compute_bleu, compute_meteor, write_predictions,\
    compute_sentence_bleu, compute_sentence_meteor, compute_sari, compute_gleu
from update_decoder import UpdateDecoder


class UpdateModule(nn.Module):
    """Edit model which learns to map a sequence of code edits to a sequence of comment edits and then applies the edits to the
       old comment in order to produce an updated comment."""
    def __init__(self, model_path, manager, detector):
        super(UpdateModule, self).__init__()
        self.model_path = model_path
        self.manager = manager
        self.detector = detector

        self.decoder = UpdateDecoder(NL_EMBEDDING_SIZE, self.manager.out_dim,
            self.manager.out_dim, self.manager.embedding_store,
            NL_EMBEDDING_SIZE, DROPOUT_RATE, self.manager.update_encoder_state_size)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)

    def forward(self, batch_data):
        """Computes the loss against the gold sequences corresponding to the examples in the batch. NOTE: teacher-forcing."""
        encoder_outputs = self.manager.get_encoder_output(batch_data, self.get_device())
        decoder_input_embeddings = self.manager.embedding_store.get_nl_embeddings(batch_data.trg_nl_ids)[:, :-1]
        decoder_states, decoder_final_state, generation_logprobs, copy_logprobs = self.decoder.forward(
            encoder_outputs.encoder_final_state, decoder_input_embeddings, encoder_outputs.encoder_hidden_states,
            encoder_outputs.code_hidden_states, encoder_outputs.old_nl_hidden_states, encoder_outputs.masks,
            encoder_outputs.code_masks, encoder_outputs.old_nl_masks)

        gold_generation_ids = batch_data.trg_nl_ids[:, 1:].unsqueeze(-1)
        gold_generation_logprobs = torch.gather(input=generation_logprobs, dim=-1,
                                                index=gold_generation_ids).squeeze(-1)
        copy_logprobs = copy_logprobs.masked_fill(
            batch_data.invalid_copy_positions[:,1:,:encoder_outputs.encoder_hidden_states.shape[1]], float('-inf'))
        gold_copy_logprobs = copy_logprobs.logsumexp(dim=-1)

        gold_logprobs = torch.logsumexp(torch.cat(
            [gold_generation_logprobs.unsqueeze(-1), gold_copy_logprobs.unsqueeze(-1)], dim=-1), dim=-1)
        gold_logprobs = gold_logprobs.masked_fill(torch.arange(batch_data.trg_nl_ids[:,1:].shape[-1],
            device=self.get_device()).unsqueeze(0) >= batch_data.trg_nl_lengths.unsqueeze(-1)-1, 0)
        
        likelihood_by_example = gold_logprobs.sum(dim=-1)

        # Normalizing by length. Seems to help
        likelihood_by_example = likelihood_by_example/(batch_data.trg_nl_lengths-1).float()

        loss_by_example = -(likelihood_by_example)

        if self.manager.task == 'dual':
            # Mask loss for consistent cases
            masked_loss_by_example = loss_by_example * batch_data.labels.type(torch.FloatTensor).cuda(self.get_device())
            # Divide by only the number of inconsistent cases in batch
            loss = masked_loss_by_example.sum()/ batch_data.labels.sum()
        else:
            loss = loss_by_example.mean()

        if self.detector is not None:
            detection_loss, _ = self.detector.compute_detection_loss(encoder_outputs, batch_data)
            loss += detection_loss
        
        return loss
    
    def beam_decode(self, batch_data):
        """Performs beam search on the decoder to get candidate predictions for every example in the batch."""
        encoder_outputs = self.manager.get_encoder_output(batch_data, self.get_device())
        predictions, scores = self.decoder.beam_decode(encoder_outputs.encoder_final_state,
            encoder_outputs.encoder_hidden_states, encoder_outputs.code_hidden_states,
            encoder_outputs.old_nl_hidden_states, encoder_outputs.masks, self.manager.max_nl_length,
            batch_data, encoder_outputs.code_masks, encoder_outputs.old_nl_masks, self.get_device())

        decoded_output = []
        batch_size = encoder_outputs.encoder_final_state.shape[0]

        if self.detector is not None:
            detection_logprobs = self.detector.get_logprobs(encoder_outputs)
            inconsistency_labels = detection_logprobs.argmax(dim=-1)
        else:
            inconsistency_labels = torch.ones([batch_size], dtype=torch.int64, device=self.get_device())

        for i in range(batch_size):
            beam_output = []
            for j in range(len(predictions[i])):
                token_ids = predictions[i][j]
                tokens = self.manager.embedding_store.get_nl_tokens(token_ids, batch_data.input_ids[i],
                    batch_data.input_str_reps[i])
                beam_output.append((tokens, scores[i][j]))
            decoded_output.append(beam_output)
        return decoded_output, inconsistency_labels
    
    def get_device(self):
        """Returns the proper device."""
        if self.torch_device_name == 'gpu':
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def run_gradient_step(self, batch_data):
        """Performs gradient step."""
        self.optimizer.zero_grad()
        loss = self.forward(batch_data)
        loss.backward()
        self.optimizer.step()
        return float(loss.cpu())

    def run_train(self, train_data, valid_data):
        """Runs training over the entire training set across several epochs. Following each epoch,
           loss on the validation data is computed. If the validation loss has improved, save the model.
           Early-stopping is employed to stop training if validation hasn't improved for a certain number
           of epochs."""
        valid_batches = self.manager.get_batches(valid_data, self.get_device())
        train_batches = self.manager.get_batches(train_data, self.get_device(), shuffle=True)

        best_loss = float('inf')
        patience_tally = 0

        for epoch in range(MAX_EPOCHS):
            if patience_tally > PATIENCE:
                print('Terminating')
                break
            
            self.train()
            random.shuffle(train_batches)
            
            train_loss = 0
            for batch_data in train_batches:
                train_loss += self.run_gradient_step(batch_data)
        
            self.eval()
            validation_loss = 0
            with torch.no_grad():
                for batch_data in valid_batches:
                    validation_loss += float(
                        self.forward(batch_data).cpu())

            validation_loss = validation_loss/len(valid_batches)

            if validation_loss <= best_loss:
                torch.save(self, self.model_path)
                saved = True
                best_loss = validation_loss
                patience_tally = 0
            else:
                saved = False
                patience_tally += 1
            
            print('Epoch: {}'.format(epoch))
            print('Training loss: {}'.format(train_loss/len(train_batches)))
            print('Validation loss: {}'.format(validation_loss))
            if saved:
                print('Saved')
            print('-----------------------------------')
            sys.stdout.flush()
    
    def get_likelihood_scores(self, comment_generation_model, formatted_beam_predictions, test_example):
        """Computes the generation likelihood score for each beam prediction based on the pre-trained
           comment generation model."""
        batch_examples = []
        for j in range(len(formatted_beam_predictions)):
            batch_examples.append(Example(test_example.id, test_example.old_comment_raw, test_example.old_comment_subtokens,
                ' '.join(formatted_beam_predictions[j]), formatted_beam_predictions[j],
                test_example.old_code_raw, test_example.old_code_subtokens, test_example.new_code_raw,
                test_example.new_code_subtokens))
        
        batch_data = comment_generation_model.get_batches(batch_examples)[0]
        return np.asarray(comment_generation_model.compute_generation_likelihood(batch_data).cpu())
    
    def get_generation_model(self):
        """Loads the pre-trained comment generation model needed for re-ranking.
           NOTE: the path is hard-coded here so may need to be modified."""
        if self.torch_device_name == 'cpu':
            comment_generation_model = torch.load(FULL_GENERATION_MODEL_PATH, map_location='cpu')
            comment_generation_model.torch_device_name = 'cpu'
            comment_generation_model.cpu()
            for c in comment_generation_model.children():
                c.cpu()
        else:
            comment_generation_model = torch.load(FULL_GENERATION_MODEL_PATH)
            comment_generation_model.torch_device_name = 'gpu'
            comment_generation_model.cuda()
            for c in comment_generation_model.children():
                c.cuda()

        comment_generation_model.eval()
        return comment_generation_model

    def run_evaluation(self, test_data, rerank, model_name):
        """Predicts updated comments for all comments in the test set and computes evaluation metrics."""
        self.eval()

        test_batches = self.manager.get_batches(test_data, self.get_device())
        test_predictions = []
        generation_predictions = []

        gold_strs = []
        pred_strs = []
        src_strs = []

        references = []
        pred_instances = []
        inconsistency_labels = []

        with torch.no_grad():
            for b_idx, batch_data in enumerate(test_batches):
                print('Evaluating {}'.format(b_idx))
                sys.stdout.flush()
                pred, labels = self.beam_decode(batch_data)
                test_predictions.extend(pred)
                inconsistency_labels.extend(labels)

        print('Beam terminating: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))

        if not rerank:
            test_predictions = [pred[0][0] for pred in test_predictions]
        else:
            print('Rerank starting: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
            comment_generation_model = self.get_generation_model()
            reranked_predictions = []
            for i in range(len(test_predictions)):
                formatted_beam_predictions = []
                model_scores = np.zeros(len(test_predictions[i]), dtype=np.float)
                old_comment_subtokens = get_processed_comment_sequence(test_data[i].old_comment_subtokens)
                
                for b, (b_pred, b_score) in enumerate(test_predictions[i]):
                    try:
                        b_pred_str = diff_utils.format_minimal_diff_spans(old_comment_subtokens, b_pred)
                    except:
                        b_pred_str = ''
                    
                    formatted_beam_predictions.append(b_pred_str.split(' '))
                    model_scores[b] = b_score
                
                likelihood_scores = self.get_likelihood_scores(comment_generation_model,
                    formatted_beam_predictions, test_data[i])
                old_meteor_scores = compute_sentence_meteor(
                        [[old_comment_subtokens] for _ in range(len(formatted_beam_predictions))],
                        formatted_beam_predictions)
                
                rerank_scores = [(model_scores[j] * MODEL_LAMBDA) + (likelihood_scores[j] * LIKELIHOOD_LAMBDA) + (
                        old_meteor_scores[j] * OLD_METEOR_LAMBDA) for j in range(len(formatted_beam_predictions))]
                
                sorted_indices = np.argsort(-np.asarray(rerank_scores))
                reranked_predictions.append(test_predictions[i][sorted_indices[0]][0])
            
            test_predictions = reranked_predictions
            print('Rerank terminating: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
        
        print('Final evaluation step starting: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))

        predicted_labels = []
        gold_labels = []
        pseudo_predicted_labels = []
        correct = 0
        pseudo_correct = 0

        for i in range(len(test_predictions)):
            if inconsistency_labels[i] == 0:
                pred_str = get_processed_comment_str(test_data[i].old_comment_subtokens)
            else:
                pred_str = diff_utils.format_minimal_diff_spans(
                    get_processed_comment_sequence(test_data[i].old_comment_subtokens), test_predictions[i])

            gold_str = get_processed_comment_str(test_data[i].new_comment_subtokens)
            src_str = get_processed_comment_str(test_data[i].old_comment_subtokens)
            prediction = pred_str.split()
        
            gold_strs.append(gold_str)
            pred_strs.append(pred_str)
            src_strs.append(src_str)

            predicted_label = inconsistency_labels[i]
            pseudo_predicted_label = int(pred_str != src_str)
            gold_label = test_data[i].label

            if predicted_label == gold_label:
                correct += 1
            if pseudo_predicted_label == gold_label:
                pseudo_correct += 1
            
            predicted_labels.append(predicted_label)
            pseudo_predicted_labels.append(pseudo_predicted_label)
            gold_labels.append(gold_label)
            
            references.append([get_processed_comment_sequence(test_data[i].new_comment_subtokens)])
            pred_instances.append(prediction)

            print('Old comment: {}'.format(src_str))
            print('Gold comment: {}'.format(gold_str))
            print('Predicted comment: {}'.format(pred_str))
            print('Raw prediction: {}'.format(' '.join(test_predictions[i])))
            print('Inconsistency label: {}'.format(inconsistency_labels[i]))
            print('Pseudo inconsistency label: {}\n'.format(pseudo_predicted_label))
            try:
                print('Old code:\n{}\n'.format(get_old_code(test_data[i])))
            except:
                print('Failed to print old code\n')
            try:
                print('New code:\n{}\n'.format(get_new_code(test_data[i])))
            except:
                print('Failed to print new code\n')
            print('----------------------------')

        if rerank:
            prediction_file = '{}_beam_rerank.txt'.format(model_name)
            pseudo_detection_file = '{}_beam_rerank_pseudo_detection.txt'.format(model_name)
        else:
            prediction_file = '{}_beam.txt'.format(model_name)
            pseudo_detection_file = '{}_beam_pseudo_detection.txt'.format(model_name)
        
        detection_file = os.path.join(PREDICTION_DIR, '{}_detection.txt'.format(model_name))
        pseudo_detection_file = os.path.join(PREDICTION_DIR, pseudo_detection_file)

        prediction_file = os.path.join(PREDICTION_DIR, prediction_file)
        src_file = os.path.join(PREDICTION_DIR, '{}_src.txt'.format(model_name))
        ref_file = os.path.join(PREDICTION_DIR, '{}_ref.txt'.format(model_name))
        
        write_predictions(pred_strs, prediction_file)
        write_predictions(src_strs, src_file)
        write_predictions(gold_strs, ref_file)

        predicted_accuracy = compute_accuracy(gold_strs, pred_strs)
        predicted_bleu = compute_bleu(references, pred_instances)
        predicted_meteor = compute_meteor(references, pred_instances)
        predicted_sari = compute_sari(test_data, pred_instances)
        predicted_gleu = compute_gleu(test_data, src_file, ref_file, prediction_file)

        print('Update Accuracy: {}'.format(predicted_accuracy))
        print('Update BLEU: {}'.format(predicted_bleu))
        print('Update Meteor: {}'.format(predicted_meteor))
        print('Update SARI: {}'.format(predicted_sari))
        print('Update GLEU: {}\n'.format(predicted_gleu))

        if self.manager.task == 'dual':
            with open(detection_file, 'w+') as f:
                for d in range(len(predicted_labels)):
                    f.write('{} {}\n'.format(test_data[d].id, predicted_labels[d]))

            detection_precision, detection_recall, detection_f1 = compute_score(
                predicted_labels, gold_labels, False)
            print('Detection Precision: {}'.format(detection_precision))
            print('Detection Recall: {}'.format(detection_recall))
            print('Detection F1: {}'.format(detection_f1))
            print('Detection Accuracy: {}\n'.format(float(correct)/len(test_data)))

        if self.manager.task == 'update':
            # Evaluating implicit detection.
            with open(pseudo_detection_file, 'w+') as f:
                for d in range(len(pseudo_predicted_labels)):
                    f.write('{} {}\n'.format(test_data[d].id, pseudo_predicted_labels[d]))
            
            pseudo_detection_precision, pseudo_detection_recall, pseudo_detection_f1 = compute_score(
                pseudo_predicted_labels, gold_labels, False)
            print('Pseudo Detection Precision: {}'.format(pseudo_detection_precision))
            print('Pseudo Detection Recall: {}'.format(pseudo_detection_recall))
            print('Pseudo Detection F1: {}'.format(pseudo_detection_f1))
            print('Pseudo Detection Accuracy: {}\n'.format(float(pseudo_correct)/len(test_data)))