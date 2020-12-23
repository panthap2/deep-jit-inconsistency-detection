import argparse
from datetime import datetime
import os
import sys
import torch

sys.path.append('comment_update')
from comment_generation import CommentGenerationModel
from update_module import UpdateModule
from detection_module import DetectionModule
from data_loader import get_data_splits
from module_manager import ModuleManager

def build_model(task, model_path, manager):
    """ Builds the appropriate model, with task-specific modules."""
    if task == 'dual':
        detection_module = DetectionModule(None, manager)
        model = UpdateModule(model_path, manager, detection_module)
    elif 'update' in task:
        model = UpdateModule(model_path, manager, None)
    else:
       model = DetectionModule(model_path, manager)
    
    return model

def load_model(model_path, evaluate_detection=False):
    """Loads a pretrained model from model_path."""
    print('Loading model from: {}'.format(model_path))
    sys.stdout.flush()
    if torch.cuda.is_available() and evaluate_detection:
        model = torch.load(model_path)
        model.torch_device_name = 'gpu'
        model.cuda()
        for c in model.children():
            c.cuda()
    else:
        model = torch.load(model_path, map_location='cpu')
        model.torch_device_name = 'cpu'
        model.cpu()
        for c in model.children():
            c.cpu()
    return model

def train(model, train_examples, valid_examples):
    """Trains a model."""
    print('Training with {} examples (validation {})'.format(len(train_examples), len(valid_examples)))
    sys.stdout.flush()
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

def evaluate(task, model, test_examples, model_name, rerank):
    """Runs evaluation over a given model."""
    print('Evaluating {} examples'.format(len(test_examples)))
    sys.stdout.flush()
    if task == 'detect':
        model.run_evaluation(test_examples, model_name)
    else:
        model.run_evaluation(test_examples, rerank, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='detect, update, or dual')
    parser.add_argument('--attend_code_sequence_states', action='store_true', help='attend to sequence-based code hidden states for detection')
    parser.add_argument('--attend_code_graph_states', action='store_true', help='attend to graph-based code hidden states for detection')
    parser.add_argument('--features', action='store_true', help='concatenate lexical and linguistic feats to code/comment input embeddings')
    parser.add_argument('--posthoc', action='store_true', help='whether to run in posthoc mode where old code is not available')
    parser.add_argument('--positive_only', action='store_true', help='whether to train on only inconsistent examples')
    parser.add_argument('--test_mode', action='store_true', help='whether to run evaluation')
    parser.add_argument('--rerank', action='store_true', help='whether to use reranking in the update module (if task is update or dual)')
    parser.add_argument('--model_path', help='path to save model (training) or path to saved model (evaluation)')
    parser.add_argument('--model_name', help='name of model (used to save model output)')
    args = parser.parse_args()

    train_examples, valid_examples, test_examples, high_level_details = get_data_splits()
    if args.positive_only:
        train_examples = [ex for ex in train_examples if ex.label == 1]
        valid_examples = [ex for ex in valid_examples if ex.label == 1]
    
    print('Train: {}'.format(len(train_examples)))
    print('Valid: {}'.format(len(valid_examples)))
    print('Test: {}'.format(len(test_examples)))

    if args.task == 'detect' and (not args.attend_code_sequence_states and not args.attend_code_graph_states):
        raise ValueError('Please specify attention states for detection')
    if args.posthoc and (args.task != 'detect' or args.features):
        # Features and update rely on code changes
        raise ValueError('Posthoc setting not supported for given arguments')

    if args.test_mode:
        print('Starting evaluation: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
        
        model = load_model(args.model_path, args.task =='detect')
        evaluate(args.task, model, test_examples, args.model_name, args.rerank)
        
        print('Terminating evaluation: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
    else:
        print('Starting training: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
        
        manager = ModuleManager(args.attend_code_sequence_states, args.attend_code_graph_states, args.features, args.posthoc, args.task)
        manager.initialize(train_examples)
        model = build_model(args.task, args.model_path, manager)

        print('Model path: {}'.format(args.model_path))
        sys.stdout.flush()
        
        train(model, train_examples, valid_examples)
        
        print('Terminating training: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))