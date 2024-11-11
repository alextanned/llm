import time
import torch.utils
import torch.utils.data
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import argparse
import os
import datetime
import json
import wandb
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoModel
)


def data_prep(args, split='train'):

    if split == 'extra':
        extra_data = pd.read_pickle("../../../data/IMDB_Finetune/extra/extra_llama3.pkl")

        X_extra, y_extra = extra_data['review'].to_list(), extra_data['label'].to_list()
        return X_extra, y_extra

    train_data = pd.read_pickle("../../../data/IMDB_Finetune/train/train_data.pkl")
    val_data = pd.read_pickle("../../../data/IMDB_Finetune/val/val_data.pkl")
    
    if args.debug:
        train_data = train_data.head(10)
        val_data = val_data.head(10)


    X_train, y_train = train_data['review'].to_list(), train_data['label'].to_list()
    X_test, y_test = val_data['review'].to_list(), val_data['label'].to_list()

    num_classes = 2


    return X_train, y_train, X_test, y_test, num_classes


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some parameters for model training.")

    parser.add_argument(
        '--save-path',
        type=str,
        required=False,
        default="./",
        help='The file path where output file will be saved.'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default="imdb",
        help='The file path where output file will be saved.'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=False,
        default='roberta-base',
        help='The name of the model to be used.'
    )

    parser.add_argument(
        '--binary',
        action='store_true',
        default=False,
        help='Specify whether to perform binary classification. Default is True. Set to False for multiclass classification.'
    )

    parser.add_argument(
        '--score-optimized',
        action='store_true',
        default=False,
        help='Optimize threshold for maximum accuracy'
    )

    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        required=False,
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        required=False,
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        required=False,
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='debug'
    )

    parser.add_argument(
        '--extra',
        action='store_true',
        default=False,
        help='use extra data'
    )


    args = parser.parse_args()
    return args


def load_model(args, model_name, tokenizer_name, num_labels, output_attentions=False, output_hidden_states=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RobertaForSequenceClassification.from_pretrained(model_name,num_labels=num_labels,output_attentions=output_attentions,output_hidden_states=output_hidden_states).to(device)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)

    return model, tokenizer

def get_max_length(sentences, tokenizer):
    max_length = 0
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        max_length = max(max_length, len(encoded_sent))
    
    print(max_length)
    return max_length
    
def create_dataset(sentences, tokenizer, labels):
    # Tokenize the sentences
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True,
                            # max_length = max_length,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation = True,
                    )

        # print(type(sent), len(encoded_dict['input_ids']))
        # exit()

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
        # print(encoded_dict['input_ids'].shape[0])

    # exit()
    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    # Create a TensorDataset with the input_ids, attention_masks, and labels
    dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
    
    return dataset

def get_data_loaders(train_dataset, val_dataset):
    # Create the training dataloader.
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,  # The training samples.
        sampler= torch.utils.data.RandomSampler(train_dataset), # Select batches randomly
        batch_size=wandb.config.batch_size # Trains with this batch size.
    )

    # Create the validation dataloader.
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, # The validation samples.
        sampler=torch.utils.data.SequentialSampler(val_dataset), # Pull out batches sequentially.
        batch_size=1 # Evaluate with this batch size.
    )

    return train_dataloader, val_dataloader
    

def get_optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr=wandb.config.learning_rate, 
                      eps=1e-8)
    return optimizer

def get_scheduler(dataloader, optimizer, extradataloader=None):
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    if extradataloader:
        total_steps = len(dataloader)* (wandb.config.epochs-wandb.config.extra_epochs)+len(extradataloader)* wandb.config.extra_epochs
    else:
        total_steps = len(dataloader) * wandb.config.epochs

    # Create the learning rate scheduler.
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,                                                
    #                                             total_iters = total_steps)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    first_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(extradataloader)* wandb.config.extra_epochs)
    second_scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader)* (wandb.config.epochs-wandb.config.extra_epochs))

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(extradataloader)*wandb.config.extra_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[first_scheduler, second_scheduler], milestones=[len(extradataloader)* wandb.config.extra_epochs])
    return scheduler

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_sentences, train_labels, val_sentences, val_labels, num_classes = data_prep(args)
    if args.extra:
        extra_sentences, extra_labels = data_prep(args, split='extra')

    # Load the model and tokenizer

    # model, tokenizer = load_model('bert-base-uncased', 'bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)   
    #model, tokenizer = load_model('microsoft/deberta-base', 'microsoft/deberta-base', num_labels=2, output_attentions=False, output_hidden_states=False)    
    #model, tokenizer = load_model('YituTech/conv-bert-base', 'YituTech/conv-bert-base', num_labels=2, output_attentions=False, output_hidden_states=False)
    #model, tokenizer = load_model('funnel-transformer/small-base', 'funnel-transformer/small-base', num_labels=2, output_attentions=False, output_hidden_states=False) 
    #model, tokenizer = load_model('studio-ousia/luke-base', 'studio-ousia/luke-base', num_labels=2, output_attentions=False, output_hidden_states=False)  
    #model, tokenizer = load_model("funnel-transformer/small-base", "funnel-transformer/small-base", num_labels=2, output_attentions=False, output_hidden_states=False)  
    model, tokenizer = load_model(args, args.model, args.model, num_labels=num_classes, output_attentions=False, output_hidden_states=False)    
    #model, tokenizer = load_model('squeezebert/squeezebert-uncased', 'squeezebert/squeezebert-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)    
    #model, tokenizer = load_model('xlm-roberta-base', 'xlm-roberta-base', num_labels=2, output_attentions=False, output_hidden_states=False)    
    #model, tokenizer = load_model('microsoft/deberta-v3-base', 'microsoft/deberta-v3-base', num_labels=2, output_attentions=False, output_hidden_states=False) 



    # Get the maximum sentence length
    # max_length = get_max_length(train_sentences+val_sentences, tokenizer)

    # Create datasets
    train_dataset = create_dataset(train_sentences, tokenizer,train_labels)
    val_dataset = create_dataset(val_sentences, tokenizer, val_labels)
    if args.extra:
        extra_dataset = create_dataset(extra_sentences, tokenizer, extra_labels)

    # # Remove sentence_ids from the datasets
    # train_dataset = index_remover(train_dataset)
    # val_dataset = index_remover(val_dataset)
    # test_dataset = index_remover(test_dataset)
    epochs = wandb.config.epochs
    if args.extra:
        extra_epochs = wandb.config.extra_epochs

    # Get data loaders
    train_dataloader, validation_dataloader = get_data_loaders(train_dataset, val_dataset)
    if args.extra:
        extra_dataloader =  torch.utils.data.DataLoader(
            extra_dataset,  # The training samples.
            sampler= torch.utils.data.RandomSampler(extra_dataset), # Select batches randomly
            batch_size=wandb.config.batch_size, # Trains with this batch size.
    )

    # Get optimizer and scheduler
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(train_dataloader, optimizer, extra_dataloader)
    seed_val=6
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Training tracking variables
        true_labels_train = []
        pred_labels_train = []
        total_train_accuracy = 0

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        if args.extra and epoch_i<extra_epochs:
            loader = extra_dataloader

        else:
            loader = train_dataloader

        # For each batch of training data...
        for step, batch in enumerate(loader):
            # print(scheduler.get_last_lr(), type(scheduler.get_last_lr()))
            # print(scheduler.optimizer.param_groups[0]['lr'], scheduler.optimizer.param_groups[0]['lr']))
            # exit()


            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)


            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels).to_tuple()

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # # Clip the norm of the gradients to 1.0.
            # # This is to help prevent the "exploding gradients" problem.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Progress update every 40 batches.
            if step % 5 == 0 and not step == 0:
                # Calculate elapsed time in minutes. 
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(loader), elapsed))
                avg_train_loss = total_train_loss / step
                wandb.log({'train_loss':avg_train_loss})

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_train_accuracy += flat_accuracy(logits, label_ids)

            true_labels_train.extend(label_ids)
            pred_labels_train.extend(np.argmax(logits, axis=1))

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)       

        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        print("  accuracy: {0:.2f}".format(avg_train_accuracy))

        f1 = f1_score(true_labels_train, pred_labels_train, average='macro')
        print("  f1 score: {0:.2f}".format(f1))
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        wandb.log({'avg_train_accuracy': avg_train_accuracy, 
                   'avg_train_loss':avg_train_loss, 
                   'train_macro_f1': f1, 
                   'epochs': epoch_i,
                   'learning_rate': float(scheduler.optimizer.param_groups[0]['lr'])})

        print("")
        print("  average training loss: {0:.3f}".format(avg_train_loss))
        print("  training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Testing
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our testing set.

        print("")
        print("Running testing...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_test_accuracy = 0
        total_test_loss = 0
        nb_eval_steps = 0
        true_labels = []
        pred_labels = []
        all_logits = []
        all_probs = []
        all_labels = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)


            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                loss, logits = model(b_input_ids, 
                                      token_type_ids=None, 
                                      attention_mask=b_input_mask,
                                      labels=b_labels).to_tuple()  
            # Accumulate the test loss.
                total_test_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_test_accuracy += flat_accuracy(logits, label_ids)

            #logits = logits.cpu().numpy()
            probs = torch.softmax(torch.tensor(logits), dim=-1).detach().cpu().numpy()
            predicted_labels = np.argmax(logits, axis=-1)
            all_probs.extend(probs.tolist())
            all_labels.extend(predicted_labels.tolist())

            # Save the labels and probabilities to a JSON file
            output_file = 'labels_bert_probs.json'
            with open(output_file, 'w') as f:
              json.dump({'labels': all_labels, 'probs': all_probs}, f)


            true_labels.extend(label_ids)
            pred_labels.extend(np.argmax(logits, axis=1))

            

        # Report the final accuracy for this testing run.
        avg_test_accuracy = total_test_accuracy / len(validation_dataloader)
        print("  accuracy: {0:.4f}".format(avg_test_accuracy))

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(validation_dataloader)
        f1 = f1_score(true_labels, pred_labels, average='macro')
        print("  f1 score: {0:.4f}".format(f1))
        
        # Measure how long the testing run took.
        testing_time = format_time(time.time() - t0)
        wandb.log({'avg_test_accuracy':avg_test_accuracy,'avg_test_loss':avg_test_loss, 'test_macro_f1': f1})
        print("  testing loss: {0:.2f}".format(avg_test_loss))
        print("  testing took: {:}".format(testing_time))


    print("")
    print("training complete!")

    print("total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    wandb.login()
    wandb.init(project="LLM_Distillation", entity="distill-llms", config={"learning_rate": args.lr, "epochs": args.epochs, "batch_size": args.bs, "extra_epochs": 10})
    train(args)
    # print(f"CSV will be saved at: {args.save_csv}")
    # print(f"Model name: {args.model_name}")
    # print(f"Classification type: {args.classification_type}")
