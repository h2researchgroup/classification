import torch
import pickle
import regex as re

from azureml.core.run import Run

import os
import random as rand
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
# from clean_text import stopwords_make, punctstr_make, unicode_make, apache_tokenize, clean_sentence_apache 

import matplotlib.pyplot as plt


from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import transformers
import sys, argparse
import numpy as np
import re
import random
from tqdm import tqdm
from collections import Counter
from transformers import BertTokenizer, BertModel
from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from transformers import AdamW
from transformers import get_scheduler

from transformers import LongformerModel, LongformerTokenizer

import argparse


# In[2]:


parser = argparse.ArgumentParser(description='Longformer Params')
parser.add_argument('--data_name', type=str, default='relational')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_memory_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--pretrained_lr', type=float, default=1e-5)
parser.add_argument('--eps', type=float, default=1e-9)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--num_warmup_steps', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--maxlen', type=int, default=1024)
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--once', type=bool, default=False)
args = vars(parser.parse_args())
args['max_length'] = args['maxlen']


print('args', args)

data_name = args['data_name']


def open_test_data(path):
    return open(path, 'rb')
with open_test_data(f'training_{data_name}_preprocessed_100321.pkl') as f:
    # the dataset is called "cult" as an artifact of previous revisions
    cult = pickle.load(f)


# In[3]

run = Run.get_submitted_run()


import itertools
full_text = []

for i in cult['text']:
    joined = list(itertools.chain(*i))
    full_text.append(" ".join(joined))


# In[4]:


cult['full_text'] = full_text


# In[5]:


def remove_tags(article):
    # Removes common tags from a texts
    article = re.sub('<plain_text> <page sequence="1">', '', article)
    article = re.sub(r'</page>(\<.*?\>)', ' \n ', article)
    # xml tags
    article = re.sub(r'<.*?>', '', article)
    article = re.sub(r'<body.*\n\s*.*\s*.*>', '', article)
    return article

tags_removed = [remove_tags(art) for art in cult['full_text']]

# removes tags from text
cult['text_no_tags'] = tags_removed


# In[6]:


## remove cult_label that are 0.5

cult = cult[cult[f'{data_name}_score']!=0.5]
cult


# In[7]:


if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[8]:

# define the classifier model
class BERTClassifier(nn.Module):

        
    def __init__(self, params):
        super().__init__()
        
        if 'dropout' in params:
            self.dropout = nn.Dropout(p=params['dropout'])
        else:
            self.dropout = None
            
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False, do_basic_tokenize=False)
#         self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.max_length = params['max_length'] if 'max_length' in params else 1024
        self.max_memory_size = params['max_memory_size']
        
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.bert = LongformerModel.from_pretrained("allenai/longformer-base-4096", gradient_checkpointing=True)

        self.num_labels = params["label_length"] if 'label_length' in params else 2

        self.fc = nn.Linear(768, self.num_labels)

    def get_batches(self, all_x, all_y, batch_size=10):

        """ Get batches for input x, y data, with data tokenized according to the BERT tokenizer 
        (and limited to a maximum number of WordPiece tokens """

        batches_x=[]
        batches_y=[]

        for i in range(0, len(all_x), batch_size):

            current_batch=[]

            x=all_x[i:i+batch_size]

            # batch_x = self.tokenizer(x, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")
            batch_x = self.tokenizer.batch_encode_plus(x, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")
            
            batch_y=all_y[i:i+batch_size]

            batches_x.append(batch_x.to(device))
            batches_y.append(torch.LongTensor(batch_y).to(device))

        return batches_x, batches_y


    def forward(self, batch_x): 

        bert_output = self.bert(input_ids=batch_x["input_ids"],
                         attention_mask=batch_x["attention_mask"],
#                          token_type_ids=batch_x["token_type_ids"],
                         output_hidden_states=True)

        # We're going to represent an entire document just by its [CLS] embedding (at position 0)
        # And use the *last* layer output (layer -1)
        # as a result of this choice, this embedding will be optimized for this purpose during the training process.

        bert_hidden_states = bert_output['hidden_states']

        out = bert_hidden_states[-1][:,0,:]
        
        if self.dropout:
            out = self.dropout(out)

        out = self.fc(out)

        return out.squeeze()

    def evaluate(self, batch_x, batch_y):

        self.eval()
        corr = 0.
        total = 0.

        with torch.no_grad():
            for x, y in zip(batch_x, batch_y):
                y_preds = self.forward(x)
                if self.max_memory_size == 1:
                    prediction=torch.argmax(y_preds)
                    # print(prediction, y_preds, y)
                    if prediction == y:
                        corr += 1.
                    total+=1
                else:
                    for idx, y_pred in enumerate(y_preds):
                        prediction=torch.argmax(y_pred)
                        # print(prediction, y_pred)
                        if prediction == y[idx]:
                            corr += 1.
                        total+=1                          
        return corr/total


# In[9]:


def oversample_shuffle(X, y, random_state=52):
    """
    Oversamples X and y for equal class proportions
    """
    
    fakeX = np.arange(len(X), dtype=int).reshape((-1, 1))
    ros = RandomOverSampler(random_state=random_state, sampling_strategy=1.0)
    fakeX, y = ros.fit_resample(fakeX, y)
    p = np.random.permutation(len(fakeX))
    fakeX, y = fakeX[p], y[p]
    X = X.iloc[fakeX.reshape((-1, ))]
    return X.to_numpy(), y.to_numpy()

def create_datasets(texts, scores, test_size=0.2, random_state=42):
    """
    A function to split the texts (X variables) and the scores (y variables), and oversample all at once
    """
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        texts, scores, test_size=test_size, random_state=random_state)
    
    X_train1, y_train1 = oversample_shuffle(X_train1, y_train1, random_state=random_state)
    X_test1, y_test1 = oversample_shuffle(X_test1, y_test1, random_state=random_state)
    
    return X_train1, y_train1, X_test1, y_test1
    
def train_bert(texts, scores, params, test_size=0.2, random_state=42, save=False):
    """
    Trains an entire BERT model with the given parameters, then returns the best accuracy out of all epochs
    
    Parameters list: a dict with the following keys (all optional)
    
    {
    batch_size: The batch size as far as the optimizer is concerned. default 32
    max_memory_size: The largest batch to actually use. This is because large batches wont fit on many GPUs.
                        Gradient accumulation will then let us run optimization on batch_size instead. default 4
    
    num_epochs: The total number of epochs to train. default 8
    lr: The learning rate to use for the final linear layer. default 1e-3
    pretrained_lr = A separate learning rate for only the pretrained model. default 1e-5
    betas = The betas parameter in the Adam optimizer. default (0.9, 0.98)
    eps = The eps parameter in the Adam optimizer. default 1e-9
    weight_decay = The weight_decay parameter in the Adam optimizer. default 0.005
    num_warmup_steps = The number of warmup steps (each step is a batch). default 0
    num_training_steps = The number of training steps in the algorithm. default num_epochs * len(batch_x) - num_warmup_steps
    
    dropout: A float (i.e. 0.1) if dropout is desired. default None
    }
    """
    print(params)
    np.random.seed(random_state)
    torch.random.manual_seed(random_state)
    
    texts = texts.copy()
    scores = scores.copy()
    
    bert_model = BERTClassifier(params={**params, "label_length": 2})
    bert_model.to(device)
    
    
    X_train1, y_train1, X_test1, y_test1 = create_datasets(
        texts, scores, test_size=test_size, random_state=random_state)

    batch_size = params['batch_size'] if 'batch_size' in params else 32
    max_memory_size = params['max_memory_size'] if 'max_memory_size' in params else 4
    
    batch_x, batch_y = bert_model.get_batches(list(X_train1), list(y_train1), batch_size=max_memory_size)
    dev_batch_x, dev_batch_y = bert_model.get_batches(list(X_test1),list( y_test1), batch_size=max_memory_size)

    print(batch_x[0]['input_ids'].shape, dev_batch_x[0]['input_ids'].shape)

    
    num_epochs = params['num_epochs'] if 'num_epochs' in params else 8
    lr = params['lr'] if 'lr' in params else 1e-3
    pretrained_lr = params['pretrained_lr'] if 'pretrained_lr' in params else 1e-5
    betas = params['betas'] if 'betas' in params else (0.9, 0.98)
    eps = params['eps'] if 'eps' in params else 1e-9
    weight_decay = params['weight_decay'] if 'weight_decay' in params else 0.005
    num_warmup_steps = params['num_warmup_steps'] if 'num_warmup_steps' in params else 0
    num_training_steps = params['num_training_steps'] if 'num_training_steps' in params else num_epochs * len(batch_x) - num_warmup_steps
    
    
    trainable_parameters = list(
        params for params in bert_model.parameters() if params.requires_grad)
    pretrained_params = set(trainable_parameters) & set(bert_model.bert.parameters())
    new_params = set(trainable_parameters) - pretrained_params
    grouped_trainable_parameters = [
        {
            'params': list(pretrained_params),
            'lr': pretrained_lr,
        },
        {
            'params': list(new_params),
            'lr': lr,
        },
    ]
    
    #torch.optim.
    optimizer = AdamW(
        grouped_trainable_parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    
    scheduler = get_scheduler(
        'linear',
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    cross_entropy=nn.CrossEntropyLoss()

    best_dev_acc = 0.

    for epoch in tqdm(range(num_epochs)):
        bert_model.train()

        # Train
        optimizer.zero_grad()
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            y_pred = bert_model.forward(x)
            loss = cross_entropy(y_pred.view(-1, bert_model.num_labels), y.view(-1))
            loss.backward()
            if (i + 1) % (batch_size // max_memory_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Evaluate
        dev_accuracy=bert_model.evaluate(dev_batch_x, dev_batch_y)
        if epoch % 1 == 0:
#             print("Epoch %s, dev accuracy: %.3f" % (epoch, dev_accuracy))
            if dev_accuracy > best_dev_acc:
                if save:
                    torch.save(bert_model.state_dict(), save)
                best_dev_acc = dev_accuracy
    if save:
        bert_model.load_state_dict(torch.load('best-model-parameters.pt'))
        
    print("\nBest Performing Model achieves dev accuracy of : %.3f" % (best_dev_acc))

    return best_dev_acc


# In[10]:


def cross_validate(texts, scores, params, folds=5, seed=42):
    np.random.seed(seed)
    states = np.random.randint(0, 100000, size=folds)
    accs = []
    for state in states:
        print(state)
        accs.append(train_bert(texts, scores, params, test_size=round(1/folds, 2), random_state=state, save=False))
    acc = np.mean(accs)
    run.log('accuracy', acc)
    return acc


# In[11]:


# 0.799367986285495
#0.8089863310259403
if args['once']:
    train_bert(cult['text_no_tags'], cult[f'{data_name}_score'], args, test_size=round(1/5, 2), random_state=42, save=False)
else:
    accuracy = cross_validate(cult['text_no_tags'], cult[f'{data_name}_score'], args, folds=args['folds'])
    accuracy

