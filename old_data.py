import os
import json
import torch
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer

def merge_annotations(x, y):
    if type(y) == str:
        return y.strip().lower()
    return x.strip().lower()

def process_contexts(text):
    citing_contexts = str(text).split('---(1)---')[1].strip()
    temp_citation_context_list = []
    j = 2
    while True:
        citing_contexts = citing_contexts.split(
            '---('+str(j)+')---')
        temp_citation_context_list.append(
            citing_contexts[0].strip())
        if len(citing_contexts) == 1:
            break
        citing_contexts = citing_contexts[1].strip()
        j += 1
    
    return temp_citation_context_list

def load_kim_dataset(data_filename = '/export/scratch/zeren/KimNLP/RawData/dec_12_annotations.tsv'):
    data = pd.read_csv(data_filename, sep='\t')

    data['annotation'] = data.apply(lambda x: merge_annotations(x['previous annotation'], x['new annotation']), axis=1)
    columns = ['cited title', 'cited abstract', 'citing title', 'citing abstract', 'citation context', 'annotation']
    data = data[columns]
    data = data.rename(columns={
        'cited title': 'cited_title', 
        'cited abstract': 'cited_abstract', 
        'citing title': 'citing_title', 
        'citing abstract': 'citing_abstract', 
        'citation context': 'citation_context', 
        'annotation': 'label'
    }).reset_index().drop(columns=['index', 'cited_title', 'citing_title'])
    data = data.fillna('')
    print('kim: Number of instances: {}'.format(data.shape[0]))

    num_train = int(data.shape[0] * 0.7)
    num_val = int(data.shape[0] * 0.2)
    num_test = data.shape[0] - num_train - num_val
    
    split = ['train'] * num_train + ['val'] * num_val + ['test'] * num_test
    np.random.shuffle(split)

    data['split'] = split
    data.to_csv(
        '/export/scratch/zeren/KimNLP/NewData/old_kim.tsv', 
        sep='\t',
        header=True,
        index=False
    )

class Dataset(object):
    def __init__(self, dataframe, modelname, max_length=512):
        self.max_length = max_length if max_length > 0 else None
        self.truncation = (max_length > 0)
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)

        self._load_data(dataframe)
        self._tokenize()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        return (
            self.citation_context_tokens[idx],
            self.citing_abstract_tokens[idx],
            self.cited_abstract_tokens[idx],
            self.labels[idx]
        )

    def get_label_weights(self):
        labels, counts = torch.unique(self.labels, return_counts=True)

        label_weights = torch.zeros_like(counts, dtype=torch.float32)
        label_weights[labels] = counts.max() / counts

        return label_weights

    def _load_data(self, annotated_data):
        self.labels = annotated_data['label'].values
        self.citation_context = [process_contexts(ctx) for ctx in annotated_data['citation_context'].values]
        self.citing_abstract = annotated_data['citing_abstract'].values
        self.cited_abstract = annotated_data['cited_abstract'].values

        self.labels = torch.LongTensor(self.labels)

    def _tokenize(self):
        self.citation_context_tokens = list()
        self.citing_abstract_tokens = list()
        self.cited_abstract_tokens = list()

        print('Tokenizing citation contexts.')
        for context in tqdm(self.citation_context):
            token_list = list()
            for ctx in context:
                tokens = self.tokenizer(
                    ctx,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=self.truncation
                )
                token_list.append(tokens)
            self.citation_context_tokens.append(token_list)
        
        print('Tokenizing citing abstracts.')
        for abstract in tqdm(self.citing_abstract):
            tokens = self.tokenizer(
                abstract,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=self.truncation
            )
            self.citing_abstract_tokens.append(tokens)
        
        print('Tokenizing cited abstracts.')
        for abstract in tqdm(self.cited_abstract):
            tokens = self.tokenizer(
                abstract,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=self.truncation
            )
            self.cited_abstract_tokens.append(tokens)

def create_data_channels(filename, modelname, max_length):
    data = pd.read_csv(filename, sep='\t')
    data = data.fillna(' ')

    print('Number of data instance: {}'.format(data.shape[0]))

    unique_labels = ['not used', 'used']
    label2id = {'used': 1, 'not used': 0, 'extended': 0}  # binary for now
    data = data[data['label'].isin(label2id.keys())]
    data['label'] = data['label'].apply(
        lambda x: label2id[x])

    data_train = data[data['split'] == 'train'].reset_index()
    data_val = data[data['split'] == 'val'].reset_index()
    data_test = data[data['split'] == 'test'].reset_index()

    train_data = Dataset(
        data_train,
        modelname=modelname,
        max_length=max_length
    )
    val_data = Dataset(
        data_val,
        modelname=modelname,
        max_length=max_length
    )
    test_data = Dataset(
        data_test,
        modelname=modelname,
        max_length=max_length
    )

    return train_data, val_data, test_data


if __name__ == '__main__':
    load_kim_dataset()