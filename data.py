import os
import json
import torch
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer

class CollateFn(object):
    def __init__(self, modelname):
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.cited_here_tokens = self.tokenizer('<CITED HERE>', return_tensors='pt')[
            'input_ids'].squeeze()[1:-1]

    def _get_readout_mask(self, tokens):
        # cited_here_tokens = torch.tensor([962, 8412, 1530, 1374])
        readout_mask = torch.zeros_like(tokens['input_ids'], dtype=torch.bool)

        batch_size = tokens['input_ids'].size(0)
        l = tokens['input_ids'].size(1)
        ctk_l = self.cited_here_tokens.size(0)
        for b in range(batch_size):
            for i in range(1, l - ctk_l):
                if torch.equal(tokens['input_ids'][b, i:i+ctk_l], self.cited_here_tokens):
                    readout_mask[b, i:i+ctk_l] = True
        return readout_mask

    def _tokenize_context(self, context):
        tokens = self.tokenizer(
            context,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        tokens['readout_mask'] = self._get_readout_mask(
            tokens
        )

        return tokens

    def __call__(self, samples):
        text, labels, ds_indices, instance_weights = list(map(list, zip(*samples)))
        batched_text = self._tokenize_context(text)
        labels = torch.stack(labels)
        ds_indices = torch.stack(ds_indices)
        instance_weights = torch.stack(instance_weights)
        return batched_text, labels, ds_indices, instance_weights

class Dataset(object):
    def __init__(self, dataframe, id2label=None):
        self.id2label = id2label
        self._load_data(dataframe)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        return (self.text[idx], self.labels[idx], self.ds_index[idx], self.instance_weights[idx])

    def compute_confusion_matrix(self, preds):
        this_label_space = len(np.unique(self.original_labels.numpy()))
        model_label_space = preds.shape[1]
        confusion_matrix = np.zeros((this_label_space, model_label_space), dtype=np.int64)

        pred_label_idx = preds.argmax(axis=1)
        for i in range(len(self.original_labels)):
            confusion_matrix[self.original_labels[i], pred_label_idx[i]] += 1
        return confusion_matrix

    def visualize_confusion_matrix(self, preds, this_label_map, model_label_map):
        confusion_matrix = self.compute_confusion_matrix(preds)

        df = pd.DataFrame(confusion_matrix, index=this_label_map, columns=model_label_map)
        print(df)

    def pseudo_label(self, preds):
        pred_label_idx = preds.argmax(axis=1)
        self.labels = torch.LongTensor(pred_label_idx)

    def update_label_with_selection(self, preds):
        confusion_matrix = self.compute_confusion_matrix(preds)
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

        pred_label_idx = preds.argmax(axis=1)
        self.labels = torch.LongTensor(pred_label_idx)
        for i in range(len(pred_label_idx)):
            self.instance_weights[i] = confusion_matrix[self.original_labels[i], pred_label_idx[i]]

    def _load_data(self, annotated_data):
        self.labels = torch.LongTensor(annotated_data['label'].tolist())
        self.original_labels = torch.LongTensor(annotated_data['label'].tolist())
        self.ds_index = torch.zeros_like(self.original_labels)
        self.instance_weights = torch.ones_like(self.original_labels).float()
        self.text = annotated_data['context'].tolist()

class Datasets(object):
    def __init__(self, datasets):
        self.text = []
        # self.ds_index = []
        self.labels = []
        self.instance_weights = []
        for i, d in enumerate(datasets):
            self.text += d.text
            # self.ds_index.append(torch.full(len(d.text), i, dtype=torch.long))
            self.labels.append(d.labels)
            self.instance_weights.append(d.instance_weights)
        self.labels = torch.cat(self.labels, dim=0)
        self.instance_weights = torch.cat(self.instance_weights, dim=0)
        self.ds_index = torch.zeros_like(self.labels) # torch.cat(self.ds_index, dim=0)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        '''Get datapoint with index'''
        return (self.text[idx], self.labels[idx], self.ds_index[idx], self.instance_weights[idx])

def create_data_channels(filename, split=None):
    data = pd.read_csv(filename, sep='\t')
    data = data.fillna(' ')

    print('Number of data instance: {}'.format(data.shape[0]))

    # map labels to ids
    unique_labels = data['label'].unique().tolist()
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    
    data['label'] = data['label'].apply(
        lambda x: label2id[x])

    data_train = data[data['split'] == 'train'].reset_index()
    data_val = data[data['split'] == 'val'].reset_index()
    data_test = data[data['split'] == 'test'].reset_index()

    train_data = Dataset(data_train)
    val_data = Dataset(data_val)
    test_data = Dataset(data_test)

    return train_data, val_data, test_data, unique_labels

def create_single_data_object(filename, split=None):
    data = pd.read_csv(filename, sep='\t')
    data = data.fillna(' ')

    print('Number of data instance: {}'.format(data.shape[0]))

    # map labels to ids
    unique_labels = data['label'].unique()
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    
    data['label'] = data['label'].apply(
        lambda x: label2id[x])

    if split is None:
        return Dataset(data), unique_labels
    else:
        return Dataset(data[data['split'] == split].reset_index()), unique_labels