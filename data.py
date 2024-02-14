import os
import json
import copy
import torch
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax

from transformers import AutoTokenizer

class CollateFn(object):
    def __init__(self, modelname, class_definitions=None, instance_weights=False):
        self.instance_weights = instance_weights
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.cited_here_tokens = self.tokenizer('<CITED HERE>', return_tensors='pt')[
            'input_ids'].squeeze()[1:-1]

        if class_definitions is not None:
            self.class_definitions = []
            self.class_head_indices = []
            for i, defs in enumerate(class_definitions):
                self.class_definitions += defs
                self.class_head_indices.append(i * torch.ones(len(defs), dtype=torch.long))
            self.class_head_indices = torch.cat(self.class_head_indices, dim=0)
            self.class_tokens = self.tokenizer(
                self.class_definitions,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

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
        if self.instance_weights:
            text, labels, ds_indices, instance_weights = list(map(list, zip(*samples)))
            batched_text = self._tokenize_context(text)
            labels = torch.stack(labels)
            ds_indices = torch.stack(ds_indices)
            instance_weights = torch.stack(instance_weights)
            return batched_text, labels, ds_indices, instance_weights
        else:
            text, labels, ds_indices = list(map(list, zip(*samples)))
            batched_text = self._tokenize_context(text)
            labels = torch.stack(labels)
            ds_indices = torch.stack(ds_indices)

            # print(torch.unique(ds_indices, return_counts=True))
            return batched_text, labels, ds_indices, copy.deepcopy(self.class_tokens), self.class_head_indices

# class PLDataset(object):
#     def __init__(self, dataframe, class_definitions):
#         self.class_definitions = class_definitions
#         self._load_data(dataframe)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         '''Get datapoint with index'''
#         return (self.text[idx], self.labels[idx], self.ds_index[idx], self.instance_weights[idx])

#     def compute_confusion_matrix(self, preds):
#         this_label_space = len(np.unique(self.original_labels.numpy()))
#         model_label_space = preds.shape[1]
#         confusion_matrix = np.zeros((this_label_space, model_label_space), dtype=np.int64)

#         pred_label_idx = preds.argmax(axis=1)
#         for i in range(len(self.original_labels)):
#             confusion_matrix[self.original_labels[i], pred_label_idx[i]] += 1
#         return confusion_matrix

#     def visualize_confusion_matrix(self, preds, this_label_map, model_label_map):
#         confusion_matrix = self.compute_confusion_matrix(preds)

#         df = pd.DataFrame(confusion_matrix, index=this_label_map, columns=model_label_map)
#         print(df)

#     def pseudo_label(self, preds):
#         pred_label_idx = preds.argmax(axis=1)
#         self.labels = torch.LongTensor(pred_label_idx)

#     def update_label_with_selection(self, preds):
#         confusion_matrix = self.compute_confusion_matrix(preds)
#         confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

#         pred_label_idx = preds.argmax(axis=1)
#         self.labels = torch.LongTensor(pred_label_idx)
#         for i in range(len(pred_label_idx)):
#             self.instance_weights[i] = confusion_matrix[self.original_labels[i], pred_label_idx[i]]

#     def _load_data(self, annotated_data):
#         self.labels = torch.LongTensor(annotated_data['label'].tolist())
#         self.original_labels = torch.LongTensor(annotated_data['label'].tolist())
#         self.ds_index = torch.zeros_like(self.original_labels)
#         self.instance_weights = torch.ones_like(self.original_labels).float()
#         self.text = annotated_data['context'].tolist()
        
class PLDataset(object):
    # def __init__(self, dataframe, class_definitions):
    #     self.class_definitions = class_definitions
    #     self._load_data(dataframe)

    def __init__(self, text, labels, original_labels, ds_index, class_definitions, lmbd=1.0):
        self.class_definitions = class_definitions
        self.labels = labels
        self.original_labels = original_labels
        self.ds_index = ds_index
        self.text = text
        self.lmbd = lmbd

    @classmethod
    def from_dataframe(cls, dataframe, class_definitions, lmbd=1.0):
        labels = torch.LongTensor(dataframe['label'].tolist())
        original_labels = torch.LongTensor(dataframe['label'].tolist())
        ds_index = torch.zeros_like(original_labels)
        text = dataframe['context'].tolist()
        return cls(text, labels, original_labels, ds_index, class_definitions, lmbd=lmbd)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        return (self.text[idx], self.labels[idx], self.ds_index[idx])

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

    def pseudo_label(self, preds, threshold=0.7):
        pred_label_idx = preds.argmax(axis=1)
        probs = softmax(preds, axis=1)
        confident_prediction_idx = np.where(probs.max(axis=1) > threshold)[0]

        if len(confident_prediction_idx) > 0:
            labels = torch.LongTensor([pred_label_idx[idx] for idx in confident_prediction_idx])
            original_labels = torch.LongTensor([self.original_labels[idx] for idx in confident_prediction_idx])
            ds_index = torch.zeros_like(labels)
            text = [self.text[idx] for idx in confident_prediction_idx]

            return PLDataset(text, labels, original_labels, ds_index, self.class_definitions)
        else:
            return None

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
        self.text = annotated_data['context'].tolist()

class Dataset(object):
    def __init__(self, dataframe, class_definitions, lmbd=1.0):
        self.class_definitions = class_definitions
        self.lmbd = lmbd
        self._load_data(dataframe)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        return (self.text[idx], self.labels[idx], self.ds_index[idx])

    def _load_data(self, annotated_data):
        self.labels = torch.LongTensor(annotated_data['label'].tolist())
        self.original_labels = torch.LongTensor(annotated_data['label'].tolist())
        self.ds_index = torch.zeros_like(self.original_labels)
        self.text = annotated_data['context'].tolist()

class Datasets(object):
    def __init__(self, datasets):
        self.text = []
        self.labels = []
        self.class_definitions = []
        self.lambdas = []
        for i, d in enumerate(datasets):
            self.text += d.text
            self.labels.append(d.labels)
            self.class_definitions.append(d.class_definitions)
            self.lambdas.append(d.lmbd)
        self.labels = torch.cat(self.labels, dim=0)
        self.ds_index = torch.zeros_like(self.labels) # torch.cat(self.ds_index, dim=0)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        '''Get datapoint with index'''
        return (self.text[idx], self.labels[idx], self.ds_index[idx])

# class MultiHeadDatasets(object):
#     def __init__(self, datasets):
#         self.text = []
#         self.ds_index = []
#         self.labels = []
#         self.class_definitions = []
#         self.lambdas = []
#         for i, d in enumerate(datasets):
#             self.text += d.text
#             self.ds_index.append(i * torch.ones(len(d.text), dtype=torch.long))
#             self.labels.append(d.labels)
#             self.class_definitions.append(d.class_definitions)
#             self.lambdas.append(d.lmbd)
#         self.labels = torch.cat(self.labels, dim=0)
#         self.ds_index = torch.cat(self.ds_index, dim=0)
    
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         '''Get datapoint with index'''
#         return (self.text[idx], self.labels[idx], self.ds_index[idx])

class MultiHeadDatasets(object):
    def __init__(self, datasets, batch_size_factor=2):
        self.text = []
        self.ds_index = []
        self.labels = []
        self.class_definitions = []
        self.lambdas = []

        self.dataset_sizes = [len(d.labels) for d in datasets]
        if len(self.dataset_sizes) > 1:
            if sum(self.dataset_sizes) / self.dataset_sizes[0] <= batch_size_factor:
            # if self.dataset_sizes[0] >= sum(self.dataset_sizes[1:]):
                self.sample_auxiliary = False
                self.adjusted_batch_size_factor = sum(self.dataset_sizes) / self.dataset_sizes[0] # <= batch_size_factor
            else:
                self.sample_auxiliary = True
                self.sample_distribution = np.array([d.lmbd for d in datasets[1:]]) / sum([d.lmbd for d in datasets[1:]])
                self.adjusted_batch_size_factor = batch_size_factor
        else:
            self.sample_auxiliary = False
            self.adjusted_batch_size_factor = 1

        for i, d in enumerate(datasets):
            self.text += d.text
            self.ds_index.append(i * torch.ones(len(d.text), dtype=torch.long))
            self.labels.append(d.labels)
            self.class_definitions.append(d.class_definitions)
            self.lambdas.append(d.lmbd)
        self.labels = torch.cat(self.labels, dim=0)
        self.ds_index = torch.cat(self.ds_index, dim=0)
    
    def sample_auxiliary_instace(self):
        sampled_dataset_idx = np.random.choice(
            np.arange(1, len(self.dataset_sizes)), 
            p=self.sample_distribution
        )
        instance_idx = np.random.choice(
            self.dataset_sizes[sampled_dataset_idx]
        ) + sum(self.dataset_sizes[:sampled_dataset_idx])
        return instance_idx

    def __len__(self):
        if self.sample_auxiliary: # if the auxiliary dataset is larger than the main dataset
            return self.dataset_sizes[0] * self.adjusted_batch_size_factor
        return len(self.labels)
    
    def __getitem__(self, idx):
        '''Get datapoint with index'''
        if idx < self.dataset_sizes[0] or not self.sample_auxiliary:
            return (self.text[idx], self.labels[idx], self.ds_index[idx])
        else:
            real_idx = self.sample_auxiliary_instace()
            return (self.text[real_idx], self.labels[real_idx], self.ds_index[real_idx])

def load_class_definitions(filename):
    with open(filename, 'r') as f:
        class_definitions = json.load(f)
    
    results = {k:{} for k in class_definitions.keys()}
    for k, v in class_definitions.items():
        for kk, vv in v.items():
            results[k][kk.lower()] = vv
    return results

def create_data_channels(filename, class_definition_filename, split=None, pl=False, lmbd=1.0):
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

    class_definitions = load_class_definitions(class_definition_filename)
    dataname = filename.split('/')[-1].split('.')[0]
    data_class_definitions = [class_definitions[dataname][lb.lower()] for lb in unique_labels]

    if pl:
        train_data = PLDataset.from_dataframe(data_train, data_class_definitions, lmbd=lmbd)
        val_data = PLDataset.from_dataframe(data_val, data_class_definitions, lmbd=lmbd)
        test_data = PLDataset.from_dataframe(data_test, data_class_definitions, lmbd=lmbd)
    else:
        train_data = Dataset(data_train, data_class_definitions, lmbd=lmbd)
        val_data = Dataset(data_val, data_class_definitions, lmbd=lmbd)
        test_data = Dataset(data_test, data_class_definitions, lmbd=lmbd)

    return train_data, val_data, test_data, unique_labels

def create_single_data_object(filename, class_definition_filename, split=None, pl=False, lmbd=1.0):
    data = pd.read_csv(filename, sep='\t')
    data = data.fillna(' ')

    print('Number of data instance: {}'.format(data.shape[0]))

    # map labels to ids
    unique_labels = data['label'].unique()
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    
    data['label'] = data['label'].apply(
        lambda x: label2id[x])

    class_definitions = load_class_definitions(class_definition_filename)
    dataname = filename.split('/')[-1].split('.')[0]
    data_class_definitions = [class_definitions[dataname][lb.lower()] for lb in unique_labels]

    if pl:
        if split is None:
            return PLDataset.from_dataframe(data, data_class_definitions, lmbd=lmbd), unique_labels
        else:
            return PLDataset.from_dataframe(data[data['split'] == split].reset_index(), data_class_definitions, lmbd=lmbd), unique_labels
    else:
        if split is None:
            return Dataset(data, data_class_definitions, lmbd=lmbd), unique_labels
        else:
            return Dataset(data[data['split'] == split].reset_index(), data_class_definitions, lmbd=lmbd), unique_labels