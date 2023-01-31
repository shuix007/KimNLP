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
    
    k = len(temp_citation_context_list)
    truncate_len = 256 - int(256 / k)

    # context_list = [' '.join(c.split(' ')[256-truncate_len:len(c)-truncate_len]) for c in temp_citation_context_list]
    return '[SEP]'.join(temp_citation_context_list)

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
    data['citation_context'] = data['citation_context'].apply(process_contexts)
    print(data['citation_context'].values[0])
    print('kim: Number of instances: {}'.format(data.shape[0]))

    num_train = int(data.shape[0] * 0.6)
    num_val = int(data.shape[0] * 0.2)
    num_test = data.shape[0] - num_train - num_val
    
    split = ['train'] * num_train + ['val'] * num_val + ['test'] * num_test
    np.random.shuffle(split)

    data['split'] = split
    data.to_csv(
        '/export/scratch/zeren/KimNLP/NewData/kim.tsv', 
        sep='\t',
        header=True,
        index=False
    )

def load_acl_scicite_dataset(data_dir, data_name):
    train_data_filename = os.path.join(
        data_dir, '{}_train_with_abstracts.jsonl'.format(data_name)
    )
    test_data_filename = os.path.join(
        data_dir, '{}_test_with_abstracts.jsonl'.format(data_name)
    )

    with open(train_data_filename, 'r') as train_f:
        train_list = train_f.readlines()
    
    with open(test_data_filename, 'r') as test_f:
        test_list = test_f.readlines()
    
    train_data = []
    test_data = []
    for json_str in train_list:
        json_dict = json.loads(json_str)
        train_data.append(json_dict)
    
    for json_str in test_list:
        json_dict = json.loads(json_str)
        test_data.append(json_dict)

    train_df = pd.DataFrame(train_data).drop(columns=['cited_title', 'citing_title'])
    test_df = pd.DataFrame(test_data).drop(columns=['cited_title', 'citing_title'])

    print('{}: Number of instances: {}'.format(data_name, train_df.shape[0]+test_df.shape[0]))

    train_df = train_df.fillna('')
    test_df = test_df.fillna('')

    num_train = int(train_df.shape[0] * 0.8)
    num_val = train_df.shape[0] - num_train
    num_test = test_df.shape[0]

    train_split = ['train'] * num_train + ['val'] * num_val
    test_split = ['test'] * num_test
    np.random.shuffle(train_split)

    train_df['split'] = train_split
    test_df['split'] = test_split

    data = pd.concat([train_df, test_df], axis=0)

    data.to_csv(
        '/export/scratch/zeren/KimNLP/NewData/{}.tsv'.format(data_name), 
        sep='\t',
        header=True, 
        index=False
    )

class Dataset(object):
    def __init__(self, dataframe, modelname, max_length=512, id2label=None):
        self.max_length = max_length if max_length > 0 else None
        self.truncation = (max_length > 0)
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.id2label = id2label

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

    def write_prediction(self, preds, filename1, filename2):
        assert len(self.labels) == len(
            preds), 'the length of labels should be the same with predictions.'

        np_label_idx = self.labels.numpy()
        pred_label_idx = preds.argmax(axis=1)
        softmax_scores = np.exp(
            preds) / np.exp(preds).sum(axis=1, keepdims=True)

        ground_truth_labels = []
        pred_labels = []
        contexts = []

        for i, context_tokens in enumerate(self.fused_citation_context_tokens):
            ground_truth_labels.append(self.id2label[np_label_idx[i]])
            pred_labels.append(self.id2label[pred_label_idx[i]])
            contexts.append(self.tokenizer.decode(
                context_tokens['input_ids'][0].tolist()))

        pd.DataFrame({
            'input_text': contexts,
            'label': ground_truth_labels,
            'pred_label': pred_labels
        }).to_csv(filename1, index=False, header=True)

        data_dict = {
            'input_text': contexts,
            'label': ground_truth_labels
        }

        for i in range(softmax_scores.shape[1]):
            data_dict[self.id2label[i]] = softmax_scores[:, i]
        pd.DataFrame(data_dict).to_csv(filename2, index=False, header=True)

    def get_label_weights(self):
        labels, counts = torch.unique(self.labels, return_counts=True)

        label_weights = torch.zeros_like(counts, dtype=torch.float32)
        label_weights[labels] = counts.max() / counts

        return label_weights

    def _load_data(self, annotated_data):
        self.labels = annotated_data['label'].values
        self.citation_context = annotated_data['citation_context'].values
        self.citing_abstract = annotated_data['citing_abstract'].values
        self.cited_abstract = annotated_data['cited_abstract'].values

        self.labels = torch.LongTensor(self.labels)

    def _tokenize(self):
        self.citation_context_tokens = list()
        self.citing_abstract_tokens = list()
        self.cited_abstract_tokens = list()

        print('Tokenizing citation contexts.')
        for context in tqdm(self.citation_context):
            tokens = self.tokenizer(
                context,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=self.truncation
            )
            self.citation_context_tokens.append(tokens)
        
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

    # map labels to ids
    unique_labels = data['label'].unique()
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    # label2id = {'used': 1, 'not used': 0, 'extended': 0}  # binary for now

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
        max_length=max_length,
        id2label=unique_labels
    )

    return train_data, val_data, test_data


if __name__ == '__main__':
    load_kim_dataset()
    load_acl_scicite_dataset('/export/scratch/zeren/KimNLP/RawData/datasets', 'acl_arc')
    load_acl_scicite_dataset('/export/scratch/zeren/KimNLP/RawData/datasets', 'scicite')