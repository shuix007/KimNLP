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
    return '[SEP]'.join(temp_citation_context_list)

def load_kim_dataset(data_filename = '/export/scratch/zeren/KimNLP/RawData/dec_12_annotations.tsv'):
    data = pd.read_csv(data_filename, sep='\t')

    data['annotation'] = data.apply(lambda x: merge_annotations(x['previous annotation'], x['new annotation']), axis=1)
    columns = ['cited doi', 'cited title', 'cited abstract', 'citing doi', 'citing title', 'citing abstract', 'citation context', 'annotation']
    data = data[columns]
    data = data.rename(columns={
        'cited title': 'cited_title', 
        'cited abstract': 'cited_abstract', 
        'citing title': 'citing_title', 
        'citing abstract': 'citing_abstract', 
        'citation context': 'citation_context', 
        'annotation': 'label'
    }).reset_index().drop(columns=['index', 'cited_title', 'citing_title'])
    data = data[data['label'] != 'not sure']
    data = data.fillna('')
    data['citation_context'] = data['citation_context'].apply(process_contexts)
    # print(data['citation_context'].values[0])
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

class CollateFn(object):
    def __init__(self, modelname):
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)

    def __call__(self, samples):
        text, labels = list(map(list, zip(*samples)))
        batched_text = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        labels = torch.stack(labels)
        return batched_text, labels

class Dataset(object):
    def __init__(self, dataframe, id2label=None, mode='context'):
        self.mode = mode
        self.id2label = id2label
        self._load_data(dataframe)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        # if self.mode == 'context':
        #     return (self.citation_context[idx], self.labels[idx])
        # elif self.mode == 'abstract':
        #     return (self.citing_abstract[idx] + '<SEP>' + self.cited_abstract[idx], self.labels[idx])
        return (self.text[idx], self.labels[idx])

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
        self.labels = annotated_data['label'].tolist()
        self.citation_context = annotated_data['citation_context'].tolist()
        self.citing_abstract = annotated_data['citing_abstract'].tolist()
        self.cited_abstract = annotated_data['cited_abstract'].tolist()

        if self.mode == 'context':
            self.text = self.citation_context
        elif self.mode == 'abstract':
            self.text = [citing_abs + '<SEP>' + cited_abs for citing_abs, cited_abs in zip(self.citing_abstract, self.cited_abstract)]
        elif self.mode == 'all':
            self.text = [context + '<SEP>' + citing_abs + '<SEP>' + cited_abs for context, citing_abs, cited_abs in zip(self.citation_context, self.citing_abstract, self.cited_abstract)]
        elif self.mode == 'mix-abstract-all':
            abstract = [citing_abs + '<SEP>' + cited_abs for citing_abs, cited_abs in zip(self.citing_abstract, self.cited_abstract)]
            all_text = [context + '<SEP>' + citing_abs + '<SEP>' + cited_abs for context, citing_abs, cited_abs in zip(self.citation_context, self.citing_abstract, self.cited_abstract)]

            self.text = abstract + all_text
            self.labels = self.labels + self.labels
        elif self.mode == 'mix-abstract-context':
            context = self.citation_context
            abstract = [citing_abs + '<SEP>' + cited_abs for citing_abs, cited_abs in zip(self.citing_abstract, self.cited_abstract)]

            self.text = context + abstract
            self.labels = self.labels + self.labels
        elif self.mode == 'mix':
            context = self.citation_context
            abstract = [citing_abs + '<SEP>' + cited_abs for citing_abs, cited_abs in zip(self.citing_abstract, self.cited_abstract)]
            all_text = [context + '<SEP>' + citing_abs + '<SEP>' + cited_abs for context, citing_abs, cited_abs in zip(self.citation_context, self.citing_abstract, self.cited_abstract)]

            self.text = context + abstract + all_text
            self.labels = self.labels + self.labels + self.labels

        self.labels = torch.LongTensor(self.labels)

def create_data_channels(filename, mode):
    data = pd.read_csv(filename, sep='\t')
    data = data.fillna(' ')

    print('Number of data instance: {}'.format(data.shape[0]))

    # map labels to ids
    unique_labels = data['label'].unique()
    if 'used' in unique_labels:
        label2id = {'used': 1, 'not used': 0, 'extended': 0}  # binary for now
    else:
        label2id = {lb: i for i, lb in enumerate(unique_labels)}
    
    data['label'] = data['label'].apply(
        lambda x: label2id[x])

    data_train = data[data['split'] == 'train'].reset_index()
    data_val = data[data['split'] == 'val'].reset_index()
    data_test = data[data['split'] == 'test'].reset_index()

    train_data = Dataset(
        data_train,
        mode=mode
    )
    val_data = Dataset(
        data_val,
        mode=mode
    )
    context_only_test_data = Dataset(
        data_test,
        mode='context',
        id2label=unique_labels
    )
    abstract_only_test_data = Dataset(
        data_test,
        mode='abstract',
        id2label=unique_labels
    )
    both_test_data = Dataset(
        data_test,
        mode='all',
        id2label=unique_labels
    )

    return train_data, val_data, (context_only_test_data, abstract_only_test_data, both_test_data)


if __name__ == '__main__':
    load_kim_dataset()
    load_acl_scicite_dataset('/export/scratch/zeren/KimNLP/RawData/datasets', 'acl_arc')
    load_acl_scicite_dataset('/export/scratch/zeren/KimNLP/RawData/datasets', 'scicite')