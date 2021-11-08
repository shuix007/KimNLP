import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer


class Dataset(object):
    def __init__(self, annotated_data, modelname, fuse_type, max_length=512):
        self.fuse_type = fuse_type
        self.early_fuse = fuse_type in ['bruteforce']  # control the way to fetch
        self.max_length = max_length if max_length > 0 else None
        self.truncation = (max_length > 0)
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)

        self.cited_here_tokens = self.tokenizer('<CITED HERE>', return_tensors='pt')[
            'input_ids'].squeeze()[1:-1]

        self._load_data(annotated_data)
        if self.early_fuse:
            self._early_fusion_context_only()
        else:
            self._late_fusion_context_only()

        self._index = -1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        if self.early_fuse:
            return [
                self.fused_citation_context_tokens[idx],
                self.labels[idx]
            ]
        else:
            return [
                self.citation_context_tokens[idx],
                self.labels[idx]
            ]

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        if self._index >= len(self.labels):
            self._index = -1
            raise StopIteration
        else:
            return self.__getitem__(self._index)

    def get_label_weights(self):
        labels, counts = torch.unique(self.labels, return_counts=True)

        label_weights = torch.zeros_like(counts, dtype=torch.float32)
        label_weights[labels] = counts.max() / counts

        return label_weights

    def _load_data(self, annotated_data):
        self.labels = []
        self.citation_context_list = []

        for i in range(annotated_data.shape[0]):
            self.labels.append(annotated_data['label'].values[i])

            # Since there multiple citation contexts, I've separated the with ---(j)--- in the spreadsheet, should ideally use regex to split them, here is a random way
            citing_contexts = str(annotated_data['citation_context'].values[i]).split(
                '---(1)---')[1].strip()
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
            self.citation_context_list.append(temp_citation_context_list)

        self.labels = torch.LongTensor(self.labels)

    def _get_readout_index(self, tokens):
        # cited_here_tokens = torch.tensor([962, 8412, 1530, 1374])
        readout_index = []

        l = tokens['input_ids'].size(1)
        ctk_l = self.cited_here_tokens.size(0)
        for i in range(1, l - ctk_l):
            if torch.equal(tokens['input_ids'][0, i:i+ctk_l], self.cited_here_tokens):
                readout_index.append(torch.arange(i, i+ctk_l))
        return torch.cat(readout_index)

    def _get_readout_mask(self, tokens, readout_index):
        mask = torch.zeros_like(tokens['input_ids'], dtype=torch.bool)
        mask[0, readout_index] = True
        return mask

    def _tokenize_context(self, context):
        tokens = self.tokenizer(
            context,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation
        )
        tokens['readout_mask'] = self._get_readout_mask(
            tokens,
            self._get_readout_index(tokens)
        )

        return tokens

    def _early_fusion_context_only(self):
        self.fused_citation_context_tokens = list()

        print('Tokenizing early fused tokens (context only).')
        for i in tqdm(range(len(self.citation_context_list))):
            fused_text = ' [SEP] '.join(self.citation_context_list[i])
            fused_tokens = self._tokenize_context(fused_text)

            self.fused_citation_context_tokens.append(fused_tokens)

    def _late_fusion_context_only(self):
        self.citation_context_tokens = list()

        print('Tokenizing late fused tokens (context only).')
        for i in tqdm(range(len(self.citation_context_list))):
            self.citation_context_tokens.append([
                self._tokenize_context(
                    ctx
                ) for ctx in self.citation_context_list[i]
            ])


class EmbeddedDataset(object):
    def __init__(self, dataset, lm_model, fuse_type, inter_context_pooling):
        self.fuse_type = fuse_type
        self.early_fuse = fuse_type in ['bruteforce']
        self.inter_context_pooling = inter_context_pooling

        assert self.inter_context_pooling in [
            'sum', 'max', 'mean', 'topk'], 'Inter context pooling type {} not supported'.format(self.inter_context_pooling)

        self.labels = dataset.labels

        if self.early_fuse:
            self._compute_early_fused_embeddings(dataset, lm_model)
        else:
            self._compute_late_fused_embeddings(dataset, lm_model)


    def get_label_weights(self):
        labels, counts = torch.unique(self.labels, return_counts=True)

        label_weights = torch.zeros_like(counts, dtype=torch.float32)
        label_weights = counts.max() / counts

        return label_weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.early_fuse:
            return [
                self.fused_context_embeds[idx],
                self.labels[idx]
            ]
        else:
            return [
                self.citation_context_embeds[idx],
                self.labels[idx]
            ]

    def _compute_early_fused_embeddings(self, dataset, model):
        self.fused_context_embeds = list()

        model.eval()
        with torch.no_grad():
            print('Computing early-fused embeddings.')
            for fused_context, _ in tqdm(dataset):
                lm_output = model.context_forward(fused_context).cpu()
                self.fused_context_embeds.append(lm_output)
            self.fused_context_embeds = torch.stack(self.fused_context_embeds)

    def _compute_late_fused_embeddings(self, dataset, model):
        self.citation_context_embeds = list()

        model.eval()
        with torch.no_grad():
            print('Computing late-fused embeddings.')
            for citation_context, _ in tqdm(dataset):
                citation_context_embeds = list()
                for context in citation_context:
                    context_embed = model.context_forward(context)
                    citation_context_embeds.append(context_embed)
                citation_context_embeds = torch.stack(citation_context_embeds)

                if self.inter_context_pooling == 'sum':
                    citation_context_embeds = citation_context_embeds.sum(
                        dim=0).cpu()
                elif self.inter_context_pooling == 'max':
                    citation_context_embeds = citation_context_embeds.max(
                        dim=0).values.cpu()
                elif self.inter_context_pooling == 'mean':
                    citation_context_embeds = citation_context_embeds.mean(
                        dim=0).cpu()
                elif self.inter_context_pooling == 'topk':
                    topk = citation_context_embeds.topk(10, dim=0).cpu()
                    citation_context_embeds = topk[0].mean(dim=0).cpu()

                self.citation_context_embeds.append(citation_context_embeds)

            self.citation_context_embeds = torch.stack(
                self.citation_context_embeds)


class MultiHeadDatasets(object):
    def __init__(self, datasets, p=None):
        self.datasets = datasets
        self.p = p if p is not None else np.full(len(self.datasets), 1. / len(
            self.datasets), dtype=np.float32)  # sample probability for the datasets

        self.lengths = np.array([len(dataset) for dataset in self.datasets])
        self.main_length = self.lengths[0]

    def __len__(self):
        return self.lengths[0]

    def __getitem__(self, idx):
        indices = self._get_indices(idx)

        instances = [d.__getitem__(indices[i])
                     for i, d in enumerate(self.datasets)]
        return instances

    def _get_indices(self, idx):
        indices = list()
        for i, l in enumerate(self.lengths):
            if l < self.main_length:
                indices.append(int((idx / self.main_length) * l))
            elif l > self.main_length:
                portion = l / self.main_length
                idx_low, idx_high = int(idx*portion), int((idx+1)*portion)
                indices.append(np.random.randint(idx_low, idx_high))
            else:
                indices.append(idx)
        return indices

    def get_label_weights(self):
        return [d.get_label_weights() for d in self.datasets]


class SingleHeadDatasets(object):
    def __init__(self, datasets, p=None):
        self.datasets = datasets
        self.p = p if p is not None else np.full(len(self.datasets), 1. / len(
            self.datasets), dtype=np.float32)  # sample probability for the datasets

        self.lengths = np.array([len(dataset) for dataset in self.datasets])
        self.total_length = np.sum(self.lengths)
        self.main_length = self.lengths[0]

        self._compute_label_weights()

        self._index = -1
        self._data_index = 0

    def _compute_label_weights(self):
        ''' Compute the weights of the labels
        '''
        self.num_labels = np.cumsum(
            [0] + [len(d.get_label_weights()) for d in self.datasets])
        labels = torch.cat([d.labels + self.num_labels[i]
                           for i, d in enumerate(self.datasets)])

        labels, counts = torch.unique(labels, return_counts=True)

        self.label_weights = torch.zeros_like(counts, dtype=torch.float32)
        self.label_weights[labels] = counts.max() / counts

    def __len__(self):
        return self.lengths[0]

    def __getitem__(self, idx):
        indices = self._get_indices(idx)

        instances = [d.__getitem__(indices[i])
                     for i, d in enumerate(self.datasets)]
        for i in range(len(instances)):
            instances[i][-1] = instances[i][-1] + \
                self.num_labels[i]  # map the labels

        return instances

    def _get_indices(self, idx):
        indices = list()
        for i, l in enumerate(self.lengths):
            if l < self.main_length:
                indices.append(int((idx / self.main_length) * l))
            elif l > self.main_length:
                portion = l / self.main_length
                idx_low, idx_high = int(idx*portion), int((idx+1)*portion)
                indices.append(np.random.randint(idx_low, idx_high))
            else:
                indices.append(idx)
        return indices

    def get_label_weights(self):
        return self.label_weights

    def get_main_label_indices(self):
        return np.arange(self.num_labels[1])


class SingleHeadEmbeddedDatasets(object):
    def __init__(self, datasets, p=None):
        self.datasets = datasets
        self.p = p if p is not None else np.full(len(self.datasets), 1. / len(
            self.datasets), dtype=np.float32)  # sample probability for the datasets

        self.lengths = np.array([len(dataset) for dataset in self.datasets])
        self.cumsum_length = np.cumsum(self.lengths)
        self.total_length = np.sum(self.lengths)
        self.main_length = self.lengths[0]

        self._compute_label_weights()

        self._index = -1
        self._data_index = 0

    def _compute_label_weights(self):
        ''' Compute the weights of the labels
        '''
        self.num_labels = np.cumsum(
            [0] + [len(d.get_label_weights()) for d in self.datasets])
        labels = torch.cat([d.labels + self.num_labels[i]
                           for i, d in enumerate(self.datasets)])

        labels, counts = torch.unique(labels, return_counts=True)

        self.label_weights = torch.zeros_like(counts, dtype=torch.float32)
        self.label_weights[labels] = counts.max() / counts

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        data_idx = 0
        actual_idx = idx

        for i in range(len(self.cumsum_length) - 1):
            if idx >= self.cumsum_length[i] and idx < self.cumsum_length[i+1]:
                data_idx = i + 1
                actual_idx = idx % self.cumsum_length[i]

        return self.datasets[data_idx].__getitem__(actual_idx)

    def _get_indices(self, idx):
        indices = list()
        for i, l in enumerate(self.lengths):
            if l < self.main_length:
                indices.append(int((idx / self.main_length) * l))
            elif l > self.main_length:
                portion = l / self.main_length
                idx_low, idx_high = int(idx*portion), int((idx+1)*portion)
                indices.append(np.random.randint(idx_low, idx_high))
            else:
                indices.append(idx)
        return indices

    def get_label_weights(self):
        return self.label_weights

    def get_main_label_indices(self):
        return np.arange(self.num_labels[1])


def create_data_channels(filename, modelname, fuse_type, max_length):
    data = pd.read_csv(filename, sep='\t')

    print('Number of data instance: {}'.format(data.shape[0]))

    # map labels to ids
    unique_labels = data['label'].unique()
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    # label2id = {'used': 1, 'not used': 0, 'extended': 0}  # binary for now

    data['label'] = data['label'].apply(
        lambda x: label2id[x])

    data_train = data[data['split'] == 'train'].reset_index()
    data_val = data[data['split'] == 'val']
    data_test = data[data['split'] == 'test']

    train_data = Dataset(
        data_train,
        modelname=modelname,
        fuse_type=fuse_type,
        max_length=max_length
    )
    val_data = Dataset(
        data_val,
        modelname=modelname,
        fuse_type=fuse_type,
        max_length=max_length
    )
    test_data = Dataset(
        data_test,
        modelname=modelname,
        fuse_type=fuse_type,
        max_length=max_length
    )

    return train_data, val_data, test_data
