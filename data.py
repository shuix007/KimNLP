import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import BertTokenizer


class Dataset(object):
    def __init__(self, annotated_data, modelname, early_fuse):
        self.early_fuse = early_fuse  # control the way to fetch

        self.tokenizer = BertTokenizer.from_pretrained(modelname)

        self._load_data(annotated_data)
        self._early_fusion()
        self._tokenize()

        self._index = -1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        if self.early_fuse:
            return (
                self.fused_text_tokens[idx],
                self.labels[idx]
            )
        else:
            return (
                self.cited_title_tokens[idx],
                self.cited_abstract_tokens[idx],
                self.citing_title_tokens[idx],
                self.citing_abstract_tokens[idx],
                self.citation_context_tokens[idx],
                self.labels[idx]
            )

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
        self.indices = []
        self.labels = []
        self.cited_title_list = []
        self.citing_title_list = []
        self.cited_abstract_list = []
        self.citing_abstract_list = []
        self.citation_context_list = []

        for i in range(len(annotated_data.index)):
            self.labels.append(annotated_data.loc[i, 'label'])
            self.indices.append(i)
            self.cited_title_list.append(
                str(annotated_data.loc[i, 'cited title']))
            self.citing_title_list.append(
                str(annotated_data.loc[i, 'citing title']))
            self.cited_abstract_list.append(
                str(annotated_data.loc[i, 'cited abstract']))
            self.citing_abstract_list.append(
                str(annotated_data.loc[i, 'citing abstract']))
            # Since there multiple citation contexts, I've separated the with ---(j)--- in the spreadsheet, should ideally use regex to split them, here is a random way
            citing_contexts = str(annotated_data.loc[i, 'citation context']).split(
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

    def _early_fusion(self):
        self.fused_text_list = list()

        for i in range(len(self.cited_title_list)):
            fused_text = ' [SEP] '.join(self.citation_context_list[i])

            fused_text += (' [SEP] ' + self.cited_title_list[i] +
                           ' [SEP] ' + self.cited_abstract_list[i] +
                           ' [SEP] ' + self.citing_title_list[i] +
                           ' [SEP] ' + self.citing_abstract_list[i])

            self.fused_text_list.append(fused_text)

    def _get_readout_index(self, tokens):
        cited_here_tokens = torch.tensor([962, 8412, 1530, 1374])
        readout_index = []

        l = tokens['input_ids'].size(1)
        for i in range(1, l - 4):
            if torch.equal(tokens['input_ids'][0, i:i+4], cited_here_tokens):
                readout_index.append(torch.arange(i, i+4))
        return torch.cat(readout_index)

    def _get_readout_mask(self, tokens, readout_index):
        mask = torch.zeros_like(tokens['input_ids'], dtype=torch.bool)
        mask[0, readout_index] = True
        return mask

    def _tokenize_context(self, context):
        tokens = self.tokenizer(
            context,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        tokens['readout_mask'] = self._get_readout_mask(
            tokens,
            self._get_readout_index(tokens)
        )

        return tokens

    def _tokenize_text(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

    def _tokenize(self):
        self.cited_title_tokens = list()
        self.citing_title_tokens = list()
        self.cited_abstract_tokens = list()
        self.citing_abstract_tokens = list()
        self.citation_context_tokens = list()

        self.fused_text_tokens = list()

        for i in tqdm(range(len(self.cited_title_list))):
            self.cited_title_tokens.append(
                self._tokenize_text(
                    self.cited_title_list[i]
                )
            )

            self.citing_title_tokens.append(
                self._tokenize_text(
                    self.citing_title_list[i]
                )
            )

            self.cited_abstract_tokens.append(
                self._tokenize_text(
                    self.cited_abstract_list[i]
                )
            )

            self.citing_abstract_tokens.append(
                self._tokenize_text(
                    self.citing_abstract_list[i]
                )
            )

            self.citation_context_tokens.append([
                self._tokenize_context(
                    ctx
                ) for ctx in self.citation_context_list[i]
            ])

            self.fused_text_tokens.append(
                self._tokenize_context(
                    self.fused_text_list[i]
                )
            )


class EmbeddedDataset(object):
    def __init__(self, dataset, lm_model, early_fuse, inter_context_pooling):
        self.early_fuse = early_fuse
        self.inter_context_pooling = inter_context_pooling

        assert self.inter_context_pooling in [
            'sum', 'max', 'mean', 'topk'], 'Inter context pooling type {} not supported'.format(self.inter_context_pooling)

        self.labels = dataset.labels

        if self.early_fuse:
            self._compute_fused_embeddings(dataset, lm_model)
        else:
            self._compute_raw_embeddings(dataset, lm_model)

    def get_label_weights(self):
        labels, counts = torch.unique(self.labels, return_counts=True)

        label_weights = torch.zeros_like(counts, dtype=torch.float32)
        label_weights[labels] = counts.max() / counts

        return label_weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.early_fuse:
            return (
                self.fused_context_embeds[idx],
                self.labels[idx]
            )
        else:
            return (
                self.cited_title_embeds[idx],
                self.cited_abstract_embeds[idx],
                self.citing_title_embeds[idx],
                self.citing_abstract_embeds[idx],
                self.citation_context_embeds[idx],
                self.labels[idx]
            )

    def _compute_fused_embeddings(self, dataset, model):
        self.fused_context_embeds = list()

        model.eval()
        with torch.no_grad():
            print('Computing early-fused embeddings.')
            for fused_context, _ in tqdm(dataset):
                lm_output = model.context_forward(fused_context).cpu()
                self.fused_context_embeds.append(lm_output)
            self.fused_context_embeds = torch.stack(self.fused_context_embeds)

    def _compute_raw_embeddings(self, dataset, model):
        self.cited_title_embeds = list()
        self.cited_abstract_embeds = list()
        self.citing_title_embeds = list()
        self.citing_abstract_embeds = list()
        self.citation_context_embeds = list()

        model.eval()
        with torch.no_grad():
            print('Computing late-fused embeddings.')
            for cited_title, cited_abstract, citing_title, citing_abstract, citation_context, _ in tqdm(dataset):
                cited_title_embed = model(cited_title).cpu()
                cited_abstract_embed = model(cited_abstract).cpu()
                citing_title_embed = model(citing_title).cpu()
                citing_abstract_embed = model(citing_abstract).cpu()

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

                self.cited_title_embeds.append(cited_title_embed)
                self.cited_abstract_embeds.append(cited_abstract_embed)
                self.citing_title_embeds.append(citing_title_embed)
                self.citing_abstract_embeds.append(citing_abstract_embed)
                self.citation_context_embeds.append(citation_context_embeds)

            self.cited_title_embeds = torch.stack(
                self.cited_title_embeds)
            self.cited_abstract_embeds = torch.stack(
                self.cited_abstract_embeds)
            self.citing_title_embeds = torch.stack(
                self.citing_title_embeds)
            self.citing_abstract_embeds = torch.stack(
                self.citing_abstract_embeds)
            self.citation_context_embeds = torch.stack(
                self.citation_context_embeds)


def create_data_channels(filename, modelname, split_ratios, earyly_fuse):
    assert np.abs(np.sum(split_ratios) -
                  1.) < 1e-8, 'The split ratios must sum to 1 instead of {}'.format(np.sum(split_ratios))

    annotated_data = pd.read_csv(filename, sep='\t')
    annotated_data = annotated_data.replace({np.nan: None})

    valid_index = np.zeros(annotated_data.shape[0], dtype=bool)
    annotated_labels = list()

    # filter out instances that are mot labelled or have invalid labels
    for i in range(len(annotated_data.index)):
        if annotated_data['new annotation'][i] is None:
            if annotated_data['previous annotation'][i] is None:
                print("Instance {} not labeled.".format(i))
                continue
            label = str(
                annotated_data['previous annotation'][i]).strip().lower()
        else:
            label = str(annotated_data['new annotation'][i]).strip().lower()

        if label in ['used', 'not used', 'extended']:
            valid_index[i] = True
            annotated_labels.append(label)
        else:
            print('The label {} of instance {} is not valid.'.format(label, i))

    print('Number of valid data instance: {}'.format(valid_index.sum()))
    annotated_data = annotated_data[valid_index]
    annotated_data['label'] = annotated_labels

    # map labels to ids
    unique_labels = annotated_data['label'].unique()
    # label2id = {lb: i for i, lb in enumerate(unique_labels)}
    label2id = {'used': 1, 'not used': 0, 'extended': 0}  # binary for now

    annotated_data['label'] = annotated_data['label'].apply(
        lambda x: label2id[x])

    # split data such that the train/val/test ratio of each label class is the same
    data_indices = np.arange(annotated_data.shape[0])
    train_ratio, val_ratio, test_ratio = split_ratios

    train_indices = list()
    val_indices = list()
    test_indices = list()

    unique_label_ids = [label2id[lb] for lb in unique_labels]
    for lb in unique_label_ids:
        lb_indices = data_indices[annotated_data['label'] == lb]
        np.random.shuffle(lb_indices)

        num_trains = int(len(lb_indices) * train_ratio)
        num_vals = int(len(lb_indices) * val_ratio)
        num_tests = len(lb_indices) - num_trains - num_vals

        lb_train, lb_val, lb_test = np.array_split(
            lb_indices, np.cumsum([num_trains, num_vals]))

        train_indices.append(lb_train)
        val_indices.append(lb_val)
        test_indices.append(lb_test)

    train_indices = np.concatenate(train_indices)
    val_indices = np.concatenate(val_indices)
    test_indices = np.concatenate(test_indices)

    annotated_data_train = annotated_data.iloc[train_indices].reset_index()
    annotated_data_val = annotated_data.iloc[val_indices].reset_index()
    annotated_data_test = annotated_data.iloc[test_indices].reset_index()

    train_data = Dataset(annotated_data_train,
                         modelname=modelname, early_fuse=earyly_fuse)
    val_data = Dataset(annotated_data_val,
                       modelname=modelname, early_fuse=earyly_fuse)
    test_data = Dataset(annotated_data_test,
                        modelname=modelname, early_fuse=earyly_fuse)

    return train_data, val_data, test_data
