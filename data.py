import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer


class Dataset(object):
    def __init__(self, annotated_data, modelname, fuse_type, context_only, max_length=512, da_alpha=0.15, da_n=8):
        self.fuse_type = fuse_type
        self.early_fuse = fuse_type in [
            'disttrunc', 'bruteforce']  # control the way to fetch
        self.context_only = context_only
        self.max_length = max_length if max_length > 0 else None
        self.truncation = (max_length > 0)
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)

        self.da_n = da_n
        self.da_alpha = da_alpha

        if 'xlnet' in modelname:
            self.cited_here_tokens = self.tokenizer('<CITED HERE>', return_tensors='pt')[
                'input_ids'].squeeze()[:5]
        else:
            # self.cited_here_tokens = torch.tensor([962, 8412, 1530, 1374])
            self.cited_here_tokens = self.tokenizer('<CITED HERE>', return_tensors='pt')[
                'input_ids'].squeeze()[1:-1]

        self._load_data(annotated_data)
        if self.context_only:
            if self.early_fuse:
                self._early_fusion_context_only()
            else:
                self._late_fusion_context_only()
        else:
            if self.early_fuse:
                self._early_fusion()
            else:
                self._tokenize()

        self._index = -1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        if self.context_only:
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
        else:
            if self.early_fuse:
                return [
                    self.fused_text_tokens[idx],
                    self.labels[idx]
                ]
            else:
                return [
                    self.cited_title_tokens[idx],
                    self.cited_abstract_tokens[idx],
                    self.citing_title_tokens[idx],
                    self.citing_abstract_tokens[idx],
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

    def _merge_tokens(self, main_tokens, added_tokens1, added_tokens2):
        main_tokens['input_ids'] = torch.cat([
            main_tokens['input_ids'],
            added_tokens1['input_ids'][:, 1:],
            added_tokens2['input_ids'][:, 1:]
        ], dim=1)

        main_tokens['attention_mask'] = torch.ones_like(
            main_tokens['input_ids'])
        main_tokens['token_type_ids'] = torch.zeros_like(
            main_tokens['input_ids'])
        return main_tokens

    def _early_fusion(self):
        if self.fuse_type == 'disttrunc':
            self._early_fusion_distributed_truncation()
        else:
            self._early_fusion_bruteforce()

    def _early_fusion_distributed_truncation(self):
        self.fused_text_tokens = list()

        print('Tokenizing early fused tokens (distributed_truncation).')
        for i in tqdm(range(len(self.cited_title_list))):
            fused_contexts = ' [SEP] '.join(self.citation_context_list[i])

            fused_text_tokens = self.tokenizer(
                fused_contexts,
                return_tensors='pt',
                max_length=512,
                truncation=True
            )

            if 512 - fused_text_tokens['input_ids'].size(1) >= 4:
                max_title_len = (
                    512 - fused_text_tokens['input_ids'].size(1)) // 2

                cited_title_tokens = self.tokenizer(
                    self.cited_title_list[i],
                    return_tensors='pt',
                    max_length=max_title_len,
                    truncation=True
                )

                citing_title_tokens = self.tokenizer(
                    self.citing_title_list[i],
                    return_tensors='pt',
                    max_length=max_title_len,
                    truncation=True
                )

                fused_text_tokens = self._merge_tokens(
                    fused_text_tokens, cited_title_tokens, citing_title_tokens)

            if 512 - fused_text_tokens['input_ids'].size(1) >= 4:
                max_abstract_len = (
                    512 - fused_text_tokens['input_ids'].size(1)) // 2

                cited_abstract_tokens = self.tokenizer(
                    self.cited_abstract_list[i],
                    return_tensors='pt',
                    max_length=max_abstract_len,
                    truncation=True
                )

                citing_abstract_tokens = self.tokenizer(
                    self.citing_abstract_list[i],
                    return_tensors='pt',
                    max_length=max_abstract_len,
                    truncation=True
                )

                fused_text_tokens = self._merge_tokens(
                    fused_text_tokens, cited_abstract_tokens, citing_abstract_tokens)

            if fused_text_tokens['input_ids'].size(1) > 512:
                print('Wrong')
                print(fused_text_tokens['input_ids'].size(
                    1), max_title_len, max_abstract_len)
                raise ValueError('Wrong')

            fused_text_tokens['readout_mask'] = self._get_readout_mask(
                fused_text_tokens,
                self._get_readout_index(fused_text_tokens)
            )

            self.fused_text_tokens.append(fused_text_tokens)

    def _early_fusion_bruteforce(self):
        self.fused_text_tokens = list()

        print('Tokenizing early fused tokens (brute force).')
        for i in tqdm(range(len(self.cited_title_list))):
            fused_text = ' [SEP] '.join(self.citation_context_list[i])

            fused_text += (' [SEP] ' + self.cited_title_list[i] +
                           ' [SEP] ' + self.cited_abstract_list[i] +
                           ' [SEP] ' + self.citing_title_list[i] +
                           ' [SEP] ' + self.citing_abstract_list[i])

            fused_tokens = self._tokenize_context(fused_text)

            self.fused_text_tokens.append(fused_tokens)

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

    def _tokenize_text(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation
        )

    def _count_long(self):
        truncated_length = list()
        for i in tqdm(range(len(self.fused_text_list))):
            tk_results = self.tokenizer(
                self.fused_text_list[i], return_tensors='pt')

            if tk_results['input_ids'].size(1) > 512:
                truncated_length.append(tk_results['input_ids'].size(1))
        print('{}/{} sentence truncated, average length of truncated context: {:.4f}/{:.4f}'.format(len(
            truncated_length), len(self.cited_title_list), np.mean(truncated_length), np.std(truncated_length)))

    def _tokenize(self):
        self.cited_title_tokens = list()
        self.citing_title_tokens = list()
        self.cited_abstract_tokens = list()
        self.citing_abstract_tokens = list()
        self.citation_context_tokens = list()

        print('Tokenizing late fused tokens.')
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

    def _mask_tokens(self, tokens):
        probability_matrix = torch.full(
            tokens['input_ids'].shape, 1 - self.da_alpha)
        mask = torch.bernoulli(probability_matrix).bool()

        # set special tokens to be True
        mask[0, 0] = True
        mask[0, -1] = True
        if 'readout_mask' in tokens:
            mask[tokens['readout_mask']] = True

        tokens2 = dict()
        tokens2['input_ids'] = tokens['input_ids'][mask].unsqueeze(0)
        tokens2['token_type_ids'] = tokens['token_type_ids'][mask].unsqueeze(0)
        tokens2['attention_mask'] = tokens['attention_mask'][mask].unsqueeze(0)
        if 'readout_mask' in tokens:
            tokens2['readout_mask'] = tokens['readout_mask'][mask].unsqueeze(0)

        return tokens2

    def data_augmentation(self):
        print('Augmenting. Alpha')
        if self.early_fuse:
            augmented_text_tokens = list()
            augmented_labels = list()
            for i in tqdm(range(len(self.fused_text_tokens))):
                for _ in range(self.da_n):
                    da_tokens = self._mask_tokens(self.fused_text_tokens[i])
                    augmented_text_tokens.append(da_tokens)
                    augmented_labels.append(self.labels[i].item())
            self.fused_text_tokens = self.fused_text_tokens + augmented_text_tokens
            self.labels = torch.cat(
                [self.labels, torch.LongTensor(augmented_labels)])
        else:
            augmented_cited_title_tokens = list()
            augmented_cited_abstract_tokens = list()
            augmented_citing_title_tokens = list()
            augmented_citing_abstract_tokens = list()
            augmented_citation_context_tokens = list()
            augmented_labels = list()

            for i in tqdm(range(len(self.cited_title_tokens))):
                for _ in range(self.da_n):
                    da_cited_title_tokens = self._mask_tokens(
                        self.cited_title_tokens[i])
                    da_cited_abstract_tokens = self._mask_tokens(
                        self.cited_abstract_tokens[i])
                    da_citing_title_tokens = self._mask_tokens(
                        self.citing_title_tokens[i])
                    da_citing_abstract_tokens = self._mask_tokens(
                        self.citing_abstract_tokens[i])

                    da_citation_context_tokens = [self._mask_tokens(
                        ctx) for ctx in self.citation_context_tokens[i]]

                    augmented_cited_title_tokens.append(da_cited_title_tokens)
                    augmented_cited_abstract_tokens.append(
                        da_cited_abstract_tokens)
                    augmented_citing_title_tokens.append(
                        da_citing_title_tokens)
                    augmented_citing_abstract_tokens.append(
                        da_citing_abstract_tokens)
                    augmented_citation_context_tokens.append(
                        da_citation_context_tokens)
                    augmented_labels.append(self.labels[i].item())

            self.cited_title_tokens = self.cited_title_tokens + augmented_cited_title_tokens
            self.cited_abstract_tokens = self.cited_abstract_tokens + \
                augmented_cited_abstract_tokens
            self.citing_title_tokens = self.citing_title_tokens + augmented_citing_title_tokens
            self.citing_abstract_tokens = self.citing_abstract_tokens + \
                augmented_citing_abstract_tokens
            self.citation_context_tokens = self.citation_context_tokens + \
                augmented_citation_context_tokens
            self.labels = torch.cat(
                [self.labels, torch.LongTensor(augmented_labels)])


class EmbeddedDataset(object):
    def __init__(self, dataset, lm_model, fuse_type, context_only, inter_context_pooling):
        self.fuse_type = fuse_type
        self.early_fuse = fuse_type in ['disttrunc', 'bruteforce']
        self.context_only = context_only
        self.inter_context_pooling = inter_context_pooling

        assert self.inter_context_pooling in [
            'sum', 'max', 'mean', 'topk'], 'Inter context pooling type {} not supported'.format(self.inter_context_pooling)

        self.labels = dataset.labels

        if self.context_only:
            if self.early_fuse:
                self._compute_fused_embeddings(dataset, lm_model)
            else:
                self._compute_late_fused_embeddings(dataset, lm_model)
        else:
            if self.early_fuse:
                self._compute_fused_embeddings(dataset, lm_model)
            else:
                self._compute_raw_embeddings(dataset, lm_model)

    def get_label_weights(self):
        labels, counts = torch.unique(self.labels, return_counts=True)

        label_weights = torch.zeros_like(counts, dtype=torch.float32)
        label_weights = counts.max() / counts

        return label_weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.context_only:
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
        else:
            if self.early_fuse:
                return [
                    self.fused_context_embeds[idx],
                    self.labels[idx]
                ]
            else:
                return [
                    self.cited_title_embeds[idx],
                    self.cited_abstract_embeds[idx],
                    self.citing_title_embeds[idx],
                    self.citing_abstract_embeds[idx],
                    self.citation_context_embeds[idx],
                    self.labels[idx]
                ]

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

        # indices = self._get_indices(idx)

        # instances = [d.__getitem__(indices[i])
        #              for i, d in enumerate(self.datasets)]
        # for i in range(len(instances)):
        #     instances[i][-1] = instances[i][-1] + \
        #         self.num_labels[i]  # map the labels

        # return instances

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


def create_data_channels(filename, modelname, split_ratios, fuse_type, context_only, max_length, da_alpha, da_n):
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

        if label in ['used', 'not used', 'extended', 'background', 'result', 'method', 'background', 'uses', 'compareorcontrast', 'motivation', 'future', 'extends']:
            valid_index[i] = True
            annotated_labels.append(label)
        else:
            print('The label {} of instance {} is not valid.'.format(label, i))

    print('Number of valid data instance: {}'.format(valid_index.sum()))
    annotated_data = annotated_data[valid_index]
    annotated_data['label'] = annotated_labels

    # map labels to ids
    unique_labels = annotated_data['label'].unique()
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    # label2id = {'used': 1, 'not used': 0, 'extended': 0}  # binary for now

    annotated_data['label'] = annotated_data['label'].apply(
        lambda x: label2id[x])

    # split data such that the train/val/test ratio of each label class is the same
    data_indices = np.arange(annotated_data.shape[0])
    train_ratio, val_ratio, test_ratio = split_ratios

    train_indices = list()
    val_indices = list()
    test_indices = list()

    unique_label_ids = set([label2id[lb] for lb in unique_labels])
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
                         modelname=modelname, fuse_type=fuse_type, context_only=context_only, max_length=max_length, da_alpha=da_alpha, da_n=da_n)
    val_data = Dataset(annotated_data_val,
                       modelname=modelname, fuse_type=fuse_type, context_only=context_only, max_length=max_length, da_alpha=da_alpha, da_n=da_n)
    test_data = Dataset(annotated_data_test,
                        modelname=modelname, fuse_type=fuse_type, context_only=context_only, max_length=max_length, da_alpha=da_alpha, da_n=da_n)

    return train_data, val_data, test_data


def create_single_data_channels(filename, modelname, fuse_type, context_only, max_length, da_alpha, da_n):
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

        if label in ['used', 'not used', 'extended', 'background', 'result', 'method', 'background', 'uses', 'compareorcontrast', 'motivation', 'future', 'extends']:
            valid_index[i] = True
            annotated_labels.append(label)
        else:
            print('The label {} of instance {} is not valid.'.format(label, i))

    print('Number of valid data instance: {}'.format(valid_index.sum()))
    annotated_data = annotated_data[valid_index]
    annotated_data['label'] = annotated_labels

    # map labels to ids
    unique_labels = annotated_data['label'].unique()
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    # label2id = {'used': 1, 'not used': 0, 'extended': 0}  # binary for now

    annotated_data['label'] = annotated_data['label'].apply(
        lambda x: label2id[x])

    data = Dataset(
        annotated_data,
        modelname=modelname,
        fuse_type=fuse_type,
        context_only=context_only,
        max_length=max_length,
        da_alpha=da_alpha,
        da_n=da_n
    )

    return data
