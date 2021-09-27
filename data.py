import torch
import numpy as np
import pandas as pd

from transformers import BertTokenizer


class Dataset(object):
    def __init__(self, filename, modelname, early_fuse):
        self.early_fuse = early_fuse  # control the way to fetch

        annotated_data = pd.read_csv(filename, sep='\t')
        annotated_data = annotated_data.replace({np.nan: None})

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

    def _load_data(self, annotated_data):
        self.indices = []
        self.labels = []
        self.cited_title_list = []
        self.citing_title_list = []
        self.cited_abstract_list = []
        self.citing_abstract_list = []
        self.citation_context_list = []

        for i in range(len(annotated_data.index)):
            if annotated_data['new annotation'][i] is None:
                if annotated_data['previous annotation'][i] is None:
                    print("Instance {} not labeled.".format(i))
                    continue
                label = str(
                    annotated_data['previous annotation'][i]).strip().lower()
            else:
                label = str(
                    annotated_data['new annotation'][i]).strip().lower()

            if label == 'used' or label == 'not used' or label == 'extended':
                self.labels.append(label)
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

            else:
                print('The label {} of instance {} is not valid.'.format(label, i))

    def _early_fusion(self):
        self.fused_text_list = list()

        for i in range(len(self.cited_title_list)):
            fused_text = self.cited_title_list[i] + ' [SEP] ' \
                + self.cited_abstract_list[i] + ' [SEP] ' \
                + self.citing_title_list[i] + ' [SEP] ' \
                + self.citing_abstract_list[i]

            for context in self.citation_context_list[i]:
                fused_text = fused_text + ' [SEP] ' + context

            self.fused_text_list.append(fused_text)

    def _get_readout_index(tokens):
        cited_here_tokens = torch.tensor([962, 8412, 1530, 1374])
        readout_index = []

        l = tokens['input_ids'].size(1)
        for i in range(1, l - 4):
            if torch.equal(tokens['input_ids'][0, i:i+4], cited_here_tokens):
                readout_index.append(torch.arange(i, i+4))
        return torch.cat(readout_index)

    def _get_readout_mask(tokens, readout_index):
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

        for i in range(len(self.cited_title_list)):
            self.cited_title_tokens.append(
                self._tokenize_text(
                    self.cited_title_tokens[i]
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

            self.citation_context_tokens.append(
                self._tokenize_context(
                    self.citation_context_list[i]
                )
            )

            self.fused_text_tokens.append(
                self._tokenize_context(
                    self.fused_text_list[i]
                )
            )
