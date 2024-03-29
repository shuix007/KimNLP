import os
import torch
import torch.nn as nn

from typing import List
from transformers import AutoModel, AutoTokenizer
from .layers import DenseLayer

def mask_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LanguageModel(nn.Module):
    def __init__(self, 
                 modelname: str, 
                 device: str, 
                 readout: str
        ):
        super(LanguageModel, self).__init__()
        self.device = device
        self.modelname = modelname
        self.readout_fn = readout

        self.model = AutoModel.from_pretrained(modelname)
        self.hidden_size = self.model.config.hidden_size

    def readout(self, model_inputs, model_outputs, readout_masks=None):
        if self.readout_fn == 'cls':
            if 'bert' in self.modelname:
                text_representations = model_outputs.last_hidden_state[:, 0]
            elif 'xlnet' in self.modelname:
                text_representations = model_outputs.last_hidden_state[:, -1]
            else:
                raise ValueError('Invalid model name {} for the cls readout.'.format(self.modelname))
        elif self.readout_fn == 'mean':
            text_representations = mask_pooling(model_outputs, model_inputs['attention_mask'])
        elif self.readout_fn == 'ch' and readout_masks is not None:
            text_representations = mask_pooling(model_outputs, readout_masks)
        else:
            raise ValueError('Invalid readout function.')
        return text_representations

    def _lm_forward(self, tokens):
        tokens = tokens.to(self.device)
        if 'readout_mask' in tokens:
            readout_mask = tokens.pop('readout_mask')
        else:
            readout_mask = None
        outputs = self.model(**tokens)
        return self.readout(tokens, outputs, readout_mask)

    def forward(self):
        raise NotImplementedError

    def save_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        torch.save(self.state_dict(), model_filename)

    def load_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        self.load_state_dict(torch.load(model_filename))

class MultiHeadLanguageModel(LanguageModel):
    def __init__(self, 
                 modelname: str, 
                 device: str, 
                 readout: str, 
                 num_classes: List
        ):
        super().__init__(
            modelname,
            device, 
            readout
        )

        self.num_classes = num_classes
        self.lns = nn.ModuleList([nn.Linear(self.hidden_size, num_class) for num_class in num_classes])

    def forward(self, input_tokens, input_head_indices, class_tokens, class_head_indices):
        head_indices = torch.unique(input_head_indices)
        text_representations = self._lm_forward(input_tokens)

        final_preds = {}
        for i in head_indices:
            if torch.any(input_head_indices == i):
                final_preds[i.item()] = self.lns[i.item()](text_representations[input_head_indices == i])
            else:
                final_preds[i.item()] = torch.tensor([]).to(self.device)
        return final_preds

class MultiHeadPsuedoLanguageModel(nn.Module):
    def __init__(self, modelname, device, readout, num_classes):
        super(MultiHeadPsuedoLanguageModel, self).__init__()
        self.device = device
        self.modelname = modelname
        self.readout = readout
        self.num_classes = num_classes
        self.target_num_classes = num_classes[0]

        self.model = AutoModel.from_pretrained(modelname)
        self.hidden_size = self.model.config.hidden_size

        self.ln = nn.Linear(self.hidden_size, self.target_num_classes)
        if len(self.num_classes) > 1:
            self.label_space_mappings = nn.ParameterDict()
            for i in range(1, len(self.num_classes)):
                label_space_mapping = nn.Parameter(torch.randn(self.target_num_classes, self.num_classes[i]))
                self.label_space_mappings[str(i)] = label_space_mapping

    def forward(self, tokens, head_indices):
        tokens = tokens.to(self.device)
        readout_mask = tokens.pop('readout_mask')
        outputs = self.model(**tokens)

        if self.readout == 'cls':
            text_representations = outputs.last_hidden_state[:, 0]
        elif self.readout == 'mean':
            text_representations = mask_pooling(outputs, tokens['attention_mask'])
        elif self.readout == 'ch':
            text_representations = mask_pooling(outputs, readout_mask)
        
        preds = self.ln(text_representations)

        final_preds = {0: preds[head_indices == 0]}
        if len(self.num_classes) > 1:
            for i in range(1, len(self.num_classes)):
                per_head_preds = preds[head_indices == i]
                final_preds[i] = torch.matmul(per_head_preds, torch.softmax(self.label_space_mappings[str(i)], dim=1))

        return final_preds

    def print_label_space_mapping(self, label_maps):
        for i in range(1, len(self.num_classes)):
            label_space_projection = torch.softmax(self.label_space_mappings[str(i)], dim=1).detach().cpu().tolist()
            print(', '.join(label_maps[i]))
            for j in range(len(label_space_projection)):
                print(', '.join([str(x) for x in label_space_projection[j]] + [label_maps[0][j]]))

    def save_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        torch.save(self.state_dict(), model_filename)

    def load_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        self.load_state_dict(torch.load(model_filename))

class MultiHeadContrastiveLanguageModel(nn.Module):
    def __init__(self, 
                 modelname: str, 
                 device: str, 
                 readout: str
        ):
        super(MultiHeadContrastiveLanguageModel, self).__init__()
        self.device = device
        self.modelname = modelname
        self.readout_fn = readout

        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.model = AutoModel.from_pretrained(modelname)
        self.hidden_size = self.model.config.hidden_size

    def readout(self, model_inputs, model_outputs, readout_masks=None):
        if self.readout_fn == 'cls':
            text_representations = model_outputs.last_hidden_state[:, 0]
        elif self.readout_fn == 'mean':
            text_representations = mask_pooling(model_outputs, model_inputs['attention_mask'])
        elif self.readout_fn == 'ch' and readout_masks is not None:
            text_representations = mask_pooling(model_outputs, readout_masks)
        else:
            text_representations = model_outputs.last_hidden_state[:, 0]
            # raise ValueError('Invalid readout function.')
        return text_representations

    def _lm_forward(self, tokens):
        tokens = tokens.to(self.device)
        if 'readout_mask' in tokens:
            readout_mask = tokens.pop('readout_mask')
        else:
            readout_mask = None
        outputs = self.model(**tokens)
        return self.readout(tokens, outputs, readout_mask)

    def forward(self, input_tokens, input_head_indices, class_tokens, class_head_indices):
        head_indices = torch.unique(class_head_indices)

        input_text_representations = self._lm_forward(input_tokens)
        class_text_representations = self._lm_forward(class_tokens)

        final_preds = {}
        for i in head_indices:
            if torch.any(input_head_indices == i):
                final_preds[i.item()] = torch.matmul(
                                            input_text_representations[input_head_indices == i], 
                                            class_text_representations[class_head_indices == i].transpose(0, 1)
                                        )
            else:
                final_preds[i.item()] = torch.tensor([]).to(self.device)

        return final_preds

    def save_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        torch.save(self.state_dict(), model_filename)

    def load_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        self.load_state_dict(torch.load(model_filename))