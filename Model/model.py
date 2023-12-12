import os
import torch
import torch.nn as nn

from transformers import AutoModel
from .layers import DenseLayer

def mask_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LanguageModel(nn.Module):
    def __init__(self, modelname, device, readout, num_classes):
        super(LanguageModel, self).__init__()
        self.device = device
        self.modelname = modelname
        self.readout = readout
        self.num_classes = num_classes

        self.model = AutoModel.from_pretrained(modelname)
        self.hidden_size = self.model.config.hidden_size

        self.ln = nn.Linear(self.hidden_size, num_classes)

    def forward(self, tokens):
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
        return preds

    # def compute_loss(self, tokens, labels):
    #     preds = self.forward(tokens)
    #     return self.loss_fn(preds, labels)

    def save_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        torch.save(self.state_dict(), model_filename)

    def load_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        self.load_state_dict(torch.load(model_filename))

class MultiHeadLanguageModel(nn.Module):
    def __init__(self, modelname, device, readout, num_classes):
        super(LanguageModel, self).__init__()
        self.device = device
        self.modelname = modelname
        self.readout = readout
        self.num_classes = num_classes

        self.model = AutoModel.from_pretrained(modelname)
        self.hidden_size = self.model.config.hidden_size

        self.lns = nn.ModuleList([nn.Linear(self.hidden_size, num_class) for num_class in num_classes])

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
        
        preds = {}
        for i in range(len(self.lns)):
            per_head_text_representations = text_representations[head_indices == i]
            preds[i] = self.lns[i](per_head_text_representations)
        return preds

    def save_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        torch.save(self.state_dict(), model_filename)

    def load_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        self.load_state_dict(torch.load(model_filename))

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