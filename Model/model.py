import os
import torch
import torch.nn as nn

from transformers import AutoModel
from .layers import DenseLayer

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LanguageModel(nn.Module):
    def __init__(self, modelname, device, readout, num_classes=None):
        super(LanguageModel, self).__init__()
        self.device = device
        self.modelname = modelname
        self.readout = readout
        self.num_classes = num_classes

        self.model = AutoModel.from_pretrained(modelname)
        self.hidden_size = self.model.config.hidden_size

        if self.num_classes is not None:
            self.ln = nn.Linear(self.hidden_size, num_classes)

    def forward(self, tokens):
        tokens = tokens.to(self.device)
        outputs = self.model(**tokens)

        if self.readout == 'cls':
            text_representations = outputs.last_hidden_state[:, 0]
        elif self.readout == 'mean':
            text_representations = mean_pooling(outputs, tokens['attention_mask'])
        
        if self.num_classes is not None:
            preds = self.ln(text_representations)
            return preds
        return text_representations

    def save_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        torch.save(self.state_dict(), model_filename)

    def load_pretrained(self, modeldir):
        model_filename = os.path.join(modeldir, 'checkpoint.pt')
        self.load_state_dict(torch.load(model_filename))
