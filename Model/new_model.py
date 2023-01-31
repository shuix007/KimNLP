import torch
import torch.nn as nn

from transformers import AutoModel
from .layers import DenseLayer


class LanguageModel(nn.Module):
    def __init__(self, modelname, device, readout):
        super(LanguageModel, self).__init__()
        self.device = device
        self.modelname = modelname

        self.readout = readout

        assert self.readout in [
            'cls', 'mean'], 'Raw text readout type {} is not supported.'.format(self.rawtext_readout)

        self.model = AutoModel.from_pretrained(modelname)
        self.hidden_size = self.model.config.hidden_size

    def _to_device(self, input_dict):
        output_dict = {}
        for key, value in input_dict.items():
            output_dict[key] = value.to(self.device) if len(
                value.size()) == 2 else value.to(self.device).squeeze(0)
        return output_dict

    def forward(self, input_dict):
        if self.readout == 'cls':
            return self.model(**self._to_device(input_dict)).last_hidden_state[0, 0]
        elif self.readout == 'mean':
            return self.model(**self._to_device(input_dict)).last_hidden_state[0].mean(dim=0)

class MLP(nn.Module):
    def __init__(self, input_dims, n_classes, device):
        super(MLP, self).__init__()
        self.device = device
        self.n_classes = n_classes

        self.top_layer = DenseLayer(input_dims, n_classes)

    def forward(self, embeds):
        hidden = embeds.to(self.device)
        logits = self.top_layer(hidden)
        return logits

class Classifier(nn.Module):
    def __init__(self, lm_model, mlp_model, use_abstract):
        super(Classifier, self).__init__()
        self.lm_model = lm_model
        self.mlp_model = mlp_model

        self.use_abstract = use_abstract

    def forward(self, citation_context, citing_abstract, cited_abstract):
        citation_context_embeds = self.lm_model(citation_context)
        if self.use_abstract:
            citing_abstract_embeds = self.lm_model(citing_abstract)
            cited_abstract_embeds = self.lm_model(cited_abstract)
            embeds = torch.cat([
                citation_context_embeds,
                citing_abstract_embeds,
                cited_abstract_embeds
            ], dim=0)
        else:
            embeds = citation_context_embeds
        
        logits = self.mlp_model(embeds)
        return logits
