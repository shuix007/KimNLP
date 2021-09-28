import torch
import torch.nn as nn

from transformers import AutoModel
from layers import DenseLayer


class LanguageModel(nn.Module):
    def __init__(self, modelname, device, rawtext_readout, context_readout, intra_context_pooling):
        self.device = device
        self.modelname = modelname
        self.rawtext_readout = rawtext_readout
        self.context_readout = context_readout
        self.intra_context_pooling = intra_context_pooling

        assert self.rawtext_readout in [
            'cls', 'mean'], 'Raw text readout type {} is not supported.'.format(self.rawtext_readout)
        assert self.context_readout in [
            'cls', 'mean', 'ch'], 'Context Readout type {} is not supported.'.format(self.context_readout)
        assert self.intra_context_pooling in [
            'max', 'mean', 'sum'], 'Pooling type {} is not supported.'.format(self.intra_context_pooling)

        self.model = AutoModel.from_pretrained(modelname)

    def _context_readout(self, lm_output, readout_mask=None):
        bert_dims = lm_output.last_hidden_state.size(-1)

        if self.context_readout == 'cls':
            return lm_output.last_hidden_state[0, 0]
        elif self.context_readout == 'mean':
            return lm_output.last_hidden_state[0].mean(0)
        elif self.context_readout == 'ch':
            if readout_mask is not None:
                result = lm_output.last_hidden_state[readout_mask].view(
                    -1, 4, bert_dims).mean(dim=1)
                if self.intra_context_pooling == 'max':
                    return result.max(dim=0).values
                elif self.intra_context_pooling == 'mean':
                    return result.mean(dim=0)
                elif self.intra_context_pooling == 'sum':
                    return result.sum(dim=0)
            else:
                raise ValueError('Cited Here readout requires readout_index.')

    def _to_device(self, input_dict):
        output_dict = {}
        for key, value in input_dict.items():
            output_dict[key] = value.to(self.device)
        return output_dict

    def forward(self, input_dict):
        if self.rawtext_readout == 'cls':
            return self.model(**self._to_device(input_dict)).last_hidden_state[0, 0]
        elif self.rawtext_readout == 'mean':
            return self.model(**self._to_device(input_dict)).last_hidden_state[0].mean(dim=0)

    def context_forward(self, input_dict):
        input_dict_device = self._to_device(input_dict)
        readout_mask = input_dict_device.pop('readout_mask')

        lm_output = self.model(**input_dict_device)
        return self._context_readout(lm_output=lm_output, readout_mask=readout_mask)

    def save_pretrained(self, modeldir):
        self.model.save_pretrained(modeldir)


class EarlyFuseMLPClassifier(nn.Module):
    def __init__(self, input_dims, hidden_list, n_classes, activation, dropout, device):
        super(EarlyFuseMLPClassifier, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.n_layers = len(hidden_list)
        self.hidden_layers = nn.ModuleList([])
        input_ = input_dims
        for i in range(len(hidden_list)):
            self.hidden_layers.append(DenseLayer(
                input_, hidden_list[i], activation=activation, bias=True))
            input_ = hidden_list[i]
        self.top_layer = DenseLayer(hidden_list[-1], n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, fused_context_embeds):
        hidden = fused_context_embeds.to(self.device)
        for layer in self.hidden_layers:
            hidden = self.dropout(layer(hidden))
        logits = self.top_layer(hidden)
        return logits


class LateFuseMLPCLassifier(nn.Module):
    def __init__(self, input_dims, hidden_list, n_classes, activation, dropout, device):
        super(LateFuseMLPCLassifier, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.n_layers = len(hidden_list)
        self.hidden_layers = nn.ModuleList()
        input_ = input_dims * 5
        for i in range(len(hidden_list)):
            self.hidden_layers.append(DenseLayer(
                input_, hidden_list[i], activation=activation, bias=True))
            input_ = hidden_list[i]
        self.top_layer = DenseLayer(hidden_list[-1], n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cited_title_embeds, cited_abstract_embeds, citing_title_embeds, citing_abstract_embeds, citation_context_embeds):
        hidden = torch.cat([
            cited_title_embeds.to(self.device),
            cited_abstract_embeds.to(self.device),
            citing_title_embeds.to(self.device),
            citing_abstract_embeds.to(self.device),
            citation_context_embeds.to(self.device)
        ], dim=-1)

        for layer in self.hidden_layers:
            hidden = self.dropout(layer(hidden))
        logits = self.top_layer(hidden)
        return logits


class EarlyFuseClassifier(nn.Module):
    def __init__(self, lm_model, mlp_model):
        super(EarlyFuseClassifier, self).__init__()
        self.lm_model = lm_model
        self.mlp_model = mlp_model

    def forward(self, fused_text_tokens):
        lm_output = self.lm_model.context_forward(fused_text_tokens)
        logits = self.mlp_model(lm_output)
        return logits


class LateFuseClassifier(nn.Module):
    def __init__(self, lm_model, mlp_model):
        super(LateFuseClassifier, self).__init__()
        self.lm_model = lm_model
        self.mlp_model = mlp_model

    def forward(self, cited_title_tokens, cited_abstract_tokens, citing_title_tokens, citing_abstract_tokens, citation_context_tokens):
        cited_title_embeds = self.lm_model(cited_title_tokens)
        cited_abstract_embeds = self.lm_model(cited_abstract_tokens)
        citing_title_embeds = self.lm_model(citing_title_tokens)
        citing_abstract_embeds = self.lm_model(citing_abstract_tokens)
        citation_context_embeds = self.lm_model.context_forward(
            citation_context_tokens)

        hidden = torch.cat([
            cited_title_embeds,
            cited_abstract_embeds,
            citing_title_embeds,
            citing_abstract_embeds,
            citation_context_embeds
        ], dim=-1)
        for layer in self.hidden_layers:
            hidden = self.dropout(layer(hidden))
        logits = self.top_layer(hidden)
        return logits