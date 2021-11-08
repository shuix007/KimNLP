import torch
import torch.nn as nn

from transformers import AutoModel
from .layers import DenseLayer


class LanguageModel(nn.Module):
    def __init__(self, modelname, device, rawtext_readout, context_readout, intra_context_pooling):
        super(LanguageModel, self).__init__()
        self.device = device
        self.modelname = modelname
        self.ctk_l = 5 if 'xlnet' in self.modelname else 4
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
        self.hidden_size = self.model.config.hidden_size

    def _context_readout(self, lm_output, readout_mask=None):
        bert_dims = lm_output.last_hidden_state.size(-1)

        if self.context_readout == 'cls':
            return lm_output.last_hidden_state[0, 0]
        elif self.context_readout == 'mean':
            return lm_output.last_hidden_state[0].mean(0)
        elif self.context_readout == 'ch':
            if readout_mask is not None:
                result = lm_output.last_hidden_state[readout_mask].view(
                    -1, self.ctk_l, bert_dims).mean(dim=1)
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
            output_dict[key] = value.to(self.device) if len(
                value.size()) == 2 else value.to(self.device).squeeze(0)
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


# class MLPClassifier(nn.Module):
#     def __init__(self, input_dims, hidden_list, n_classes, activation, dropout, device):
#         super(MLPClassifier, self).__init__()
#         self.device = device
#         self.n_classes = n_classes
#         self.n_layers = len(hidden_list)
#         self.hidden_layers = nn.ModuleList([])
#         input_ = input_dims
#         for i in range(len(hidden_list)):
#             self.hidden_layers.append(DenseLayer(
#                 input_, hidden_list[i], activation=activation, bias=True))
#             input_ = hidden_list[i]
#         self.top_layer = DenseLayer(hidden_list[-1], n_classes)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, fused_context_embeds):
#         hidden = fused_context_embeds.to(self.device)
#         for layer in self.hidden_layers:
#             hidden = self.dropout(layer(hidden))
#         logits = self.top_layer(hidden)
#         return logits

class MLPClassifier(nn.Module):
    def __init__(self, input_dims, hidden_list, n_classes, activation, dropout, device):
        super(MLPClassifier, self).__init__()
        self.device = device
        self.n_classes = n_classes
        # self.n_layers = len(hidden_list)
        # self.hidden_layers = nn.ModuleList([])
        # input_ = input_dims
        # for i in range(len(hidden_list)):
        #     self.hidden_layers.append(DenseLayer(
        #         input_, hidden_list[i], activation=activation, bias=True))
        #     input_ = hidden_list[i]
        self.top_layer = DenseLayer(input_dims, n_classes)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, fused_context_embeds):
        hidden = fused_context_embeds.to(self.device)
        # for layer in self.hidden_layers:
        #     hidden = self.dropout(layer(hidden))
        logits = self.top_layer(hidden)
        return logits

class EarlyFuseClassifier(nn.Module):
    def __init__(self, lm_model, mlp_model):
        super(EarlyFuseClassifier, self).__init__()
        self.lm_model = lm_model
        self.mlp_model = mlp_model

    def forward(self, fused_text_tokens):
        lm_output = self.lm_model.context_forward(
            fused_text_tokens).unsqueeze(0)
        logits = self.mlp_model(lm_output)
        return logits


class LateFuseClassifier(nn.Module):
    def __init__(self, lm_model, mlp_model, inter_context_pooling):
        super(LateFuseClassifier, self).__init__()
        self.lm_model = lm_model
        self.mlp_model = mlp_model

        self.inter_context_pooling = inter_context_pooling
        assert self.inter_context_pooling in [
            'sum', 'max', 'mean', 'topk'], 'Inter context pooling type {} not supported'.format(self.inter_context_pooling)

    def forward(self, citation_context_tokens):
        citation_context_embeds = list()
        for context in citation_context_tokens:
            context_embed = self.lm_model.context_forward(context)
            citation_context_embeds.append(context_embed)
        citation_context_embeds = torch.stack(citation_context_embeds)

        if self.inter_context_pooling == 'sum':
            citation_context_embeds = citation_context_embeds.sum(
                dim=0)
        elif self.inter_context_pooling == 'max':
            citation_context_embeds = citation_context_embeds.max(
                dim=0).values
        elif self.inter_context_pooling == 'mean':
            citation_context_embeds = citation_context_embeds.mean(
                dim=0)
        elif self.inter_context_pooling == 'topk':
            topk = citation_context_embeds.topk(10, dim=0)
            citation_context_embeds = topk[0].mean(dim=0)

        logits = self.mlp_model(
            citation_context_embeds.unsqueeze(0)
        )
        return logits
