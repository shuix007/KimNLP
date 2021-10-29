import torch
import torch.nn as nn

class MultiHeadEarlyFuseClassifier(nn.Module):
    def __init__(self, lm_model, mlp_models):
        super(MultiHeadEarlyFuseClassifier, self).__init__()
        self.lm_model = lm_model
        self.mlp_models = mlp_models

    def forward(self, head_idx, fused_text_tokens):
        lm_output = self.lm_model.context_forward(
            fused_text_tokens).unsqueeze(0)
        logits = self.mlp_models[head_idx](lm_output)
        return logits


class MultiHeadLateFuseClassifier(nn.Module):
    def __init__(self, lm_model, mlp_models, inter_context_pooling):
        super(MultiHeadLateFuseClassifier, self).__init__()
        self.lm_model = lm_model
        self.mlp_models = mlp_models

        self.inter_context_pooling = inter_context_pooling
        assert self.inter_context_pooling in [
            'sum', 'max', 'mean', 'topk'], 'Inter context pooling type {} not supported'.format(self.inter_context_pooling)

    def forward(self, head_idx, cited_title_tokens, cited_abstract_tokens, citing_title_tokens, citing_abstract_tokens, citation_context_tokens):
        cited_title_embeds = self.lm_model(cited_title_tokens)
        cited_abstract_embeds = self.lm_model(cited_abstract_tokens)
        citing_title_embeds = self.lm_model(citing_title_tokens)
        citing_abstract_embeds = self.lm_model(citing_abstract_tokens)

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

        logits = self.mlp_models[head_idx](
            cited_title_embeds.unsqueeze(0),
            cited_abstract_embeds.unsqueeze(0),
            citing_title_embeds.unsqueeze(0),
            citing_abstract_embeds.unsqueeze(0),
            citation_context_embeds.unsqueeze(0)
        )
        return logits


class MultiHeadContextOnlyLateFuseClassifier(nn.Module):
    def __init__(self, lm_model, mlp_models, inter_context_pooling):
        super(MultiHeadContextOnlyLateFuseClassifier, self).__init__()
        self.lm_model = lm_model
        self.mlp_models = mlp_models

        self.inter_context_pooling = inter_context_pooling
        assert self.inter_context_pooling in [
            'sum', 'max', 'mean', 'topk'], 'Inter context pooling type {} not supported'.format(self.inter_context_pooling)

    def forward(self, head_idx, citation_context_tokens):
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

        logits = self.mlp_models[head_idx](
            citation_context_embeds.unsqueeze(0)
        )
        return logits