import torch
import torch.nn as nn
import torch.nn.functional as F

def readout(bert_result, readout_type, readout_mask=None, pooling=None):
    bert_dims = bert_result.last_hidden_state.size(-1)

    if readout_type == "cls":
        return bert_result.last_hidden_state[0, 0]
    elif readout_type == "mean":
        return bert_result.last_hidden_state[0].mean(0)
    elif readout_type == "ch":
        if readout_mask is not None:
            result = (
                bert_result.last_hidden_state[readout_mask]
                .view(-1, 4, bert_dims)
                .mean(1)
            )
            if pooling == "max":
                return result.max(0).values
            elif pooling == "mean":
                return result.mean(0)
            elif pooling == "sum":
                return result.sum(0)
            else:
                raise ValueError("Pooling {} is not supported.".format(pooling))
        else:
            raise ValueError("Cited Here readout requires readout_index.")
    else:
        raise ValueError("Readout type {} is not supported.".format(readout_type))

class BertClassifier(nn.Module):
    def __init__(
        self,
        n_classes,
        bert,
        emb_dim,
        hidden_list,
        dropout=0.5,
        flag="mean",
        readout="mean",
        pooling="max",
    ):
        super().__init__()
        self.flag = flag
        self.n_classes = n_classes
        self.n_layers = len(hidden_list)
        self.hidden_layers = nn.ModuleList([])
        self.batch_norm = nn.ModuleList([])
        input_ = emb_dim * 5
        for i in range(len(hidden_list)):
            self.hidden_layers.append(nn.Linear(input_, hidden_list[i]))
            self.batch_norm.append(nn.BatchNorm1d(hidden_list[i]))
            input_ = hidden_list[i]
        self.top_layer = nn.Linear(hidden_list[-1], n_classes)
        self.dropout = nn.Dropout(dropout)
        self.scibert = bert
        self.readout = readout
        self.pooling = pooling

    def forward(
        self,
        cited_title_token,
        cited_abstract_token,
        citing_title_token,
        citing_abstract_token,
        citation_context_token,
    ):
        #         tuple
        cited_abstract = self.scibert(**cited_abstract_token)
        citing_abstract = self.scibert(**citing_abstract_token)

        citation_context = []
        for i in range(len(citation_context_token)):
            readout_index = citation_context_token[i].pop("readout_mask")
            citation_context.append(
                readout(
                    self.scibert(**citation_context_token[i]),
                    readout_type=self.readout,
                    readout_mask=readout_index,
                    pooling=self.pooling,
                )
            )

        citation_context = torch.stack(citation_context, dim=0)
        if self.flag == "sum":
            citation_context = citation_context.sum(dim=0, keepdims=True)
        elif self.flag == "max":
            citation_context = citation_context.max(dim=0, keepdims=True)
        elif self.flag == "mean":
            citation_context = citation_context.mean(dim=0, keepdims=True)
        elif self.flag == "topk":
            topk = citation_context.topk(10, dim=0, keepdims=True)
            citation_context = topk[0].mean(dim=0, keepdims=True)
        else:
            pass

        input_emb = torch.cat(
            [
                cited_abstract.last_hidden_state[:, 0],
                citing_abstract.last_hidden_state[:, 0],
                cited_abstract.last_hidden_state[:, 0] * citation_context,
                citing_abstract.last_hidden_state[:, 0] * citation_context,
                citation_context,
            ],
            dim=1,
        )
        hidden = self.dropout(torch.tanh(self.hidden_layers[0](input_emb)))

        for i in range(self.n_layers - 1):
            hidden = self.dropout(torch.tanh(self.hidden_layers[i + 1](hidden)))
        scores = self.top_layer(hidden)

        return scores