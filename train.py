import os
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score


class Trainer(object):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.early_fuse = args.early_fuse
        self.model = model

        self.loss_fn = nn.CrossEntropyLoss(
            weight=train_dataset.get_label_weights().to(self.device))

        self.optimizer = Adam(model.parameters(), lr=args.lr)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=1, shuffle=True)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False)

        self.logger = SummaryWriter(os.path.join(args.workspace, 'log'))

    def compute_loss(self, labels, logits):
        return self.loss_fn(logits, labels)

    def compute_metrics(self, labels, logits):
        return roc_auc_score(labels, logits[:, 1])  # temp

    def train_one_epoch(self):
        self.model.train()

        num_batches = (len(self.train_dataloader) // self.batch_size) + 1
        for _ in range(num_batches):
            count = 0
            preds, labels = list(), list()

            if self.early_fuse:
                for fused_context, label in self.train_dataloader:
                    preds.append(self.model(fused_context))
                    labels.append(label)
                    count += 1

                    if count == self.batch_size:
                        count = 0
                        break
            else:
                for cited_title, cited_abstract, citing_title, citing_abstract, citation_context, label in self.train_dataloader:
                    preds.append(self.model(cited_title, cited_abstract, citing_title, citing_abstract, citation_context))
                    labels.append(label)
                    count += 1

                    if count == self.batch_size:
                        count = 0
                        break

            preds = torch.cat(preds, dim=0)
            labels = torch.LongTensor(labels).to(self.device)
            
            loss = self.compute_loss(labels, preds)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
    
    def eval_one_epoch(self, test):
        self.model.eval()

        if test:
            data_loader = self.test_dataloader # for inference
        else:
            data_loader = self.val_dataloader # for validation

        preds, labels = list(), list()
        with torch.no_grad():
            if self.early_fuse:
                for fused_context, label in data_loader:
                    preds.append(self.model(fused_context).detach().cpu().numpy())
                    labels.append(label)
            else:
                for cited_title, cited_abstract, citing_title, citing_abstract, citation_context, label in data_loader:
                    preds.append(self.model(cited_title, cited_abstract, citing_title, citing_abstract, citation_context).detach().cpu().numpy())
                    labels.append(label)
            
            preds = np.concatenate(preds, axis=0)
            labels = np.array(label, dtype=np.int64)
            
            roc = self.compute_metrics(labels, preds)
            return roc

    def train(self):
        for epoch in range(1, self.num_epochs):
            loss = self.train_one_epoch()
            roc = self.eval_one_epoch(test=False)
            
            
    def test(self):
        roc = self.eval_one_epoch(test=True)

    def log_tensorboard(writer, NDCG, HR, loss, topk_list, step):
        ''' Write experiment log to tensorboad. '''

        writer.add_scalar('Training loss', loss, step)
        for k in topk_list:
            writer.add_scalar('HR@{}'.format(k), HR[k], step)
            writer.add_scalar('NDCG@{}'.format(k), NDCG[k], step)


    def log_print(NDCG, HR, topk_list):
        ''' Stdout of experiment log. '''

        print('='*33)
        for k in topk_list:
            print(
                'HR@{}: {:.4f}, NDCG@{}: {:.4f}'.format(k, HR[k], k, NDCG[k]))


class PreTrainer(object):
    def __init__(self, mlp_model, train_dataloader, val_dataloader, test_dataloader):
        pass

    def train_one_epoch(self):
        pass

    def train(self):
        pass
