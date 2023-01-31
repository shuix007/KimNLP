import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score, f1_score
from scheduler import get_slanted_triangular_scheduler, get_linear_scheduler

def collate_fn(samples):
    samples = list(map(list, zip(*samples)))
    return samples

def dict2gpu(tokens, device):
    dictionary = {}
    for key, value in tokens.items():
        dictionary[key] = value.to(device)
    return dictionary

def listofdict2gpu(input, device):
    list_of_dictionary = [{} for _ in input]
    for i in range(len(input)):
        for key, value in input[i].items():
            list_of_dictionary[i][key] = value.to(device)
    return list_of_dictionary

class Trainer(object):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.tol = args.tol

        self.workspace = args.workspace
        self.model = model

        self.loss_fn = nn.CrossEntropyLoss(
            weight=train_dataset.get_label_weights().to(self.device))

        self.optimizer = AdamW(self.model.parameters(),
                                lr=args.lr, weight_decay=args.l2)

        if args.scheduler == 'exp':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.decay_rate, verbose=True)
        elif args.scheduler == 'slanted':
            self.scheduler = get_slanted_triangular_scheduler(
                self.optimizer, num_epochs=self.num_epochs)
        elif args.scheduler == 'linear':
            self.scheduler = get_linear_scheduler(
                self.optimizer, num_training_epochs=self.num_epochs, initial_lr=args.lr, final_lr=args.lr/32)
        elif args.scheduler == 'const':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=self.num_epochs, gamma=1., verbose=True)
        else:
            raise ValueError(
                'Scheduler {} not implemented.'.format(args.scheduler))
        print('Using {} scheduler.'.format(args.scheduler))

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        self.logger = SummaryWriter(
            os.path.join(self.workspace, 'log'))

        self.best_epoch = 0
        self.best_roc = 0.

    def compute_loss(self, labels, logits):
        return self.loss_fn(logits, labels)

    def compute_metrics(self, labels, logits):
        return f1_score(labels, logits.argmax(axis=1), average='macro')

    def early_stop(self, roc, epoch):
        cur_roc = sum(roc) if type(roc) == list else roc

        if cur_roc > self.best_roc:
            self.best_roc = cur_roc
            self.best_epoch = epoch
            self.save_model()

        return (epoch - self.best_epoch >= self.tol)

    def train_one_epoch(self):
        total_loss = 0.
        self.model.train()

        num_batches = (len(self.train_dataloader) // self.batch_size) + 1
        for _ in range(num_batches):
            count = 0
            preds, labels = list(), list()

            for citation_context, citing_abstract, cited_abstract, label in self.train_dataloader:
                citation_context = listofdict2gpu(citation_context[0], self.device)
                citing_abstract = dict2gpu(citing_abstract[0], self.device)
                cited_abstract = dict2gpu(cited_abstract[0], self.device)
                for i in range(len(citation_context)):
                    citation_context[i]['readout_mask'] = None
                preds.append(self.model(
                    None,
                    cited_abstract,
                    None,
                    citing_abstract,
                    citation_context
                ))
                labels.append(label)
                count += 1

                if count == self.batch_size:
                    count = 0
                    break

            preds = torch.cat(preds, dim=0)
            labels = torch.LongTensor(labels).to(self.device).squeeze(dim=1)
            loss = self.compute_loss(labels, preds)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches

    def eval_one_epoch(self, test):
        self.model.eval()

        if test:
            data_loader = self.test_dataloader  # for inference
        else:
            data_loader = self.val_dataloader  # for validation

        preds, labels = list(), list()
        with torch.no_grad():
            for citation_context, citing_abstract, cited_abstract, label in data_loader:
                citation_context = listofdict2gpu(citation_context[0], self.device)
                citing_abstract = dict2gpu(citing_abstract[0], self.device)
                cited_abstract = dict2gpu(cited_abstract[0], self.device)
                for i in range(len(citation_context)):
                    citation_context[i]['readout_mask'] = None
                preds.append(self.model(
                    None,
                    cited_abstract,
                    None,
                    citing_abstract,
                    citation_context
                ).detach().cpu().numpy())
                labels.append(label)

            preds = np.concatenate(preds, axis=0)
            labels = np.concatenate(labels, axis=0)

            roc = self.compute_metrics(labels, preds)

            if test:
                return roc, preds
            return roc

    def train(self):
        for epoch in range(1, self.num_epochs+1):
            start = time.time()
            loss = self.train_one_epoch()
            end_train = time.time()
            roc = self.eval_one_epoch(test=False)
            end = time.time()

            self.log_tensorboard(roc, loss, epoch)
            self.log_print(roc, loss, epoch, end_train-start, end-end_train)
            self.scheduler.step()

            if self.early_stop(roc=roc, epoch=epoch):
                break

    def test(self):
        roc, preds = self.eval_one_epoch(test=True)
        print('Test results:')
        print('Test roc: {:.4f}'.format(roc))
        print('Best val roc: {:.4f}'.format(self.best_roc))
        return preds

    def save_model(self):
        model_filename = os.path.join(self.workspace, 'best_model.pt')
        torch.save(self.model, model_filename)

    def load_model(self):
        model_filename = os.path.join(self.workspace, 'best_model.pt')
        self.model = torch.load(model_filename)

    def log_tensorboard(self, roc, loss, epoch):
        ''' Write experiment log to tensorboad. '''
        cur_roc = roc[0] if type(roc) == list else roc

        self.logger.add_scalar('Training loss', loss, epoch)
        self.logger.add_scalar('ROC', cur_roc, epoch)

    def log_print(self, roc, loss, epoch, train_time, eval_time):
        ''' Stdout of experiment log. '''

        if type(roc) == list:
            main_roc = roc[0]
            aux_rocs = ', '.join(['%.4f' % r for r in roc[1:]])
            print('Epoch: {}, train time: {:.4f}, eval time: {:.4f}, training loss: {:.4f}, main val roc: {:.4f}, aux roc: {}'.format(
                epoch, train_time, eval_time, loss, main_roc, aux_rocs))
        elif type(roc) in [float, np.float64]:
            print('Epoch: {}, train time: {:.4f}, eval time: {:.4f}, training loss: {:.4f}, val roc: {:.4f}'.format(
                epoch, train_time, eval_time, loss, roc))
        else:
            raise TypeError('Type roc {} is wrong.'.format(type(roc)))
