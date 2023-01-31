import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from scipy.special import softmax
from sklearn.metrics import roc_auc_score, f1_score
from scheduler import get_slanted_triangular_scheduler, get_linear_scheduler


class Trainer(object):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        self.device = args.device
        self.batch_size = args.batch_size_finetune
        self.num_epochs = args.num_epochs_finetune
        self.early_fuse = args.fuse_type in ['bruteforce']
        self.tol = args.tol

        self.workspace = args.workspace
        self.model = model

        self.loss_fn = nn.CrossEntropyLoss(
            weight=train_dataset.get_label_weights().to(self.device))

        if not args.one_step or not args.diff_lr:
            self.optimizer = AdamW(self.model.parameters(),
                                   lr=args.lr_finetune, weight_decay=args.l2)
        else:
            self.optimizer = AdamW([
                {'params': self.model.lm_model.parameters(), 'lr': args.lr_finetune,
                 'weight_decay': args.l2},
                {'params': self.model.mlp_model.parameters(), 'lr': args.lr,
                 'weight_decay': args.l2}
            ])
            print('Setting different learning rates to {:.4f} for LM and {:.4f} for MLP.'.format(
                args.lr_finetune, args.lr))

        if args.scheduler == 'exp':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.decay_rate, verbose=True)
        elif args.scheduler == 'slanted':
            self.scheduler = get_slanted_triangular_scheduler(
                self.optimizer, num_epochs=self.num_epochs)
        elif args.scheduler == 'linear':
            self.scheduler = get_linear_scheduler(
                self.optimizer, num_training_epochs=self.num_epochs, initial_lr=args.lr_finetune, final_lr=args.lr_finetune/32)
        elif args.scheduler == 'const':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=self.num_epochs, gamma=1., verbose=True)
        else:
            raise ValueError(
                'Scheduler {} not implemented.'.format(args.scheduler))
        print('Using {} scheduler.'.format(args.scheduler))

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=1, shuffle=True)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False)

        self.logger = SummaryWriter(
            os.path.join(self.workspace, 'log'))

        self.best_epoch = 0
        self.best_roc = 0.

    def compute_loss(self, labels, logits):
        return self.loss_fn(logits, labels)

    def compute_metrics(self, labels, logits):
        # return roc_auc_score(labels, logits[:, 1])  # temp
        # temp
        return f1_score(labels, logits.argmax(axis=1), average='macro')
        # return roc_auc_score(labels, softmax(logits, axis=1), multi_class='ovo')

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

            for fused_context, label in self.train_dataloader:
                preds.append(self.model(fused_context))
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
            for fused_context, label in data_loader:
                preds.append(self.model(
                    fused_context).detach().cpu().numpy())
                labels.append(label)

            preds = np.concatenate(preds, axis=0)
            labels = torch.cat(labels, dim=0).numpy()

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
        torch.save(self.model.state_dict(), model_filename)

    def load_model(self):
        model_filename = os.path.join(self.workspace, 'best_model.pt')
        self.model.load_state_dict(torch.load(model_filename))

    def checkpoint(self, epoch):
        checkpoint_filename = os.path.join(self.workspace, 'checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, checkpoint_filename)

    def load_checkpoint(self):
        checkpoint_filename = os.path.join(self.workspace, 'checkpoint.pt')
        state_dict = torch.load(checkpoint_filename)

        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

        return state_dict['epoch']

    # def log_tensorboard(self, roc, loss, epoch):
    #     ''' Write experiment log to tensorboad. '''

    #     self.logger.add_scalar('Training loss', loss, epoch)
    #     self.logger.add_scalar('ROC', roc, epoch)

    def log_tensorboard(self, roc, loss, epoch):
        ''' Write experiment log to tensorboad. '''
        cur_roc = roc[0] if type(roc) == list else roc

        self.logger.add_scalar('Training loss', loss, epoch)
        self.logger.add_scalar('ROC', cur_roc, epoch)

    # def log_print(self, roc, loss, epoch, train_time, eval_time):
    #     ''' Stdout of experiment log. '''

    #     print('Epoch: {}, train time: {:.4f}, eval time: {:.4f}, training loss: {:.4f}, val roc: {:.4f}'.format(
    #         epoch, train_time, eval_time, loss, roc))

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


class PreTrainer(Trainer):
    def __init__(self, mlp_model, train_dataset, val_dataset, test_dataset, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.early_fuse = args.fuse_type in ['bruteforce']
        self.tol = args.tol

        self.workspace = args.workspace
        self.model = mlp_model

        self.loss_fn = nn.CrossEntropyLoss(
            weight=train_dataset.get_label_weights().to(self.device))

        self.optimizer = Adam(self.model.parameters(),
                              lr=args.lr, weight_decay=args.l2)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=self.num_epochs, gamma=1.
        )

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)

        self.logger = SummaryWriter(os.path.join(self.workspace, 'log'))

        self.best_epoch = 0
        self.best_roc = 0.

    def train_one_epoch(self):
        total_loss = 0.
        self.model.train()

        for fused_context, labels in self.train_dataloader:
            preds = self.model(fused_context)
            labels = torch.LongTensor(labels).to(self.device)

            loss = self.compute_loss(labels, preds)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def save_model(self):
        model_filename = os.path.join(self.workspace, 'best_mlp_model.pt')
        torch.save(self.model.state_dict(), model_filename)

    def load_model(self):
        model_filename = os.path.join(self.workspace, 'best_mlp_model.pt')
        self.model.load_state_dict(torch.load(model_filename))

    def checkpoint(self, epoch):
        checkpoint_filename = os.path.join(self.workspace, 'mlp_checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_filename)

    def load_checkpoint(self):
        checkpoint_filename = os.path.join(self.workspace, 'mlp_checkpoint.pt')
        state_dict = torch.load(checkpoint_filename)

        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        return state_dict['epoch']


class MultiHeadTrainer(Trainer):
    def __init__(self, model, train_datasets, val_datasets, test_dataset, args):
        self.device = args.device
        self.batch_size = args.batch_size_finetune
        self.num_epochs = args.num_epochs_finetune
        self.early_fuse = args.fuse_type in ['bruteforce']
        self.tol = args.tol

        self.workspace = args.workspace
        self.model = model

        self.loss_fns = [nn.CrossEntropyLoss(weight=lb_weights.to(
            self.device)) for lb_weights in train_datasets.get_label_weights()]
        # self.loss_fns = [nn.CrossEntropyLoss() for lb_weights in train_datasets.get_label_weights()]
        self.lambdas = list(map(float, args.lambdas.split(',')))

        assert len(self.lambdas) == len(
            self.loss_fns), "Number of loss functions should be the same with the number of lambdas."
        assert np.abs(
            self.lambdas[0] - 1.) < 1e-8, "Lambda for the main dataset should be one."

        self.num_heads = len(self.loss_fns)
        # self.num_ratios = np.array(
        #     [d / train_datasets.lengths[0] for d in train_datasets.lengths])
        # self.num_ratios[self.num_ratios >= 3] = 3.  # truncation
        self.num_ratios = np.array(
            [min(train_datasets.lengths[0] / d, 1) for d in train_datasets.lengths])

        print('Lambdas and num_ratios')
        print(self.lambdas)
        print(self.num_ratios)

        if not args.one_step or not args.diff_lr:
            self.optimizer = AdamW(self.model.parameters(),
                                   lr=args.lr_finetune, weight_decay=args.l2)
        else:
            self.optimizer = AdamW([
                {'params': self.model.lm_model.parameters(), 'lr': args.lr_finetune,
                 'weight_decay': args.l2},
                {'params': self.model.mlp_model.parameters(), 'lr': args.lr,
                 'weight_decay': args.l2}
            ])
            print('Setting different learning rates to {:.4f} for LM and {:.4f} for MLP.'.format(
                args.lr_finetune, args.lr))

        if args.scheduler == 'exp':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.decay_rate, verbose=True)
        elif args.scheduler == 'slanted':
            self.scheduler = get_slanted_triangular_scheduler(
                self.optimizer, num_epochs=self.num_epochs)
        elif args.scheduler == 'linear':
            self.scheduler = get_linear_scheduler(
                self.optimizer, num_training_epochs=self.num_epochs, initial_lr=args.lr_finetune, final_lr=args.lr_finetune/32)
        elif args.scheduler == 'const':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=self.num_epochs, gamma=1., verbose=True)
        else:
            raise ValueError(
                'Scheduler {} not implemented.'.format(args.scheduler))
        print('Using {} scheduler.'.format(args.scheduler))

        self.train_dataloaders = DataLoader(
            train_datasets, batch_size=1, shuffle=True)
        self.val_dataloaders = [DataLoader(
            val_dataset, batch_size=1, shuffle=False) for val_dataset in val_datasets]
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False)

        self.logger = SummaryWriter(
            os.path.join(self.workspace, 'log'))

        self.best_epoch = 0
        self.best_roc = 0.

    def compute_loss(self, labels, logits):
        loss = 0.
        for head_idx in range(self.num_heads):
            if labels[head_idx].size(0) > 0:
                loss = loss + self.loss_fns[head_idx](
                    logits[head_idx],
                    labels[head_idx]
                ) * self.lambdas[head_idx] * self.num_ratios[head_idx]

        return loss

    def train_one_epoch(self):
        total_loss = 0.
        self.model.train()

        num_batches = (len(self.train_dataloaders) // self.batch_size) + 1
        for _ in range(num_batches):
            preds = [[] for _ in range(self.num_heads)]
            labels = [[] for _ in range(self.num_heads)]

            count = 0
            for head_idx, context, label in self.train_dataloaders:
                preds[head_idx].append(
                    self.model(
                        head_idx,
                        context
                    )
                )
                labels[head_idx].append(label)
                count += 1
                if count == self.batch_size:
                    count = 0
                    break
            
            final_preds = list()
            for p in preds:
                if len(p) > 0:
                    final_preds.append(torch.cat(p, dim=0))
                else:
                    final_preds.append([])
            # preds = [torch.cat(p, dim=0) for p in preds]
            labels = [torch.LongTensor(lb).to(self.device) for lb in labels]

            loss = self.compute_loss(labels, final_preds)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches

    def eval_one_epoch(self, test):
        self.model.eval()

        if test:  # for inference
            preds, labels = list(), list()
            with torch.no_grad():
                for fused_context, label in self.test_dataloader:
                    preds.append(self.model(
                        0, fused_context).detach().cpu().numpy())
                    labels.append(label)

                preds = np.concatenate(preds, axis=0)
                labels = torch.cat(labels, dim=0).numpy()

                roc = self.compute_metrics(labels, preds)
                return roc
        else:
            with torch.no_grad():
                roc = list()
                for head_idx, data_loader in enumerate(self.val_dataloaders):
                    preds, labels = list(), list()
                    for fused_context, label in data_loader:
                        preds.append(self.model(
                            head_idx, fused_context).detach().cpu().numpy())
                        labels.append(label)

                    preds = np.concatenate(preds, axis=0)
                    labels = torch.cat(labels, dim=0).numpy()

                    roc.append(self.lambdas[head_idx] *
                               self.compute_metrics(labels, preds))
                return roc


class SingleHeadTrainer(Trainer):
    def __init__(self, model, train_datasets, val_datasets, test_dataset, args):
        self.device = args.device
        self.batch_size = args.batch_size_finetune
        self.num_epochs = args.num_epochs_finetune
        self.early_fuse = args.fuse_type in ['bruteforce']
        self.tol = args.tol

        self.workspace = args.workspace
        self.model = model

        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=train_datasets.get_label_weights().to(self.device))
        self.lambdas = list(map(float, args.lambdas.split(',')))

        assert len(self.lambdas) == len(
            train_datasets.datasets), "Number of datasets should be the same with the number of lambdas."
        assert np.abs(
            self.lambdas[0] - 1.) < 1e-8, "Lambda for the main dataset should be one."

        self.num_heads = len(self.lambdas)
        self.num_ratios = np.array(
            [min(train_datasets.lengths[0] / d, 1) for d in train_datasets.lengths])
        # self.num_ratios[self.num_ratios >= 3] = 3.  # truncation

        print('Lambdas and num_ratios')
        print(self.lambdas)
        print(self.num_ratios)

        self.label_indices = train_datasets.get_label_indices()
        self.label_offsets = train_datasets.get_label_offsets()

        if not args.one_step or not args.diff_lr:
            self.optimizer = AdamW(self.model.parameters(),
                                   lr=args.lr_finetune, weight_decay=args.l2)
        else:
            self.optimizer = AdamW([
                {'params': self.model.lm_model.parameters(), 'lr': args.lr_finetune,
                 'weight_decay': args.l2},
                {'params': self.model.mlp_model.parameters(), 'lr': args.lr,
                 'weight_decay': args.l2}
            ])
            print('Setting different learning rates to {:.4f} for LM and {:.4f} for MLP.'.format(
                args.lr_finetune, args.lr))

        if args.scheduler == 'exp':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.decay_rate, verbose=True)
        elif args.scheduler == 'slanted':
            self.scheduler = get_slanted_triangular_scheduler(
                self.optimizer, num_epochs=self.num_epochs)
        elif args.scheduler == 'linear':
            self.scheduler = get_linear_scheduler(
                self.optimizer, num_training_epochs=self.num_epochs, initial_lr=args.lr_finetune, final_lr=args.lr_finetune/32)
        elif args.scheduler == 'const':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=self.num_epochs, gamma=1., verbose=True)
        else:
            raise ValueError(
                'Scheduler {} not implemented.'.format(args.scheduler))
        print('Using {} scheduler.'.format(args.scheduler))

        self.train_dataloaders = DataLoader(
            train_datasets, batch_size=1, shuffle=True)
        self.val_dataloaders = [DataLoader(
            val_dataset, batch_size=1, shuffle=False) for val_dataset in val_datasets]
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False)

        self.logger = SummaryWriter(
            os.path.join(self.workspace, 'log'))

        self.best_epoch = 0
        self.best_roc = 0.

    def compute_loss(self, labels, logits):
        loss = 0.
        for head_idx in range(self.num_heads):
            if labels[head_idx].size(0) > 0:
                loss = loss + self.loss_fn(
                    logits[head_idx],
                    labels[head_idx] + self.label_offsets[head_idx]
                ) * self.lambdas[head_idx]# * self.num_ratios[head_idx]

        return loss

    def train_one_epoch(self):
        total_loss = 0.
        self.model.train()

        num_batches = (len(self.train_dataloaders) // self.batch_size) + 1
        for _ in range(num_batches):
            preds = [[] for _ in range(self.num_heads)]
            labels = [[] for _ in range(self.num_heads)]

            count = 0
            for head_idx, context, label in self.train_dataloaders:
                preds[head_idx].append(self.model(context))
                labels[head_idx].append(label)

                count += 1
                if count == self.batch_size:
                    count = 0
                    break
            
            final_preds = list()
            for p in preds:
                if len(p) > 0:
                    final_preds.append(torch.cat(p, dim=0))
                else:
                    final_preds.append([])
            # preds = [torch.cat(p, dim=0) for p in preds]
            labels = [torch.LongTensor(lb).to(self.device) for lb in labels]

            loss = self.compute_loss(labels, final_preds)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches

    def eval_one_epoch(self, test):
        self.model.eval()

        if test:  # for inference
            preds, labels = list(), list()
            with torch.no_grad():
                for fused_context, label in self.test_dataloader:
                    preds.append(self.model(
                        fused_context).detach().cpu().numpy())
                    labels.append(label)

                preds = np.concatenate(preds, axis=0)[:, self.label_indices[0]]
                labels = torch.cat(labels, dim=0).numpy()

                roc = self.compute_metrics(labels, preds)
                return roc, preds
        else:  # for validation
            with torch.no_grad():
                roc = list()
                for head_idx, data_loader in enumerate(self.val_dataloaders):
                    preds, labels = list(), list()
                    for fused_context, label in data_loader:
                        preds.append(self.model(
                            fused_context).detach().cpu().numpy())
                        labels.append(label)

                    preds = np.concatenate(preds, axis=0)[
                        :, self.label_indices[head_idx]]
                    labels = torch.cat(labels, dim=0).numpy()

                    roc.append(self.lambdas[head_idx] *
                               self.compute_metrics(labels, preds))
                return roc


class SingleHeadPreTrainer(Trainer):
    def __init__(self, mlp_model, train_dataset, val_dataset, test_dataset, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.early_fuse = args.fuse_type in ['bruteforce']
        self.tol = args.tol

        self.workspace = args.workspace
        self.model = mlp_model

        self.loss_fn = nn.CrossEntropyLoss(
            weight=train_dataset.get_label_weights().to(self.device))
        # self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(),
                              lr=args.lr, weight_decay=args.l2)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=self.num_epochs, gamma=1.
        )

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)

        self.logger = SummaryWriter(os.path.join(self.workspace, 'log'))

        self.label_indices = train_dataset.get_label_indices()

        self.best_epoch = 0
        self.best_roc = 0.

    def train_one_epoch(self):
        total_loss = 0.
        self.model.train()

        for fused_context, labels in self.train_dataloader:
            preds = self.model(fused_context)
            labels = torch.LongTensor(labels).to(self.device)

            loss = self.compute_loss(labels, preds)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def eval_one_epoch(self, test):
        self.model.eval()

        if test:
            data_loader = self.test_dataloader  # for inference
        else:
            data_loader = self.val_dataloader  # for validation

        preds, labels = list(), list()
        with torch.no_grad():
            for fused_context, label in data_loader:
                preds.append(self.model(
                    fused_context).detach().cpu().numpy())
                labels.append(label)

            preds = np.concatenate(preds, axis=0)[:, self.label_indices[0]]
            labels = torch.cat(labels, dim=0).numpy()

            roc = self.compute_metrics(labels, preds)
            return roc

    def save_model(self):
        model_filename = os.path.join(self.workspace, 'best_mlp_model.pt')
        torch.save(self.model.state_dict(), model_filename)

    def load_model(self):
        model_filename = os.path.join(self.workspace, 'best_mlp_model.pt')
        self.model.load_state_dict(torch.load(model_filename))

    def checkpoint(self, epoch):
        checkpoint_filename = os.path.join(self.workspace, 'mlp_checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_filename)

    def load_checkpoint(self):
        checkpoint_filename = os.path.join(self.workspace, 'mlp_checkpoint.pt')
        state_dict = torch.load(checkpoint_filename)

        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        return state_dict['epoch']
