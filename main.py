import os
from re import M

import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np

from Model import SASRec
from optimizer import Optimizers
from data import create_data_channels, CollateFn
from train import train
from test import test
from utils import log_tensorboard, log_print, save_args

seed = 1209384752;
# seed = 1209384756
""" 1209384756"""
frac_list = [0.05*i for i in range(1, 20)];
num_runs = 100;
learning_rate = 5e-5;
lr1 = 0.0001;
lr2 = 0.0001;
lr3 = 0.0001;
lr4 = 0;
l2_regularization = 0.0001;
num_epochs = 100;
patience = 5;
batch_size = 8;
verbose = False;
hidden_dim = 4;
save_postfix = "mlp"
device = 'cuda'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--workspace', required=True)

    # training configuration
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--tol', default=10, type=int)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--seed', default=1209384752, type=int)

    # model configuration
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--hidden_dims', default='1024,128,8', type=str)
    parser.add_argument('--early_fuse', action='store_true')
    parser.add_argument('--rawtext_readout', default='cls', type=str)
    parser.add_argument('--context_readout', default='ch', type=str)
    parser.add_argument('--intra_context_pooling', default='mean', type=str)
    parser.add_argument('--inter_context_pooling', default='mean', type=str)

    args = parser.parse_args()

    # fix all random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True

