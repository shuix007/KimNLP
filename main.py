import os
import argparse
import torch
import random
import numpy as np

from Model import LanguageModel, EarlyFuseClassifier, EarlyFuseMLPClassifier, LateFuseClassifier, LateFuseMLPCLassifier
from data import EmbeddedDataset, create_data_channels
from train import Trainer, PreTrainer
from utils import save_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--workspace', required=True)

    # training configuration
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batch_size_finetune', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_finetune', default=5e-5, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2', default=0.0001, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--split_ratios', default='0.7,0.2,0.1', type=str)
    parser.add_argument('--tol', default=5, type=int)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--seed', default=1209384752,
                        type=int)  # seed = 1209384756

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

    # save the arguments
    if not os.path.exists(args.workspace):
        os.mkdir(args.workspace)
    save_args(args, args.workspace)

    data_filename = os.path.join(args.data_dir, args.dataset+'.tsv')
    modelname = 'allenai/scibert_scivocab_uncased'
    split_ratios = list(map(int, args.split_ratios.split(',')))
    hidden_dims = list(map(int, args.hidden_dims.split(',')))

    token_train_data, token_val_data, token_test_data = create_data_channels(
        data_filename,
        modelname,
        split_ratios=split_ratios,
        earyly_fuse=args.early_fuse
    )
    n_classes = len(token_train_data.get_label_weights())

    lm_model = LanguageModel(
        modelname=modelname,
        device=args.device,
        rawtext_readout=args.rawtext_readout,
        context_readout=args.context_readout,
        intra_context_pooling=args.intra_context_pooling
    ).to(args.device)

    if args.early_fuse:
        mlp_model = EarlyFuseMLPClassifier(
            input_dims=lm_model.hidden_size,
            hidden_list=hidden_dims,
            n_classes=n_classes,
            activation=torch.nn.ReLU(),
            dropout=args.dropout,
            device=args.device
        ).to(args.device)

        model = EarlyFuseClassifier(
            lm_model=lm_model,
            mlp_model=mlp_model
        )
    else:
        mlp_model = LateFuseMLPCLassifier(
            input_dims=lm_model.hidden_size,
            hidden_list=hidden_dims,
            n_classes=n_classes,
            activation=torch.nn.ReLU(),
            dropout=args.dropout,
            device=args.device
        ).to(args.device)

        model = LateFuseClassifier(
            lm_model=lm_model,
            mlp_model=mlp_model
        )

    tensor_train_data = EmbeddedDataset(
        token_train_data, lm_model, args.early_fuse, inter_context_pooling=args.inter_context_pooling)
    tensor_val_data = EmbeddedDataset(
        token_val_data, lm_model, args.early_fuse, inter_context_pooling=args.inter_context_pooling)
    tensor_test_data = EmbeddedDataset(
        token_test_data, lm_model, args.early_fuse, inter_context_pooling=args.inter_context_pooling)

    mlp_trainer = PreTrainer(
        mlp_model,
        tensor_train_data,
        tensor_val_data,
        tensor_test_data,
        args
    )
    mlp_trainer.train()
    mlp_trainer.load_model()
    mlp_trainer.test()

    finetuner = Trainer(
        model,
        token_train_data,
        token_val_data,
        token_test_data,
        args
    )
    finetuner.train()
    finetuner.load_model()
    finetuner.test()
