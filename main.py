import os
# change the default cache dir so that huggingface won't take the cse space.
os.environ['TRANSFORMERS_CACHE'] = '/export/scratch/zeren/KimNLP/HuggingfaceCache/'

from utils import save_args, select_activation_fn
from train import Trainer, PreTrainer, MultiHeadTrainer, SingleHeadTrainer, SingleHeadPreTrainer
from data import EmbeddedDataset, MultiHeadDatasets, SingleHeadDatasets, SingleHeadEmbeddedDatasets, create_data_channels
from Model.layers import DenseLayer
from Model import LanguageModel, EarlyFuseClassifier, LateFuseClassifier, MLPClassifier, MultiHeadEarlyFuseClassifier, MultiHeadLateFuseClassifier
import numpy as np
import random
import torch
import argparse


def main_singlehead(args):
    data_filename = os.path.join(args.data_dir, args.dataset+'.tsv')
    if args.aux_datasets != '':
        aux_data_filenames = [os.path.join(
            args.data_dir, dataset+'.tsv') for dataset in args.aux_datasets.split(',')]
    else:
        aux_data_filenames = []
    modelname = 'allenai/scibert_scivocab_uncased' if args.lm == 'scibert' else 'bert-base-uncased'
    hidden_dims = list(map(int, args.hidden_dims.split(',')))

    token_train_data, token_val_data, token_test_data = create_data_channels(
        data_filename,
        modelname,
        fuse_type=args.fuse_type,
        max_length=args.max_length
    )

    token_train_data_list = [token_train_data]
    token_val_data_list = [token_val_data]
    token_test_data_list = [token_test_data]
    for aux_d in aux_data_filenames:
        aux_token_train_data, aux_token_val_data, aux_token_test_data = create_data_channels(
            aux_d,
            modelname,
            fuse_type=args.fuse_type,
            max_length=args.max_length
        )
        token_train_data_list.append(aux_token_train_data)
        token_val_data_list.append(aux_token_val_data)
        token_test_data_list.append(aux_token_test_data)

    singlehead_train_datasets = SingleHeadDatasets(token_train_data_list)
    n_classes = len(singlehead_train_datasets.get_label_weights())

    lm_model = LanguageModel(
        modelname=modelname,
        device=args.device,
        rawtext_readout=args.rawtext_readout,
        context_readout=args.context_readout,
        intra_context_pooling=args.intra_context_pooling
    ).to(args.device)

    mlp_model = MLPClassifier(
        input_dims=lm_model.hidden_size,
        hidden_list=hidden_dims,
        n_classes=n_classes,
        activation=select_activation_fn(args.activation_fn),
        dropout=args.dropout_rate,
        device=args.device
    ).to(args.device)

    if args.fuse_type in ['bruteforce']:
        model = EarlyFuseClassifier(
            lm_model=lm_model,
            mlp_model=mlp_model
        )
    else:
        model = LateFuseClassifier(
            lm_model=lm_model,
            mlp_model=mlp_model,
            inter_context_pooling=args.inter_context_pooling
        )

    if not args.inference_only:
        # warmup each head
        if not args.one_step:
            tensor_train_data_list = list()
            for head_idx in range(len(token_train_data_list)):
                tensor_train_data = EmbeddedDataset(
                    token_train_data_list[head_idx], lm_model, args.fuse_type, inter_context_pooling=args.inter_context_pooling)
                tensor_train_data_list.append(tensor_train_data)
            singlehead_tensor_train_data = SingleHeadEmbeddedDatasets(
                tensor_train_data_list)

            tensor_val_data = EmbeddedDataset(
                token_val_data, lm_model, args.fuse_type, inter_context_pooling=args.inter_context_pooling)
            tensor_test_data = EmbeddedDataset(
                token_test_data, lm_model, args.fuse_type, inter_context_pooling=args.inter_context_pooling)
            mlp_trainer = SingleHeadPreTrainer(
                mlp_model,
                singlehead_tensor_train_data,
                tensor_val_data,
                tensor_test_data,
                args
            )
            print('Pretraining MLP.')
            mlp_trainer.train()
            mlp_trainer.load_model()
            mlp_trainer.test()

        finetuner = SingleHeadTrainer(
            model,
            singlehead_train_datasets,
            token_val_data_list,
            # token_val_data,
            token_test_data,
            args
        )
        print('Finetuning LM + MLP.')
        finetuner.train()
        finetuner.load_model()
        preds = finetuner.test()
    else:
        finetuner = SingleHeadTrainer(
            model,
            singlehead_train_datasets,
            token_val_data_list,
            # token_val_data,
            token_test_data,
            args
        )
        finetuner.load_model()
        preds = finetuner.test()

    token_test_data.write_prediction(
        preds,
        os.path.join(
            # args.workspace,
            '_'.join([
                args.dataset,
                args.lm,
                args.context_readout,
                str(args.seed),
                'predictions.csv'
            ])
        ),
        os.path.join(
            # args.workspace,
            '_'.join([
                args.dataset,
                args.lm,
                args.context_readout,
                str(args.seed),
                'scores.csv'
            ])
        )
    )


def main_multihead(args):
    data_filename = os.path.join(args.data_dir, args.dataset+'.tsv')
    if args.aux_datasets != '':
        aux_data_filenames = [os.path.join(
            args.data_dir, dataset+'.tsv') for dataset in args.aux_datasets.split(',')]
    else:
        aux_data_filenames = []
    modelname = 'allenai/scibert_scivocab_uncased' if args.lm == 'scibert' else 'bert-base-uncased'
    hidden_dims = list(map(int, args.hidden_dims.split(',')))

    token_train_data, token_val_data, token_test_data = create_data_channels(
        data_filename,
        modelname,
        fuse_type=args.fuse_type,
        max_length=args.max_length
    )

    token_train_data_list = [token_train_data]
    token_val_data_list = [token_val_data]
    token_test_data_list = [token_test_data]
    for aux_d in aux_data_filenames:
        aux_token_train_data, aux_token_val_data, aux_token_test_data = create_data_channels(
            aux_d,
            modelname,
            fuse_type=args.fuse_type,
            max_length=args.max_length
        )
        token_train_data_list.append(aux_token_train_data)
        token_val_data_list.append(aux_token_val_data)
        token_test_data_list.append(aux_token_test_data)

    multihead_train_datasets = MultiHeadDatasets(token_train_data_list)
    n_classes = [len(lb_weights)
                 for lb_weights in multihead_train_datasets.get_label_weights()]

    lm_model = LanguageModel(
        modelname=modelname,
        device=args.device,
        rawtext_readout=args.rawtext_readout,
        context_readout=args.context_readout,
        intra_context_pooling=args.intra_context_pooling
    ).to(args.device)

    mlp_models = list()
    for n_cls in n_classes:
        mlp_model = MLPClassifier(
            input_dims=lm_model.hidden_size,
            hidden_list=hidden_dims,
            n_classes=n_cls,
            activation=select_activation_fn(args.activation_fn),
            dropout=args.dropout_rate,
            device=args.device
        ).to(args.device)
        mlp_models.append(mlp_model)

    if args.fuse_type in ['bruteforce']:
        model = MultiHeadEarlyFuseClassifier(
            lm_model=lm_model,
            mlp_models=mlp_models
        )
    else:
        model = MultiHeadLateFuseClassifier(
            lm_model=lm_model,
            mlp_models=mlp_models,
            inter_context_pooling=args.inter_context_pooling
        )

    if not args.inference_only:
        # warmup each head
        if not args.one_step:
            for head_idx in range(len(n_classes)):
                tensor_train_data = EmbeddedDataset(
                    token_train_data_list[head_idx], lm_model, args.fuse_type, inter_context_pooling=args.inter_context_pooling)
                tensor_val_data = EmbeddedDataset(
                    token_val_data_list[head_idx], lm_model, args.fuse_type, inter_context_pooling=args.inter_context_pooling)
                tensor_test_data = EmbeddedDataset(
                    token_test_data_list[head_idx], lm_model, args.fuse_type, inter_context_pooling=args.inter_context_pooling)

                mlp_trainer = PreTrainer(
                    mlp_models[head_idx],
                    tensor_train_data,
                    tensor_val_data,
                    tensor_test_data,
                    args
                )
                print('Pretraining MLP for the {}-th head.'.format(head_idx))
                mlp_trainer.train()
                mlp_trainer.load_model()
                mlp_trainer.test()

        finetuner = MultiHeadTrainer(
            model,
            multihead_train_datasets,
            token_val_data_list,
            # [token_val_data],
            token_test_data,
            args
        )
        print('Finetuning LM + MLP.')
        finetuner.train()
        finetuner.load_model()
        finetuner.test()
    else:
        finetuner = MultiHeadTrainer(
            model,
            multihead_train_datasets,
            token_val_data_list,
            # [token_val_data],
            token_test_data,
            args
        )
        finetuner.load_model()
        finetuner.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--workspace', required=True)

    parser.add_argument('--aux_datasets', default='', type=str)

    # training configuration
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--batch_size_finetune', default=32, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_finetune', default=2e-5, type=float)
    parser.add_argument('--decay_rate', default=0.5, type=float)
    parser.add_argument('--decay_step', default=5, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_epochs_finetune', default=10, type=int)
    parser.add_argument('--scheduler', default='slanted', type=str)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--lambdas', default='1.0', type=str)
    parser.add_argument('--tol', default=10, type=int)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--one_step', action='store_true')
    parser.add_argument('--diff_lr', action='store_true')
    parser.add_argument('--seed', default=42, type=int)  # seed = 1209384756

    # model configuration
    parser.add_argument('--lm', default='bert', type=str)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--hidden_dims', default='1024,128,8', type=str)
    parser.add_argument('--activation_fn', default='relu', type=str)
    parser.add_argument('--fuse_type', default='bruteforce', type=str)
    parser.add_argument('--rawtext_readout', default='cls', type=str)
    parser.add_argument('--context_readout', default='ch', type=str)
    parser.add_argument('--intra_context_pooling', default='mean', type=str)
    parser.add_argument('--inter_context_pooling', default='mean', type=str)

    # multi-task configuration
    parser.add_argument('--multitask', default='singlehead', type=str)

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

    if args.multitask == 'singlehead':
        main_singlehead(args)

    if args.multitask == 'multihead':
        main_multihead(args)
