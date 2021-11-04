import os
import argparse
import torch
import random
import numpy as np

from Model import LanguageModel, EarlyFuseClassifier, EarlyFuseMLPClassifier, LateFuseClassifier, LateFuseMLPCLassifier, MultiHeadEarlyFuseClassifier, MultiHeadLateFuseClassifier, MultiHeadContextOnlyLateFuseClassifier
from Model.model import ContextOnlyLateFuseClassifier
from data import EmbeddedDataset, MultiHeadDatasets, SingleHeadDatasets, SingleHeadEmbeddedDatasets, create_data_channels
from train import Trainer, PreTrainer, MultiHeadTrainer, SingleHeadTrainer, SingleHeadPreTrainer
from utils import save_args, select_activation_fn

# change the default cache dir so that huggingface won't take the cse space.
os.environ['TRANSFORMERS_CACHE'] = '/export/scratch/zeren/KimNLP/HuggingfaceCache/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--workspace', required=True)

    parser.add_argument('--aux_datasets', default='', type=str)

    # training configuration
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--batch_size_finetune', default=12, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_finetune', default=5e-5, type=float)
    parser.add_argument('--decay_rate', default=0.5, type=float)
    parser.add_argument('--decay_step', default=5, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_epochs_finetune', default=100, type=int)
    parser.add_argument('--scheduler', default='slanted', type=str)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2', default=1e-6, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--split_ratios', default='0.7,0.2,0.1', type=str)
    parser.add_argument('--lambdas', default='1.0', type=str)
    parser.add_argument('--tol', default=10, type=int)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--two_step', action='store_true')
    parser.add_argument('--diff_lr', action='store_true')
    parser.add_argument('--seed', default=1209384752,
                        type=int)  # seed = 1209384756

    # model configuration
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--hidden_dims', default='1024,128,8', type=str)
    parser.add_argument('--activation_fn', default='relu', type=str)
    parser.add_argument('--context_only', action='store_true')
    parser.add_argument('--fuse_type', default='bruteforce', type=str)
    parser.add_argument('--rawtext_readout', default='cls', type=str)
    parser.add_argument('--context_readout', default='ch', type=str)
    parser.add_argument('--intra_context_pooling', default='mean', type=str)
    parser.add_argument('--inter_context_pooling', default='mean', type=str)

    # data augmentation configuration
    parser.add_argument('--da_alpha', default=0.05, type=float)
    parser.add_argument('--da_n', default=16, type=int)
    parser.add_argument('--da', action='store_true')

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
    if args.aux_datasets != '':
        aux_data_filenames = [os.path.join(args.data_dir, dataset+'.tsv') for dataset in args.aux_datasets.split(',')]
    else:
        aux_data_filenames = []
    # modelname = 'xlnet-base-cased' 
    modelname = 'allenai/scibert_scivocab_uncased'
    split_ratios = list(map(float, args.split_ratios.split(',')))
    hidden_dims = list(map(int, args.hidden_dims.split(',')))

    token_train_data, token_val_data, token_test_data = create_data_channels(
        data_filename,
        modelname,
        split_ratios=split_ratios,
        fuse_type=args.fuse_type,
        context_only=args.context_only,
        max_length=args.max_length,
        da_alpha=args.da_alpha,
        da_n=args.da_n
    )

    token_train_data_list = [token_train_data]
    token_val_data_list = [token_val_data]
    token_test_data_list = [token_test_data]
    for aux_d in aux_data_filenames:
        aux_token_train_data, aux_token_val_data, aux_token_test_data = create_data_channels(
            aux_d,
            modelname,
            split_ratios=split_ratios,
            fuse_type=args.fuse_type,
            context_only=args.context_only,
            max_length=args.max_length,
            da_alpha=args.da_alpha,
            da_n=args.da_n
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

    if args.context_only:
        if args.fuse_type in ['disttrunc', 'bruteforce']:
            mlp_model = EarlyFuseMLPClassifier(
                input_dims=lm_model.hidden_size,
                hidden_list=hidden_dims,
                n_classes=n_classes,
                activation=select_activation_fn(args.activation_fn),
                dropout=args.dropout_rate,
                device=args.device
            ).to(args.device)

            model = EarlyFuseClassifier(
                lm_model=lm_model,
                mlp_model=mlp_model
            )
        else:
            mlp_model = EarlyFuseMLPClassifier(
                input_dims=lm_model.hidden_size,
                hidden_list=hidden_dims,
                n_classes=n_classes,
                activation=select_activation_fn(args.activation_fn),
                dropout=args.dropout_rate,
                device=args.device
            ).to(args.device)

            model = ContextOnlyLateFuseClassifier(
                lm_model=lm_model,
                mlp_model=mlp_model,
                inter_context_pooling=args.inter_context_pooling
            )
    else:
        if args.fuse_type in ['disttrunc', 'bruteforce']:
            mlp_model = EarlyFuseMLPClassifier(
                input_dims=lm_model.hidden_size,
                hidden_list=hidden_dims,
                n_classes=n_classes,
                activation=select_activation_fn(args.activation_fn),
                dropout=args.dropout_rate,
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
                activation=select_activation_fn(args.activation_fn),
                dropout=args.dropout_rate,
                device=args.device
            ).to(args.device)

            model = LateFuseClassifier(
                lm_model=lm_model,
                mlp_model=mlp_model,
                inter_context_pooling=args.inter_context_pooling
            )

    if not args.inference_only:
        # warmup each head
        if args.two_step:
            tensor_train_data_list = list()
            for head_idx in range(len(token_train_data_list)):
                tensor_train_data = EmbeddedDataset(
                    token_train_data_list[head_idx], lm_model, args.fuse_type, args.context_only, inter_context_pooling=args.inter_context_pooling)
                tensor_train_data_list.append(tensor_train_data)
            singlehead_tensor_train_data = SingleHeadEmbeddedDatasets(tensor_train_data_list)

            tensor_val_data = EmbeddedDataset(token_val_data, lm_model, args.fuse_type, args.context_only, inter_context_pooling=args.inter_context_pooling)
            tensor_test_data = EmbeddedDataset(token_test_data, lm_model, args.fuse_type, args.context_only, inter_context_pooling=args.inter_context_pooling)
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

        if args.da:
            token_train_data.data_augmentation()

        finetuner = SingleHeadTrainer(
            model,
            singlehead_train_datasets,
            token_val_data,
            token_test_data,
            args
        )
        print('Finetuning LM + MLP.')
        finetuner.train()
        finetuner.load_model()
        finetuner.test()
    else:
        finetuner = SingleHeadTrainer(
            model,
            singlehead_train_datasets,
            token_val_data,
            token_test_data,
            args
        )
        finetuner.load_model()
        finetuner.test()