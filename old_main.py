import os
# change the default cache dir so that huggingface won't take the cse space.
os.environ['TRANSFORMERS_CACHE'] = '/export/scratch/zeren/KimNLP/HuggingfaceCache/'

from utils import save_args, select_activation_fn
from old_train import Trainer
from old_data import create_data_channels
from old_model import BertClassifier

from transformers import AutoModel
import numpy as np
import random
import torch
import argparse


def main(args):
    data_filename = os.path.join(args.data_dir, args.dataset+'.tsv')
    # modelname = 'allenai/scibert_scivocab_uncased' if args.lm == 'scibert' else 'bert-base-uncased'
    model_name = args.lm
    # hidden_dims = list(map(int, args.hidden_dims.split(',')))

    token_train_data, token_val_data, token_test_data = create_data_channels(
        data_filename,
        model_name,
        max_length=args.max_length
    )

    n_classes = len(token_train_data.get_label_weights())

    lm_model = AutoModel.from_pretrained(model_name).to(args.device)
    model = BertClassifier(
        n_classes=n_classes,
        bert=lm_model,
        emb_dim=768,
        hidden_list=[1024, 128, 8],
        dropout=0.2,
        flag="mean",
        readout="mean",
        pooling="max"
    ).to(args.device)

    if not args.inference_only:
        finetuner = Trainer(
            model,
            token_train_data,
            token_val_data,
            token_test_data,
            args
        )
        print('Finetuning LM + MLP.')
        finetuner.train()
        finetuner.load_model()
        preds = finetuner.test()
    else:
        finetuner = Trainer(
            model,
            token_train_data,
            token_val_data,
            token_test_data,
            args
        )
        finetuner.load_model()
        preds = finetuner.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--workspace', required=True)

    # training configuration
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--decay_rate', default=0.5, type=float)
    parser.add_argument('--decay_step', default=5, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--scheduler', default='slanted', type=str)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--tol', default=10, type=int)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--use_abstract', action='store_true')
    parser.add_argument('--seed', default=42, type=int)  # seed = 1209384756

    # model configuration
    parser.add_argument('--lm', default='bert', type=str)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--readout', default='mean', type=str)

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

    main(args)
