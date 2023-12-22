import os
# change the default cache dir so that huggingface won't take the cse space.
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme1n1/zeren/HuggingfaceCache/'

from utils import save_args
from trainer import Trainer, MultiHeadTrainer
from data import create_data_channels, create_single_data_object, Datasets, MultiHeadDatasets
from Model import LanguageModel, MultiHeadPsuedoLanguageModel, MultiHeadContrastiveLanguageModel
import numpy as np
import random
import torch
import argparse

N_CLASSES = {
    'kim': 3,
    'acl': 6,
    'scicite': 3
}

def main_pl(args):
    # data_filename = os.path.join(args.data_dir, args.dataset+'.tsv')
    datasets = args.dataset.split('-')
    data_filenames = [os.path.join(args.data_dir, ds+'.tsv') for ds in datasets]

    if args.lm == 'scibert':
        modelname = 'allenai/scibert_scivocab_uncased'
    elif args.lm == 'bert':
        modelname = 'bert-base-uncased'
    else:
        modelname = args.lm

    train_data, val_data, test_data, model_label_map = create_data_channels(
        data_filenames[0],
        args.class_definition
    )
    if len(data_filenames) > 1:
        aux_data, aux_label_map = create_single_data_object(
            data_filenames[1], args.class_definition, split='train'
        )

    model = MultiHeadPsuedoLanguageModel(
        modelname=modelname,
        device=args.device,
        readout=args.readout,
        num_classes=[N_CLASSES[datasets[0]]] # [N_CLASSES[ds] for ds in datasets]
    ).to(args.device)

    if not args.inference_only:
        finetuner = Trainer(
            model,
            train_data,
            val_data,
            test_data,
            args
        )
        print('Finetuning LM + MLP.')
        finetuner.train()
        finetuner.load_model()
        preds = finetuner.test()
    else:
        finetuner = Trainer(
            model,
            train_data,
            val_data,
            test_data,
            args
        )
        finetuner.load_model()
        preds = finetuner.test()

    for i in range(10):
        # model.print_label_space_mapping(label_maps)
        print('The {}-th PL iteration.'.format(i))
        aux_preds = finetuner.test(outside_dataset=aux_data)
        aux_data.visualize_confusion_matrix(aux_preds, aux_label_map, model_label_map)

        print('Pseudo-labeling the auxiliary dataset.')
        if args.pl == 'pl':
            aux_data.pseudo_label(aux_preds)
        elif args.pl == 'pls':
            aux_data.update_label_with_selection(aux_preds)
        else:
            raise NotImplementedError
        train_datasets = Datasets([train_data, aux_data])

        model = MultiHeadPsuedoLanguageModel(
            modelname=modelname,
            device=args.device,
            readout=args.readout,
            num_classes=[N_CLASSES[datasets[0]]] # [N_CLASSES[ds] for ds in datasets]
        ).to(args.device)

        finetuner = Trainer(
            model,
            train_datasets,
            val_data,
            test_data,
            args
        )
        print('Finetuning LM + MLP with pseudo-labels.')
        finetuner.train()
        finetuner.load_model()
        preds = finetuner.test()

def main_cl(args):
    # data_filename = os.path.join(args.data_dir, args.dataset+'.tsv')
    datasets = args.dataset.split('-')
    data_filenames = [os.path.join(args.data_dir, ds+'.tsv') for ds in datasets]

    if args.lm == 'scibert':
        modelname = 'allenai/scibert_scivocab_uncased'
    elif args.lm == 'bert':
        modelname = 'bert-base-uncased'
    else:
        modelname = args.lm

    train_data, val_data, test_data, model_label_map = create_data_channels(
        data_filenames[0],
        args.class_definition
    )
    train_datasets_list = [train_data]
    if len(data_filenames) > 1:
        for data_filename in data_filenames[1:]:
            aux_data, aux_label_map = create_single_data_object(
                data_filename, args.class_definition, split='train'
            )
            train_datasets_list.append(aux_data)
    train_datasets = MultiHeadDatasets(train_datasets_list)

    model = MultiHeadContrastiveLanguageModel(
        modelname=modelname,
        device=args.device,
        readout=args.readout
    ).to(args.device)

    if not args.inference_only:
        finetuner = MultiHeadTrainer(
            model,
            train_datasets,
            val_data,
            test_data,
            args
        )
        print('Finetuning LM + MLP.')
        finetuner.train()
        finetuner.load_model()
        preds = finetuner.test()
    else:
        finetuner = MultiHeadTrainer(
            model,
            train_datasets,
            val_data,
            test_data,
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
    parser.add_argument('--class_definition', default='Data/class_def.json', type=str)

    # training configuration
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
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
    parser.add_argument('--lm', default='scibert', type=str)
    parser.add_argument('--pl', default='pls', type=str)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--readout', default='cls', type=str)

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

    # main_pl(args)
    main_cl(args)
