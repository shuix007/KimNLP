import os
# change the default cache dir so that huggingface won't take the cse space.
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme1n1/zeren/HuggingfaceCache/'

from utils import save_args
from trainer import Trainer, MultiHeadTrainer
from data import create_data_channels, create_single_data_object, Datasets, MultiHeadDatasets
from Model import LanguageModel, MultiHeadPsuedoLanguageModel, MultiHeadContrastiveLanguageModel, MultiHeadLanguageModel
import numpy as np
import random
import torch
import argparse

N_CLASSES = {
    'kim': 3,
    'acl': 6,
    'scicite': 3,
    'scicite_005': 3,
    'scicite_010': 3,
    'scicite_020': 3,
    'scicite_050': 3,
    'scicite_080': 3,
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
        args.class_definition,
        pl=True
    )
    if len(data_filenames) > 1:
        aux_data, aux_label_map = create_single_data_object(
            data_filenames[1], args.class_definition, split='train', pl=True
        )

    model = MultiHeadLanguageModel(
        modelname=modelname,
        device=args.device,
        readout=args.readout,
        num_classes=[N_CLASSES[datasets[0]]]
    ).to(args.device)

    if not args.inference_only:
        finetuner = MultiHeadTrainer(
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
        finetuner = MultiHeadTrainer(
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
            psuedo_labeled_dataset = aux_data.pseudo_label(aux_preds, threshold=0.9)
            print('Number of psuedo-labeled data: {}/{}'.format(len(psuedo_labeled_dataset), len(aux_data)))
        elif args.pl == 'pls':
            aux_data.update_label_with_selection(aux_preds)
        else:
            raise NotImplementedError
        train_datasets = Datasets([train_data, psuedo_labeled_dataset])

        model = MultiHeadLanguageModel(
            modelname=modelname,
            device=args.device,
            readout=args.readout,
            num_classes=[N_CLASSES[datasets[0]]] # [N_CLASSES[ds] for ds in datasets]
        ).to(args.device)

        finetuner = MultiHeadTrainer(
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

def main_mtl(args):
    datasets = args.dataset.split('-')
    lambdas = [float(l) for l in args.lambdas.split('-')]

    if lambdas[0] != 1:
        lambdas[0] = 1.
        print('The first lambda is set to 1.')
    assert len(datasets) == len(lambdas), "The size of lambdas should be the same as the number of datasets."

    data_filenames = [os.path.join(args.data_dir, ds+'.tsv') for ds in datasets]

    if args.lm == 'scibert':
        modelname = 'allenai/scibert_scivocab_uncased'
    elif args.lm == 'bert':
        modelname = 'bert-base-uncased'
    else:
        modelname = args.lm

    train_data, val_data, test_data, model_label_map = create_data_channels(
        data_filenames[0],
        args.class_definition,
        lmbd=lambdas[0]
    )
    train_datasets_list = [train_data]
    if len(data_filenames) > 1:
        for i, data_filename in enumerate(data_filenames[1:]):
            aux_data, aux_label_map = create_single_data_object(
                data_filename, args.class_definition, split='train', lmbd=lambdas[i+1]
            )
            train_datasets_list.append(aux_data)
    train_datasets = MultiHeadDatasets(train_datasets_list, batch_size_factor=args.batch_size_factor)
    if train_datasets.adjusted_batch_size_factor > 1:
        args.batch_size = int(args.batch_size * train_datasets.adjusted_batch_size_factor)
        print('Adjusting the training batch size to {}.'.format(args.batch_size))

    model = MultiHeadLanguageModel(
        modelname=modelname,
        device=args.device,
        readout=args.readout,
        num_classes=[N_CLASSES[ds] for ds in datasets]
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
        print('Evaluating end of training checkpoint.')
        preds = finetuner.test()
        finetuner.load_model()
        print('Evaluating best val checkpoint.')
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
    print('Lambdas: {}'.format(lambdas))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--lambdas', required=True)
    parser.add_argument('--data_dir', default='Data/', type=str)
    parser.add_argument('--workspace', default='Workspaces/Test', type=str)
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
    parser.add_argument('--pl', default='pl', type=str)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--batch_size_factor', default=2, type=int)
    parser.add_argument('--readout', default='ch', type=str)

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
    # main_cl(args)
    main_mtl(args)
