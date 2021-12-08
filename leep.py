import os
# change the default cache dir so that huggingface won't take the cse space.
os.environ['TRANSFORMERS_CACHE'] = '/export/scratch/zeren/KimNLP/HuggingfaceCache/'

import argparse
import torch
import torch.nn.functional as F

import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score

from Model import LanguageModel, EarlyFuseClassifier, LateFuseClassifier, MLPClassifier, MultiHeadEarlyFuseClassifier, MultiHeadLateFuseClassifier
from Model.layers import DenseLayer
from data import EmbeddedDataset, MultiHeadDatasets, SingleHeadDatasets, SingleHeadEmbeddedDatasets, Dataset
from train import Trainer, PreTrainer, MultiHeadTrainer, SingleHeadTrainer, SingleHeadPreTrainer
from utils import save_args, select_activation_fn

def create_data_channels(filename, modelname, fuse_type, max_length):
    data = pd.read_csv(filename, sep='\t')

    print('Number of data instance: {}'.format(data.shape[0]))

    # map labels to ids
    unique_labels = data['label'].unique()
    label2id = {lb: i for i, lb in enumerate(unique_labels)}
    print(label2id)

    data['label'] = data['label'].apply(
        lambda x: label2id[x])

    data_train = data[data['split'] == 'train'].reset_index()
    data_val = data[data['split'] == 'val'].reset_index()
    data_test = data[data['split'] == 'test'].reset_index()

    train_data = Dataset(
        data_train,
        modelname=modelname,
        fuse_type=fuse_type,
        max_length=max_length
    )
    val_data = Dataset(
        data_val,
        modelname=modelname,
        fuse_type=fuse_type,
        max_length=max_length
    )
    test_data = Dataset(
        data_test,
        modelname=modelname,
        fuse_type=fuse_type,
        max_length=max_length
    )

    return train_data, val_data, test_data

class Trainer(object):
    def __init__(self, model, dataset, args):
        self.device = args.device
        self.batch_size = args.batch_size_finetune
        self.num_epochs = args.num_epochs_finetune

        self.workspace = args.workspace
        self.model = model

        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    def compute_leep(self, labels, logits):
        n = len(labels)

        n_rows = len(np.unique(labels))
        n_cols = logits.shape[1]

        P = np.zeros((n_rows, n_cols), dtype=np.float32)
        for i in range(n):
            row_idx = labels[i]
            P[row_idx] += logits[i]

        Pz = P.sum(axis=0, keepdims=True)
        Pyz = P / Pz

        target_logits = logits @ Pyz.T
        preds = target_logits.argmax(axis=1)
        leep = np.log(target_logits[np.arange(n), labels]).mean()
        
        accuracy = accuracy_score(y_true=labels, y_pred=preds)
        f1 = f1_score(y_true=labels, y_pred=preds, average='macro')

        print(Pyz)
        print(leep, accuracy, f1)
        return leep

    def eval(self):
        self.model.eval()

        preds, labels = list(), list()
        with torch.no_grad():
            for fused_context, label in self.dataloader:
                preds.append(self.model(
                    fused_context).detach().cpu())
                labels.append(label)

            preds = F.softmax(torch.cat(preds, dim=0), dim=1).numpy()
            labels = torch.cat(labels, dim=0).numpy()

            return self.compute_leep(labels, preds)

    def load_model(self):
        model_filename = os.path.join(self.workspace, 'best_model.pt')
        self.model.load_state_dict(torch.load(model_filename))

def run(args, data_data, model_data, seed):
    data_filename = os.path.join(args.data_dir, data_data+'.tsv')
    modelname = 'allenai/scibert_scivocab_uncased'
    hidden_dims = list(map(int, args.hidden_dims.split(',')))
    args.workspace = 'Workspaces/main{}_bruteforce_ch_seed{}_l20_10epochs/'.format(model_data, seed)

    token_train_data, token_val_data, token_test_data = create_data_channels(
        data_filename,
        modelname,
        fuse_type=args.fuse_type,
        max_length=args.max_length
    )

    lm_model = LanguageModel(
        modelname=modelname,
        device=args.device,
        rawtext_readout=args.rawtext_readout,
        context_readout=args.context_readout,
        intra_context_pooling=args.intra_context_pooling
    ).to(args.device)
    
    n_classes = 6 if model_data == 'acl' else 3

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
    
    finetuner = Trainer(
        model,
        token_train_data,
        args
    )
    finetuner.load_model()
    return finetuner.eval()

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
    parser.add_argument('--n_classes', default=3, type=int)

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
        print('Wrong workspace.')

    # run(args, data_data='kim', model_data='scicite', seed=42)

    for model_data in ['kim', 'acl', 'scicite']:
        for data_data in ['kim', 'acl', 'scicite']:
            if model_data != data_data:
                key = 'model-' + model_data + '-data-' + data_data
                for seed in [42, 3515, 4520]:
                    print(key)
                    run(args, data_data=data_data, model_data=model_data, seed=seed)

    # for model_data in ['kim', 'acl', 'scicite']:
    #     for data_data in ['kim', 'acl', 'scicite']:
    #         if model_data != data_data:
    #             key = 'model-' + model_data + '-data-' + data_data
    #             matrices = list()
    #             for seed in [42, 3515, 4520]:
    #                 confusion_matrix = run(args, data_data=data_data, model_data=model_data, seed=seed)
    #                 matrices.append(confusion_matrix)
    #             mat = np.stack(matrices).mean(axis=0)
    #             row_mat = mat / mat.sum(axis=1, keepdims=True)
    #             col_mat = mat / mat.sum(axis=0, keepdims=True)
    #             n_rows, n_cols = row_mat.shape

    #             base_row_entropy = entropy(mat.sum(axis=0), base=n_cols)
    #             base_col_entropy = entropy(mat.sum(axis=1), base=n_rows)

    #             row_weight = mat.sum(axis=1) / mat.sum()
    #             col_weight = mat.sum(axis=0) / mat.sum()
    #             rownorm_entropy = base_row_entropy - entropy(row_mat, base=n_cols, axis=1).dot(row_weight)#.mean()
    #             colnorm_entropy = base_col_entropy - entropy(col_mat, base=n_rows, axis=0).dot(col_weight)#.mean()
    #             mean_entropy = (rownorm_entropy + colnorm_entropy) / 2
    #             print(key, rownorm_entropy, colnorm_entropy, mean_entropy)
    
    # mean_confusion_matrices = dict()
    # for key, mat in confusion_matrices.items():
    #     mat = np.stack(mat).mean(axis=0)
    #     mean_confusion_matrices[key] = mat
    
    # rownorm_entropy = dict()
    # colnorm_entropy = dict()
    # full_entropy = dict()
    # for key, mat in mean_confusion_matrices.items():
    #     row_mat = mat / mat.sum(axis=1, keepdims=True)
    #     col_mat = mat / mat.sum(axis=0, keepdims=True)

    #     n_rows, n_cols = row_mat.shape
    #     rownorm_entropy[key] = entropy(row_mat, base=n_cols, axis=1).mean()
    #     colnorm_entropy[key] = entropy(col_mat, base=n_rows, axis=0).mean()
    #     full_entropy[key] = (rownorm_entropy[key] + colnorm_entropy[key]) / 2
    # print(rownorm_entropy)
    # print(colnorm_entropy)
    # print(full_entropy)