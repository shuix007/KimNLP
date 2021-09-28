import os
import torch
import numpy as np

from tqdm import tqdm


def save_args(args, workspace):
    args_path = os.path.join(workspace, 'args.txt')
    with open(args_path, 'w') as f:
        f.write(str(args).replace(', ', ',\n'))