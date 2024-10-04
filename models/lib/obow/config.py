import argparse
import os
import random
import warnings
import pathlib
import yaml

import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn
import torch.distributed
import torch.multiprocessing

import models.lib.obow.feature_extractor
import models.lib.obow.classification
import models.lib.obow.utils
import models.lib.obow.datasets
from models.lib.obow import project_root


def get_arguments():
    """ Parse input arguments. """
    default_dst_dir = str(pathlib.Path(project_root) / "experiments")
    parser = argparse.ArgumentParser(
        description='Linear classification evaluation using a pre-trained with '
                    'OBoW feature extractor (from the student network).')
    parser.add_argument(
        '-j', '--workers', default=4, type=int,
        help='Number of data loading workers (default: 4)')
    parser.add_argument(
        '-b', '--batch-size', default=256, type=int,
        help='Mini-batch size (default: 256), this is the total '
             'batch size of all GPUs on the current node when '
             'using Data Parallel or Distributed Data Parallel.')
    parser.add_argument(
        '--start-epoch', default=0, type=int,
        help='Manual epoch number to start training in case of restart (default 0).'
             'If -1, then it stargs training from the last available checkpoint.')
    parser.add_argument(
        '-p', '--print-freq', default=200, type=int,
        help='Print frequency (default: 200)')
    parser.add_argument(
        '--world-size', default=1, type=int,
        help='Number of nodes for distributed training (default 1)')
    parser.add_argument(
        '--rank', default=0, type=int,
        help='Node rank for distributed training (default 0)')
    parser.add_argument(
        '--dist-url', default='tcp://127.0.0.1:4444', type=str,
        help='Url used to set up distributed training '
             '(default tcp://127.0.0.1:4444)')
    parser.add_argument(
        '--dist-backend', default='nccl', type=str,
        help='Distributed backend (default nccl)')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='Seed for initializing training (default None)')
    parser.add_argument(
        '--gpu', default=None, type=int,
        help='GPU id to use (default: None). If None it will try to use all '
             'the available GPUs.')
    parser.add_argument(
        '--multiprocessing-distributed', action='store_true',
        help='Use multi-processing distributed training to launch '
             'N processes per node, which has N GPUs. This is the '
             'fastest way to use PyTorch for either single node or '
             'multi node data parallel training')
    parser.add_argument(
        '--dst-dir', default=default_dst_dir, type=str,
        help='Base directory where the experiments data (i.e, checkpoints) of '
             'the pre-trained OBoW model is stored (default: '
             f'{default_dst_dir}). The final directory path would be: '
             '"dst-dir / config", where config is the name of the config file.')
    parser.add_argument(
        '--config', type=str, default="models/lib/obow/ResNet50_OBoW_full",
        help='Config file that was used for training the OBoW model.')
    parser.add_argument(
        '--name', default='semi_supervised', type=str,
        help='The directory name of the experiment. The final directory '
             'where the model and logs would be stored is: '
             '"dst-dir / config / name", where dst-dir is the base directory '
             'for the OBoW model and config is the name of the config file '
             'that was used for training the model.')
    parser.add_argument(
        '--evaluate', action='store_true', help='Evaluate the model.')
    parser.add_argument(
        '--dataset', default='', type=str,
        help='Dataset that will be used for the linear classification '
             'evaluation. Supported options: ImageNet, Places205.')
    parser.add_argument(
        '--data-dir', type=str, default='',
        help='Directory path to the ImageNet or Places205 datasets.')
    parser.add_argument('--subset', default=-1, type=int,
        help='The number of images per class  that they would be use for '
             'training (default -1). If -1, then all the availabe images are '
             'used.')
    parser.add_argument(
        '-n', '--batch-norm', action='store_true',
        help='Use batch normalization (without affine transform) on the linear '
             'classifier. By default this option is deactivated.')
    parser.add_argument('--epochs', default=100, type=int,
        help='Number of total epochs to run (default 100).')
    parser.add_argument('--lr', '--learning-rate', default=10.0, type=float,
        help='Initial learning rate (default 10.0)', dest='lr')
    parser.add_argument('--cos-schedule', action='store_true',
        help='If True then a cosine learning rate schedule is used. Otherwise '
             'a step-wise learning rate schedule is used. In this latter case, '
             'the schedule and lr-decay arguments must be specified.')
    parser.add_argument(
        '--schedule', default=[15, 30, 45,], nargs='*', type=int,
        help='Learning rate schedule (when to drop lr by a lr-decay ratio) '
             '(default: 15, 30, 45). This argument is only used in case of '
             'step-wise learning rate schedule (when the cos-schedule flag is '
             'not activated).')
    parser.add_argument(
        '--lr-decay', default=0.1, type=float,
        help='Learning rate decay step (default 0.1). This argument is only '
        'used in case of step-wise learning rate schedule (when the '
        'cos-schedule flag is not activated).' )
    parser.add_argument('--momentum', default=0.9, type=float,
        help='Momentum (default 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
        help='Weight decay (default: 0.)', dest='weight_decay')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument(
        '--precache', action='store_true',
        help='Precache features for the linear classifier. Those features are '
             'deleted after the end of training.')
    parser.add_argument(
        '--cache-dir', default='', type=str,
        help='destination directory for the precached features.')
    parser.add_argument(
        '--cache-5crop', action='store_true',
        help='Use five crops when precaching features (only for the train set).')
    parser.add_argument(
        '-c')
    parser.add_argument(
        '--local_rank')
    parser.add_argument("--method")
    parser.add_argument("--compression")
    parser.add_argument("--final_test.dataset.params.test_method")        
    args = parser.parse_args()
    args.feature_extractor_dir = pathlib.Path(args.dst_dir) / args.config
    os.makedirs(args.feature_extractor_dir, exist_ok=True)
    args.exp_dir = args.feature_extractor_dir /  args.name
    os.makedirs(args.exp_dir, exist_ok=True)

    # Load the configuration params of the experiment
    full_config_path = pathlib.Path(project_root) / "config" / (args.config + ".yaml")
    print(f"Loading experiment {full_config_path}")
    full_config_path = '/SSD0/guozonghui/project/FFD/ffd_vit2/models/lib/obow/ResNet50_OBoW_full.yaml'
    with open(full_config_path, "r") as f:
        args.exp_config = yaml.load(f, Loader=yaml.SafeLoader)

    print(f"Logs and/or checkpoints will be stored on {args.exp_dir}")

    if args.precache:
        if args.cache_dir == '':
            raise ValueError(
                'To precache the features (--precache argument) you need to '
                'specify with the --cache-dir argument the directory where the '
                'features will be stored.')
        cache_dir_name = f"{args.config}"
        args.cache_dir = pathlib.Path(args.cache_dir) / cache_dir_name
        os.makedirs(args.cache_dir, exist_ok=True)
        args.cache_dir = pathlib.Path(args.cache_dir) / "cache_features"
        os.makedirs(args.cache_dir, exist_ok=True)

    return args