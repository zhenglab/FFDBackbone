import os
import random
import wandb
import shutil
import time
import datetime
import warnings
import torch
import numpy as np
from torch import Tensor
from typing import Optional, List
from timm.models.layers import DropPath, trunc_normal_
import torch.nn as nn

def set_seed(SEED):
    """This function set the random seed for the training process
    
    Args:
        SEED (int): the random seed
    """
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True


def setup(cfg):
    if getattr(cfg, 'torch_home', None):
        os.environ['TORCH_HOME'] = cfg.torch_home
    warnings.filterwarnings("ignore")
    seed = cfg.seed
    set_seed(seed)


def init_exam_dir(cfg):
    if cfg.local_rank == 0:
        if not os.path.exists(cfg.exam_dir):
            os.makedirs(cfg.exam_dir)
        ckpt_dir = os.path.join(cfg.exam_dir, 'ckpt')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        train_img_dir = os.path.join(cfg.exam_dir, 'train_img')
        if not os.path.exists(train_img_dir):
            os.makedirs(train_img_dir)


def init_wandb_workspace(cfg):
    """This function initializes the wandb workspace
    """
    if cfg.wandb.name is None:
        cfg.wandb.name = cfg.config.split('/')[-1].replace('.yaml', '')
    wandb.init(**cfg.wandb)
    allow_val_change = False if cfg.wandb.resume is None else True
    wandb.config.update(cfg, allow_val_change)
    wandb.save(cfg.config)
    if cfg.debug or wandb.run.dir == '/tmp':
        cfg.exam_dir = 'wandb/debug'
        if os.path.exists(cfg.exam_dir):
            shutil.rmtree(cfg.exam_dir)
        os.makedirs(cfg.exam_dir, exist_ok=True)
    else:
        cfg.exam_dir = os.path.dirname(wandb.run.dir)
    os.makedirs(os.path.join(cfg.exam_dir, 'ckpts'), exist_ok=True)
    return cfg


def save_test_results(img_paths, y_preds, y_trues, filename='results.log'):
    assert len(y_trues) == len(y_preds) == len(img_paths)

    with open(filename, 'w') as f:
        for i in range(len(img_paths)):
            print(img_paths[i], end=' ', file=f)
            print(y_preds[i], file=f)
            print(y_trues[i], end=' ', file=f)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask
    
    def flatten(self):
        return  NestedTensor(self.tensors.flatten(0,1), self.mask)

    def __repr__(self):
        return str(self.tensors)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

