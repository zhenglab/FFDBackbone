import sys
import os
from omegaconf import OmegaConf
import torch.utils.data as data

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from common.data import create_base_transforms, create_base_dataloader,create_base_sbi_transforms

from .Image_dataset import *
from .Image_dataset_test import *

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_dataloader(args, split):
    """Set dataloader.

    Args:
        args (object): Args load from get_params function.
        split (str): One of ['train', 'test']
    """
    transform = create_base_transforms(args.transform_params, split=split)
    dataset_cfg = getattr(args, split).dataset
    dataset_params = OmegaConf.to_container(dataset_cfg.params, resolve=True)
    dataset_params['transform'] = transform
    _dataset = eval(dataset_cfg.name)(**dataset_params)
    _dataloader = create_base_dataloader(args, _dataset, split=split)
    return _dataloader



def get_final_dataloader(args, split):
    """Set dataloader.

    Args:
        args (object): Args load from get_params function.
        split (str): One of ['train', 'test']
    """
    transform = create_base_transforms(args.transform_params, split=split)

    dataset_cfg = getattr(args, 'final_test').dataset
    print('dataset:',dataset_cfg.params)
    dataset_params = OmegaConf.to_container(dataset_cfg.params, resolve=True)
    dataset_params['transform'] = transform
    _dataset = eval(dataset_cfg.name)(**dataset_params)
    _dataloader = create_base_dataloader(args, _dataset, split=split)
    return _dataloader

def get_final_image_dataloader(args, split):
    """Set dataloader.

    Args:
        args (object): Args load from get_params function.
        split (str): One of ['train', 'test']
    """
    transform = create_base_sbi_transforms(args.transform_params, split=split)

    dataset_cfg = getattr(args, 'final_test').dataset
    dataset_params = OmegaConf.to_container(dataset_cfg.params, resolve=True)
    dataset_params['transform'] = transform

    _dataset = eval(dataset_cfg.name)(**dataset_params)

    _dataloader = create_base_dataloader(args, _dataset, split=split)

    return _dataloader

def get_sbi_dataloader(args, split):
    """Set dataloader.

    Args:
        args (object): Args load from get_params function.
        split (str): One of ['train', 'test']
    """
    transform = create_base_transforms(args.transform_params, split=split)
    dataset_cfg = getattr(args, split).dataset
    dataset_params = OmegaConf.to_container(dataset_cfg.params, resolve=True)
    dataset_params['transform'] = transform
    _dataset = eval(dataset_cfg.name)(**dataset_params)
    _dataloader = create_sbi_dataloader(args, _dataset, split=split)
    return _dataloader

def create_sbi_dataloader(args, dataset, split):
    """Base data loader

    Args:
        args: Dataset config args
        split (string): Load "train", "val" or "test"

    Returns:
        [dataloader]: Corresponding Dataloader
    """
    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
    shuffle = True if sampler is None and split == 'train' else False
    batch_size = getattr(args, split).batch_size
    num_workers = args.num_workers if 'num_workers' in args else 8
    drop_last = False if split == 'test' else True
    dataloader = DataLoaderX(dataset,
                                 batch_size=batch_size//2,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=drop_last,
                                 collate_fn=dataset.collate_fn,)
    return dataloader


