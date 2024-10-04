import os
import sys
import torch
from utils import *
import torch.nn as nn
from common import losses
from common.utils import *
import torch.distributed as dist
from models import *
from datasets import *
from shutil import copyfile
from datasets.factory import get_dataloader
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
torch.autograd.set_detect_anomaly(True)
from engine_finetune import train_one_epoch, test_one_epoch
args = get_params()
setup(args)
init_exam_dir(args)
###########################
# main logic for training #
###########################
def main():
    # use distributed training with nccl backend 
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl', init_method="env://")
    torch.cuda.set_device(args.local_rank)
    args.world_size = dist.get_world_size()
    # set logger
    logger = get_logger(str(args.local_rank), console=args.local_rank==0, log_path=os.path.join(args.exam_dir, f'train_{args.local_rank}.log'))
    train_dataloader = get_dataloader(args, 'train')
    test_dataloader = get_dataloader(args, 'test')
    args.model.params.local_rank = args.local_rank
    model = eval(args.model.name)(**args.model.params)
    if args.local_rank == 0:
        file_name = os.path.join(args.exam_dir, 'Model.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(str(model))
            opt_file.write('\n')
        config_file = args.config
        config_file_name = os.path.basename(config_file)
        copyfile(config_file, os.path.join(args.exam_dir, config_file_name))
    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    criterion = losses.__dict__[args.loss.name](**(args.loss.params if getattr(args.loss, "params", None) else {})).cuda(args.local_rank)
    # set optimizer
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args.optimizer.params))
    print(optimizer)
    global_step = 1
    start_epoch = 1
    # resume model for a given checkpoint file
    if args.model.resume:
        logger.info(f'resume from {args.model.resume}')
        checkpoint = torch.load(args.model.resume, map_location='cpu')
        if 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
            if (not getattr(args.model, 'only_resume_model', False)):
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'global_step' in checkpoint:
                    global_step = checkpoint['global_step']
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
        else:
            sd = checkpoint
        model.load_state_dict(sd, strict=True)

    lr_scheduler, num_epochs = create_scheduler(args.scheduler, optimizer)
    args.train.max_epoches = num_epochs
    if lr_scheduler is not None and start_epoch > 1:
        lr_scheduler.step(start_epoch-1)
    # Training loops
    for epoch in range(start_epoch, num_epochs+1):
        train_dataloader.sampler.set_epoch(epoch)
        train_one_epoch(train_dataloader, model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler)
        global_step += len(train_dataloader)
        test_one_epoch(test_dataloader, model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler)

        
if __name__ == '__main__':
    main()
