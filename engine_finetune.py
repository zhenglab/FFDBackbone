import os
import sys
import time
import torch
from utils import *
from common.utils import *
import glob as glob_lib
from models import *
from datasets import *
from tqdm import tqdm
from collections import OrderedDict
from common.utils import map_util
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))
torch.autograd.set_detect_anomaly(True)


def train_one_epoch(dataloader, model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler):
    epoch_size = len(dataloader)
    acces = AverageMeter('Acc', ':.4f')
    real_acces = AverageMeter('RealAcc', ':.4f')
    fake_acces = AverageMeter('FakeACC', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    batch_time = AverageMeter('Time', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, real_acces, fake_acces, losses, data_time, batch_time])
    model.train(True)
    end = time.time()
    num_updates = (epoch-1) * len(dataloader)
    for idx, datas in enumerate(dataloader):
        data_time.update(time.time() - end)
        images = datas['images']
        labels = datas['labels']
        video_path = datas['video_path']
        sampled_frame_idxs = datas['sampled_frame_idxs']
        images = images.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank)
        outputs = model(images)
        labels_r = labels.unsqueeze(1).repeat(1,outputs.size(1)).flatten(0,1)
        outputs_r2 = outputs.flatten(0,1)
        loss = criterion(outputs_r2, labels_r)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # check grad
        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad==True:
                print('nograd:', name)
        optimizer.step()
        # compute accuracy metrics
        acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outputs, labels)
        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates, metric=acces.avg)
        # update statistical meters 
        acces.update(acc, images.size(0))
        real_acces.update(real_acc, real_cnt)
        fake_acces.update(fake_acc, fake_cnt)
        losses.update(loss.item(), images.size(0))

        # log training metrics at a certain frequency
        if (idx + 1) % args.train.print_info_step_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            cur_lr = sum(lrl) / len(lrl)
            logger.info(f'TRAIN Epoch-{epoch}, Step-{global_step}: {progress.display(idx+1)}  lr: {cur_lr:.7f}')
        if args.local_rank == 0:
            if idx==0 or idx== epoch_size-1 or idx%500==0:
                split = int(images.size(0)//2)
                img_save_dir = os.path.join(os.path.join(args.exam_dir, 'train_img'),)
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                input_real = images[labels == 0]
                input_fake = images[labels == 1]
                outputs_real = outputs[labels == 0].squeeze(1)
                outputs_fake = outputs[labels == 1].squeeze(1)                    
                images = map_util.save_real_data_images_withrebuild(video_path,outputs_real, outputs_fake, input_real, input_fake, input_real, input_fake, labels, args.transform_params.mean, args.transform_params.std, sampled_frame_idxs,img_save_dir, str(epoch)+"-"+str(idx), return_np=False)
        global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()


def train_dual_one_epoch(dataloader, model, criterion_b, criterion_e, optimizer, epoch, global_step, args, logger,lr_scheduler, total_epoch):
    epoch_size = len(dataloader)
    acces = AverageMeter('Acc', ':.4f')
    real_acces = AverageMeter('RealAcc', ':.4f')
    fake_acces = AverageMeter('FakeACC', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    losses_c = AverageMeter('CELoss', ':.4f')
    losses_e1 = AverageMeter('EDULoss1', ':.4f')
    losses_e2 = AverageMeter('EDULoss2', ':.4f')
    losses_p = AverageMeter('PLoss', ':.4f')

    data_time = AverageMeter('Data', ':.4f')
    batch_time = AverageMeter('Time', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, real_acces, fake_acces, losses, losses_c, losses_e1, losses_e2, losses_p, data_time, batch_time])
    model.train(True)
    end = time.time()
    num_updates = (epoch-1) * len(dataloader)
    for idx, datas in enumerate(dataloader):
        data_time.update(time.time() - end)
        images = datas['images']
        labels = datas['labels']
        video_path = datas['video_path']
        sampled_frame_idxs = datas['sampled_frame_idxs']
        images = images.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank)
        outputs, outputs_m, outputs_a, ploss = model(images)
        labels_r = labels.unsqueeze(1).repeat(1,outputs.size(1)).flatten(0,1)
        outputs_r = outputs.flatten(0,1)
        outputs_r2 = outputs_m.flatten(0,1)
        outputs_r3 = outputs_a.flatten(0,1)        
        loss_b = criterion_b(outputs_r, labels_r)

        k = dict()
        k['epoch'] = epoch
        k['total_epoch'] = total_epoch
        losses_edl = criterion_e(outputs_r2, labels_r, **k)
        log_vars = OrderedDict()
        for loss_name, loss_value in losses_edl.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        loss_e1 = sum(_value for _key, _value in log_vars.items()
                if 'loss' in _key)

        losses_edl2 = criterion_e(outputs_r3, labels_r, **k)
        log_vars = OrderedDict()
        for loss_name, loss_value in losses_edl2.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        loss_e2 = sum(_value for _key, _value in log_vars.items()
                if 'loss' in _key)

        loss = loss_b + loss_e1 + loss_e2 + ploss

        # backward
        optimizer.zero_grad()
        loss.backward()
        # check grad
        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad==True:
                print('nograd:', name)
        optimizer.step()
        # compute accuracy metrics
        acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outputs, labels)
        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates, metric=acces.avg)
        # update statistical meters 
        acces.update(acc, images.size(0))
        real_acces.update(real_acc, real_cnt)
        fake_acces.update(fake_acc, fake_cnt)
        losses.update(loss.item(), images.size(0))
        losses_c.update(loss.item(), images.size(0))
        losses_e1.update(loss.item(), images.size(0))
        losses_e2.update(loss.item(), images.size(0))
        losses_p.update(loss.item(), images.size(0))



        # log training metrics at a certain frequency
        if (idx + 1) % args.train.print_info_step_freq == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            cur_lr = sum(lrl) / len(lrl)
            logger.info(f'TRAIN Epoch-{epoch}, Step-{global_step}: {progress.display(idx+1)}  lr: {cur_lr:.7f}')
        if args.local_rank == 0:
            if idx==0 or idx== epoch_size-1 or idx%500==0:
                split = int(images.size(0)//2)
                img_save_dir = os.path.join(os.path.join(args.exam_dir, 'train_img'),)
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                input_real = images[labels == 0]
                input_fake = images[labels == 1]
                outputs_real = outputs[labels == 0].squeeze(1)
                outputs_fake = outputs[labels == 1].squeeze(1)                    
                images = map_util.save_real_data_images_withrebuild(video_path,outputs_real, outputs_fake, input_real, input_fake, input_real, input_fake, labels, args.transform_params.mean, args.transform_params.std, sampled_frame_idxs,img_save_dir, str(epoch)+"-"+str(idx), return_np=False)
        global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()
        
def test_one_epoch(dataloader, model, criterion, optimizer, epoch, global_step, args, logger,lr_scheduler):
    model.eval()
    y_outputs, y_labels = [], []
    loss_t = 0.
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(dataloader)):
            images = datas['images']
            labels = datas['labels']
            images = images.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank)
            outputs = model(images)

            labels_r = labels.unsqueeze(1).repeat(1,outputs.size(1)).flatten(0,1)
            outputs_r = outputs.flatten(0,1)
            loss = criterion(outputs_r, labels_r)

            loss_t += loss * labels.size(0)
            y_outputs.extend(outputs)
            y_labels.extend(labels)
    # gather outputs from all distributed nodes
    gather_y_outputs = gather_tensor(y_outputs, args.world_size, to_numpy=False)
    gather_y_labels  = gather_tensor(y_labels, args.world_size, to_numpy=False)
    # compute accuracy metrics
    acc, real_acc, fake_acc, _, _ = compute_metrics(gather_y_outputs, gather_y_labels)
    weight_acc = 0.
    if real_acc and fake_acc:
        weight_acc = 2 / (1 / real_acc + 1 / fake_acc)
    # compute loss
    loss_t = reduce_tensor(loss_t, mean=False)
    loss = (loss_t / len(dataloader.dataset)).item()
    # log test metrics and save the model into the checkpoint file
    lr = optimizer.param_groups[0]['lr']
    logger.info('[TEST] EPOCH-{} Step-{} ACC: {:.4f} RealACC: {:.4f} FakeACC: {:.4f} Loss: {:.5f} lr: {:.7f}'.format(epoch, global_step, acc, real_acc, fake_acc, loss, lr))
    if args.local_rank == 0:
        last_epoch_max_acc = args.train.last_epoch_max_acc
        current_epoch_acc = float(format(acc, '.4f'))
        test_metrics = {
            'test_acc': acc,
            'test_weight_acc': weight_acc,
            'test_real_acc': real_acc,
            'test_fake_acc': fake_acc,
            'test_loss': loss,
            'lr': lr,
            "epoch": epoch
        }
        checkpoint = OrderedDict()
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['global_step'] = global_step
        checkpoint['metrics'] = test_metrics
        checkpoint['args'] = args
        if epoch == args.train.max_epoches:
            checkpoint_save_name = "Final-Epoch-{}-Step-{}-ACC-{:.4f}-RealACC-{:.4f}-FakeACC-{:.4f}-Loss-{:.5f}-LR-{:.6g}.tar".format(epoch, global_step, acc, real_acc, fake_acc, loss, lr)
            checkpoint_save_dir = os.path.join(os.path.join(args.exam_dir, 'ckpt'), checkpoint_save_name)
            torch.save(checkpoint, checkpoint_save_dir)
        elif current_epoch_acc >= last_epoch_max_acc:
            checkpoint_save_name = "Epoch-{}-Step-{}-ACC-{:.4f}-RealACC-{:.4f}-FakeACC-{:.4f}-Loss-{:.5f}-LR-{:.6g}.tar".format(epoch, global_step, acc, real_acc, fake_acc, loss, lr)
            checkpoint_save_dir = os.path.join(os.path.join(args.exam_dir, 'ckpt'), checkpoint_save_name)
            torch.save(checkpoint, checkpoint_save_dir)
            ckpt_path = os.path.join(args.exam_dir, 'ckpt')
            ckpt_files = glob_lib.glob(ckpt_path+"/Epoch*-ACC-{:.4f}-RealACC-*".format(last_epoch_max_acc))
            for ckpt_file in ckpt_files:
                if ckpt_file.split("/")[-1] != checkpoint_save_name:
                    os.remove(ckpt_file)
            args.train.last_epoch_max_acc = current_epoch_acc
    if lr_scheduler is not None:
        lr_scheduler.step(epoch-1)

def test_dual_one_epoch(dataloader, model, criterion_b, criterion_e, optimizer, epoch, global_step, args, logger,lr_scheduler, total_epoch):
    model.eval()
    y_outputs, y_labels = [], []
    loss_t = 0.
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(dataloader)):
            images = datas['images']
            labels = datas['labels']
            images = images.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank)
            outputs = model(images)

            outputs, outputs_m, outputs_a, ploss = model(images)
            labels_r = labels.unsqueeze(1).repeat(1,outputs.size(1)).flatten(0,1)
            outputs_r = outputs.flatten(0,1)
            outputs_r2 = outputs_m.flatten(0,1)
            outputs_r3 = outputs_a.flatten(0,1)        
            loss_b = criterion_b(outputs_r, labels_r)
            k = dict()
            k['epoch'] = epoch
            k['total_epoch'] = total_epoch
            losses_edl = criterion_e(outputs_r2, labels_r, **k)
            log_vars = OrderedDict()
            for loss_name, loss_value in losses_edl.items():
                if isinstance(loss_value, torch.Tensor):
                    log_vars[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')
            loss_e1 = sum(_value for _key, _value in log_vars.items()
                    if 'loss' in _key)

            losses_edl2 = criterion_e(outputs_r3, labels_r, **k)
            log_vars = OrderedDict()
            for loss_name, loss_value in losses_edl2.items():
                if isinstance(loss_value, torch.Tensor):
                    log_vars[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')
            loss_e2 = sum(_value for _key, _value in log_vars.items()
                    if 'loss' in _key)
            loss = loss_b + loss_e1 + loss_e2 + ploss

            loss_t += loss * labels.size(0)
            y_outputs.extend(outputs)
            y_labels.extend(labels)
    # gather outputs from all distributed nodes
    gather_y_outputs = gather_tensor(y_outputs, args.world_size, to_numpy=False)
    gather_y_labels  = gather_tensor(y_labels, args.world_size, to_numpy=False)
    # compute accuracy metrics
    acc, real_acc, fake_acc, _, _ = compute_metrics(gather_y_outputs, gather_y_labels)
    weight_acc = 0.
    if real_acc and fake_acc:
        weight_acc = 2 / (1 / real_acc + 1 / fake_acc)
    # compute loss
    loss_t = reduce_tensor(loss_t, mean=False)
    loss = (loss_t / len(dataloader.dataset)).item()
    # log test metrics and save the model into the checkpoint file
    lr = optimizer.param_groups[0]['lr']
    logger.info('[TEST] EPOCH-{} Step-{} ACC: {:.4f} RealACC: {:.4f} FakeACC: {:.4f} Loss: {:.5f} lr: {:.7f}'.format(epoch, global_step, acc, real_acc, fake_acc, loss, lr))
    if args.local_rank == 0:
        last_epoch_max_acc = args.train.last_epoch_max_acc
        current_epoch_acc = float(format(acc, '.4f'))
        test_metrics = {
            'test_acc': acc,
            'test_weight_acc': weight_acc,
            'test_real_acc': real_acc,
            'test_fake_acc': fake_acc,
            'test_loss': loss,
            'lr': lr,
            "epoch": epoch
        }
        checkpoint = OrderedDict()
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['global_step'] = global_step
        checkpoint['metrics'] = test_metrics
        checkpoint['args'] = args
        if epoch == args.train.max_epoches:
            checkpoint_save_name = "Epoch-Final-{}-Step-{}-ACC-{:.4f}-RealACC-{:.4f}-FakeACC-{:.4f}-Loss-{:.5f}-LR-{:.6g}.tar".format(epoch, global_step, acc, real_acc, fake_acc, loss, lr)
            checkpoint_save_dir = os.path.join(os.path.join(args.exam_dir, 'ckpt'), checkpoint_save_name)
            mainbranch_weight_dir = os.path.join(os.path.join(args.exam_dir, 'ckpt'), 'Final-mainbranch.tar')
            torch.save(checkpoint, checkpoint_save_dir)
            convert_weight(checkpoint, mainbranch_weight_dir)

        elif current_epoch_acc >= last_epoch_max_acc:
            checkpoint_save_name = "Epoch-{}-Step-{}-ACC-{:.4f}-RealACC-{:.4f}-FakeACC-{:.4f}-Loss-{:.5f}-LR-{:.6g}.tar".format(epoch, global_step, acc, real_acc, fake_acc, loss, lr)
            checkpoint_save_dir = os.path.join(os.path.join(args.exam_dir, 'ckpt'), checkpoint_save_name)
            torch.save(checkpoint, checkpoint_save_dir)
            ckpt_path = os.path.join(args.exam_dir, 'ckpt')
            ckpt_files = glob_lib.glob(ckpt_path+"/Epoch*-ACC-{:.4f}-RealACC-*".format(last_epoch_max_acc))
            for ckpt_file in ckpt_files:
                if ckpt_file.split("/")[-1] != checkpoint_save_name:
                    os.remove(ckpt_file)
            args.train.last_epoch_max_acc = current_epoch_acc
        

    if lr_scheduler is not None:
        lr_scheduler.step(epoch-1)
