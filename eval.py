import os
import sys
from omegaconf import OmegaConf
from tqdm import tqdm
from shutil import copyfile
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd
from models import *
from datasets import *
from datasets.factory import get_final_dataloader
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))
from common.utils import *
from utils import *
import glob
from common.evaluations import video_evaluation
args = get_params()
setup(args)
###########################
# main logic for test #
###########################

def main():
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl', init_method="env://")
    torch.cuda.set_device(args.local_rank)
    args.world_size = dist.get_world_size()
    model = eval(args.model.backbone)(**args.model.params)
    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    result_dir = os.path.dirname(args.config)
    print(result_dir)
    ckpts = sorted(glob.glob(result_dir+'/ckpt/Final*.tar'))
    i = 0
    for ckpt in ckpts:
        if i ==len(ckpts)-1:
            args.model.resume = ckpt
            ckpt_load_path = args.model.resume
            print(f'ckpt_load_path: {ckpt_load_path}')
            if not ckpt_load_path:
                raise ValueError("You must load a checkpoint by specifying the `model.resume` argument.")
            checkpoint = torch.load(ckpt_load_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                sd = checkpoint['state_dict']
            else:
                sd = checkpoint
            msg = model.load_state_dict(sd,strict=False)
            print('eval_load', msg)
            methods = ['FaceForensics_c23', 'Celeb-DF', 'DFDC', 'FFIW']
            for method in methods:
                args.final_test.dataset.params.method = method
                args.final_test.dataset.params.split = 'test'
                test_dataloader = get_final_dataloader(args, 'test')
                # main test function
                test(test_dataloader, model, args)
        i +=1

def test(dataloader, model, args):
    model.eval()
    test_label = 'test'
    y_outputs, y_labels, y_idxes = [], [], []
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(dataloader)):
            images = datas['images']
            labels = datas['labels']
            video_path = datas['video_path']
            idxes = datas['index']
            images = images.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank)
            idxes = idxes.cuda(args.local_rank)
            output = model(images)
            outputs=torch.nn.functional.softmax(output, dim=2)[:,:,1]
            outputs = torch.mean(outputs, dim=1)
            if len(labels.shape) > 1:
                labels,_ = torch.max(labels,1)
            y_outputs.extend(outputs)
            y_labels.extend(labels)
            y_idxes.extend(idxes)
    gather_y_outputs = gather_tensor(y_outputs, args.world_size, to_numpy=False)
    gather_y_labels  = gather_tensor(y_labels, args.world_size, to_numpy=False)
    gather_y_idxes  = gather_tensor(y_idxes, args.world_size, to_numpy=False)
    test_result_list = []
    for i, idx in enumerate(gather_y_idxes):
        video_name = dataloader.dataset.all_list[idx][0][0]
        video_name_tmp = video_name.split("/")
        video_name = video_name[2:].replace('/'+video_name_tmp[-2]+'/'+video_name_tmp[-1], "").replace('/'+video_name_tmp[1]+"/", '')
        video_label = gather_y_labels[i].cpu().item()
        video_predict = gather_y_outputs[i].cpu().item()
        test_result_list.append([video_name, video_label, video_predict])
    test_result_list = sorted(test_result_list, key=(lambda x:x[0]))
    result_dir = args.model.resume.replace('ckpt/','')[:-4]
    result_dir = os.path.join(result_dir, test_label)
    os.makedirs(result_dir, exist_ok=True)
    predict_file = result_dir+"/"+args.final_test.dataset.params.method+".csv"
    pd.DataFrame(test_result_list, columns=["video", "label", "predict"]).to_csv(predict_file, index=False)
    config_file = args.config
    config_file_name = os.path.basename(config_file)
    copyfile(config_file, os.path.join(result_dir, config_file_name))
    auc, acc = video_evaluation.final_scores(result_file=predict_file)
    return auc, acc

if __name__ == "__main__":
    main()
