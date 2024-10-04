
import os
from pylab import *
import matplotlib.font_manager as fm # to create font
import json
import pandas as pd
import math
from glob import glob
import random
seed_value = 1234 
random.seed(seed_value)
REAL_LABLE = 0
FAKE_LABEL = 1

dataset_root = {
    'FaceForensics_c23_train': 'FF++/face_v1/',
    'FaceForensics': 'FF_frames/face_v1/',
    'Celeb-DF': 'Celeb-DF_frames/face_v1/',
    'DFDC': 'DFDC_frames/face_v1/',
    'FFIW':'../FFIW/',}

def get_data_list(dataset_name, base_root, split, only_real=False):
    dataset_info = []
    if "FaceForensics" in dataset_name:
        compress = dataset_name.split("_")[1]
        if split == 'train':
            root = os.path.join(base_root, dataset_root[dataset_name+"_train"])
        else:
            root = os.path.join(base_root, dataset_root[dataset_name.split("_")[0]])
        dataset_info = get_FF_list(root, split, compress=compress, only_real=only_real)
    elif dataset_name == 'Celeb-DF' and split=='test':
        root = os.path.join(base_root, dataset_root[dataset_name])
        video_list_txt = os.path.join(root, 'List_of_testing_videos.txt')
        with open(video_list_txt) as f:
            for data in f:
                line=data.split()
                dataset_info.append((line[1][:-4],FAKE_LABEL-int(line[0])))
    elif dataset_name == 'DFDC' and split=='test':
        root = os.path.join(base_root, dataset_root[dataset_name])
        label=pd.read_csv(root+'labels.csv',delimiter=',')
        dataset_info = [(video_name[:-4], label) for video_name, label in zip(label['filename'].tolist(), label['label'].tolist())]
        root = root+'test_videos/'
    elif dataset_name == 'FFIW' and split=='test':
        root = ''
        real_root = os.path.join(dataset_root[dataset_name],'source')
        fake_root = os.path.join(dataset_root[dataset_name],'target')
        real_path = glob(real_root + '/*', recursive=True)
        fake_path = glob(fake_root + '/*', recursive=True)
        for i,path in enumerate(real_path):
            dataset_info.append((path, REAL_LABLE))
        for i,path in enumerate(fake_path):
            dataset_info.append((path, FAKE_LABEL))
    else:
        print('not support!', dataset_name)
        assert 0
    return dataset_info, root


def get_FF_list(root, split, compress='c23', only_real=False):
    split_json_path = os.path.join(root, 'splits', f'{split}.json')
    json_data = json.load(open(split_json_path, 'r'))
    if only_real:
        real_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
        real_video_dir = os.path.join('original_sequences', 'youtube', compress, 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), REAL_LABLE] for x in real_names]
    else:
        real_names = []
        fake_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
            fake_names.extend([f'{item[0]}_{item[1]}', f'{item[1]}_{item[0]}'])
        real_video_dir = os.path.join('original_sequences', 'youtube', compress, 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), 0] for x in real_names]
        ff_fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        for method in ff_fake_types:
            fake_video_dir = os.path.join('manipulated_sequences', method, compress, 'videos')
            for x in fake_names:
                dataset_info.append((os.path.join(fake_video_dir,x),FAKE_LABEL))
    return dataset_info


def check_frame_len(video_len, num_segments):
    inner_index = list(range(video_len))
    pad_length = math.ceil((num_segments-video_len)/2)
    post_module = inner_index[1:-1][::-1] + inner_index
    l_post = len(post_module)
    post_module = post_module * (pad_length // l_post + 1)
    post_module = post_module[:pad_length]
    assert len(post_module) == pad_length
    pre_module = inner_index + inner_index[1:-1][::-1]
    l_pre = len(post_module)
    pre_module = pre_module * (pad_length // l_pre + 1)
    pre_module = pre_module[-pad_length:]
    assert len(pre_module) == pad_length
    sampled_clip_idxs = pre_module + inner_index + post_module
    sampled_clip_idxs = sampled_clip_idxs[:num_segments]
    return sampled_clip_idxs
