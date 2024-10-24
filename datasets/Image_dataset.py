import os
import numpy as np
import torch
import torch.utils.data as data
import random
from collections import OrderedDict
from common.utils import map_util
from PIL import Image
from datasets.utils import dataloader_util

class Image_dataset(data.Dataset):
    def __init__(self,
                 root,
                 method='FaceForensics_c23',
                 split='train',
                 num_segments=8,
                 transform=None,
                 cutout=False,
                 is_sbi=False,
                 image_size=224):
        super().__init__()
        self.root = root
        self.dataset_info = []
        self.method = method
        self.split = split
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.is_cutout = cutout
        self.is_sbi = is_sbi
        self.cutout = map_util.Cutout()
        self.parse_dataset_info()
    
    def get_sampled_idx(self, file_path):
        frame_path = os.path.join(file_path, 'frame')
        file_ext = '.png'
        file_names = [f for f in os.listdir(frame_path) if f.endswith(file_ext)]
        all_frame_idxs = np.array([int(f[:-len(file_ext)]) for f in file_names])
        all_frame_idxs.sort()
        if len(all_frame_idxs) >= self.num_segments:
            step = len(all_frame_idxs) // self.num_segments
            sampled_frame_idxs = all_frame_idxs[::step][:self.num_segments]
        else:
            sampled_frame_idxs = []
            idxs = dataloader_util.check_frame_len(len(all_frame_idxs), self.num_segments)
            sampled_frame_idxs = all_frame_idxs[idxs]
        sampled_frame_idxs.sort()
        return sampled_frame_idxs
    
    def parse_dataset_info(self):
        """Parse the video dataset information"""
        dataset_list, data_root = dataloader_util.get_data_list(self.method, self.root, self.split, only_real=False)
        print('Number of videos loaded:', len(dataset_list), '\nNumber of frames per video:', self.num_segments)
        self.all_list = []
        error = 0
        for _, file_info in enumerate(dataset_list):  # 3600
            file_path, video_label = file_info[0], file_info[1]
            file_path = data_root + file_path
            if os.path.isdir(file_path):
                sampled_frame_idx = self.get_sampled_idx(file_path)
                for _, frame_idx in enumerate(sampled_frame_idx):
                    filename_frame = os.path.join(file_path, 'frame', f"{frame_idx}.png")
                    video_labels = torch.tensor(video_label)        
                    self.all_list.append((filename_frame, video_labels))     
            else:
                self.all_list.append((file_path+'/0/0.png', video_label, None))
                error = error+1
        print('Successfully loaded frames:', len(self.all_list))
        print('Failed to load frames', str(error))
        random.shuffle(self.all_list)    
               
    def __getitem__(self, index):
        flag=True        
        while flag:
            file_info = self.all_list[index]
            video_labels = file_info[1]
            all_frames = []
            sampled_frame_idxs = []
            filename_frame = file_info[0]
            frame = np.asarray(Image.open(filename_frame))
            if self.transform is not None:
                tmp_imgs = {"image": frame}
                all_frames = self.transform(**tmp_imgs)
                all_frames = OrderedDict(sorted(all_frames.items(), key=lambda x: x[0]))
                all_frames = list(all_frames.values())
                if self.is_cutout:
                    all_frames = torch.stack(all_frames)
                    process_imgs = self.cutout(all_frames)
                else:
                    process_imgs = torch.stack(all_frames)
            for i in range(self.num_segments):
                sampled_frame_idxs.append(file_info[0])
            flag=False
        return {"images":process_imgs, "labels":video_labels, "video_path":filename_frame, "sampled_frame_idxs":sampled_frame_idxs}    
    def __len__(self):
        return len(self.all_list)
