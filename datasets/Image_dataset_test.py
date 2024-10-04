import os
import numpy as np
import torch
import torch.utils.data as data
from collections import OrderedDict
from PIL import Image
from datasets.utils import dataloader_util
from glob import glob
from common.utils import map_util


class Image_dataset_test(data.Dataset):
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
        print(self.root)
        self.method = method
        self.split = split
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.is_cutout = cutout
        self.is_sbi = is_sbi
        self.method = method
        self.parse_dataset_info()
        self.cutout = map_util.Cutout()

    def get_sampled_idx(self, file_path):
        frame_path = os.path.join(file_path, 'frame')
        file_ext = '.png'
        file_names = [f for f in os.listdir(frame_path) if f.endswith(file_ext)]
        all_frame_idxs = np.array([int(f[:-len(file_ext)]) for f in file_names])
        if len(all_frame_idxs) >= self.num_segments:
            step = len(all_frame_idxs) // self.num_segments
            sampled_frame_idxs = all_frame_idxs[::step][:self.num_segments]
        else:
            sampled_frame_idxs = []
            idxs = dataloader_util.check_frame_len(len(all_frame_idxs), self.num_segments)
            sampled_frame_idxs = all_frame_idxs[idxs]
        return sampled_frame_idxs
    
    def parse_dataset_info(self):
        """Parse the video dataset information
        """
        dataset_list, data_root = dataloader_util.get_data_list(self.method, self.root, self.split, only_real=False)
        print('Number of videos loaded:', len(dataset_list), '\nNumber of frames per video:', self.num_segments)
        self.all_list = []
        error = 0
        for i, file_info in enumerate(dataset_list):  # 3600
            file_path, video_label = file_info[0], file_info[1]
            file_path = data_root + file_path
            frame_list =[]
            if os.path.isdir(file_path):               
                if not 'FFIW' in file_path:
                    sampled_frame_idx = self.get_sampled_idx(file_path)
                    for i,frame_idx in enumerate(sampled_frame_idx):
                        filename_frame = os.path.join(file_path, 'frame', f"{frame_idx}.png")
                        video_labels = torch.tensor(video_label)
                        filename_info = os.path.join(file_path, 'info', f"{frame_idx}.pkl")
                        frame_list.append((filename_frame, video_labels, filename_info))
                elif 'FFIW' in file_path:
                    video_labels = torch.tensor(video_label)
                    frame_path = glob(file_path + '/*.png', recursive=True)
                    for i,filename_frame in enumerate(frame_path):
                        frame_list.append((filename_frame, video_labels, None))
            else:
                frame_list.append((file_path+'/0/0.png', video_label, None))
                error = error+1
            self.all_list.append(frame_list)
        print('Successfully loaded videos:', len(self.all_list))
        print('Failed to load videos', str(error))
        
    def __getitem__(self, index):
        frame_list = self.all_list[index]
        file_path = frame_list[0][0]
        file_path_arr = file_path.split("/")
        file_path = file_path.replace('/'+file_path_arr[-2]+"/"+file_path_arr[-1], '')
        get_frames = []
        if len(frame_list) == self.num_segments:
            all_frames = []
            video_labels = frame_list[0][1]
            additional_targets = {}
            for i,frame in enumerate(frame_list):
                frame = Image.open(frame[0])
                frame = np.asarray(frame)
                all_frames.append(frame)
                if i == 0:
                    tmp_imgs = {"image": all_frames[0]}
                else:
                    additional_targets[f"image{i}"] = "image"
                    tmp_imgs[f"image{i}"] = all_frames[i]
            self.transform.add_targets(additional_targets)
            all_frames = self.transform(**tmp_imgs)
            all_frames = OrderedDict(sorted(all_frames.items(), key=lambda x: x[0]))
            all_frames = list(all_frames.values())
            all_frames = torch.stack(all_frames)  # T, C, H, W
        else:
            all_frames = torch.zeros((self.num_segments, 3, self.image_size, self.image_size))
            video_labels = frame_list[0][1]+2
        for i, frame in enumerate(all_frames):
            get_frames.append(frame)
        process_imgs = torch.stack(get_frames)
        video_labels = torch.tensor(video_labels)
        return {"images":process_imgs, "labels":video_labels, "video_path":file_path, 'index':index}
    def __len__(self):
        return len(self.all_list)
