import os
import pandas as pd
import json

def init_ff(data_path):
    root_dir = '../../datasets/rawdata/FF++/'
    # root_dir = '/home/guozonghui/project/datasets/FF++/videos/'
    file_dir = root_dir+data_path    #original_sequences/youtube/c23/videos
    params = sorted([os.path.join(file_dir, x)  for x in os.listdir(file_dir) if x[-4:] == ".mp4"])

    return params, root_dir
def init_cdf_test(data_path):
    # root_dir = '../../datasets/rawdata/Celeb-DF/videos/'
    root_dir = '../../datasets/rawdata/Celeb-DF/'
    video_list_txt=root_dir+'/List_of_testing_videos.txt'
    test_list=[]
    with open(video_list_txt) as f:
        for data in f:
            line=data.split()
            test_list+=[line[1].split("/")[-1]]
    folder_list = []
    for d in os.listdir(root_dir):
        if "-" in d:
            tmp = dict()
            tmp['part_name'] = d
            tmp['list'] = []
            for f in os.listdir(os.path.join(root_dir, d)):
                if f in test_list:
                    tmp['list'].append(f)
            folder_list.append(tmp)
    return folder_list, root_dir

def init_dfdc_test(data_path):
    root_dir = '../../datasets/rawdata/DFDC/'
    label=pd.read_csv(root_dir+'labels.csv',delimiter=',')
    folder_list=[]
    fakes = []
    for i in label['filename'].tolist():
        fakes.append(f'{i[:-4]}')
    tmp = {'part_name':'test_videos', 'originals': None, 'fakes':fakes}
    folder_list.append(tmp)
    # folder_list=[f'{root_dir}/test_videos/{i}' for i in label['filename'].tolist()]
    return folder_list, root_dir

from lib import dfdc_utils
def init_dfdc_train(data_path):
    root_dir = '../../datasets/rawdata/DFDC/train_videos/'
    folder_list = []
    for d in os.listdir(root_dir):
        if "dfdc" in d:
            part = int(d.split("_")[-1])
            for f in os.listdir(os.path.join(root_dir, d)):
                if "metadata.json" in f:
                    current_path = os.path.join(root_dir, d)
                    originals, fakes = dfdc_utils.get_originals_and_fakes(current_path)
                    tmp = {'part_name':d, 'originals': originals, 'fakes':fakes}
                    folder_list.append(tmp)
    return [folder_list], root_dir

def init_cdf_train(data_path):
    # root_dir = '../../datasets/rawdata/Celeb-DF/videos/'
    root_dir = '../../datasets/rawdata/Celeb-DF/'
    video_list_txt=root_dir+'/List_of_testing_videos.txt'
    test_list=[]
    with open(video_list_txt) as f:
        for data in f:
            line=data.split()
            test_list+=[line[1].split("/")[-1]]
    folder_list = []
    for d in os.listdir(root_dir):
        if "-" in d:
            tmp = dict()
            tmp['part_name'] = d
            tmp['list'] = []
            for f in os.listdir(os.path.join(root_dir, d)):
                if f not in test_list:
                    tmp['list'].append(f)
            folder_list.append(tmp)
    return folder_list, root_dir

def init_DeeperForensics_fake(data_path):
    # root_dir = '../../datasets/rawdata/DeeperForensics-1.0/manipulated_videos/end_to_end/'
    root_dir = '../../datasets/rawdata/DeeperForensics-1.0/'
    file_dir = root_dir+data_path    #original_sequences/youtube/c23/videos
    params = sorted([os.path.join(file_dir, x)  for x in os.listdir(file_dir) if x[-4:] == ".mp4"])

    return params, root_dir

def init_DeeperForensics_real(data_path):
    root_dir = '../../datasets/rawdata/DeeperForensics-1.0/source_videos/videos/'
    folder_list = []
    for d in os.listdir(root_dir):
        if 'face_v1' not in d and 'face_v2' not in d and 'frame_face_v3' not in d:
            for f1 in os.listdir(os.path.join(root_dir, d)):
                # if 'light_uniform' in f1:
                    for f2 in os.listdir(os.path.join(root_dir, d, f1)):
                        for f3 in os.listdir(os.path.join(root_dir, d, f1, f2)):
                            if 'BlendShape' not in f3:
                                # if 'camera_front' in f3:
                                # if 'camera_down' in f3:  
                                    for f4 in os.listdir(os.path.join(root_dir, d, f1, f2, f3)):
                                        if f4[-4:] == ".mp4":
                                            folder_list.append(f4)
                            else:
                                if f3[-4:] == ".mp4":
                                    folder_list.append(f3)
    return folder_list, root_dir

def init_ffiw_source(data_path):
    root_dir = '../../datasets/rawdata/FFIW10K-v1-release/'
    
    folder_list = []
    for d in os.listdir(root_dir+data_path):
        # if '.DS_Store' not in d:
        if d in ['train', 'val']:
            for f in os.listdir(os.path.join(root_dir+data_path, d)):
                if f[-4:] == ".mp4":
                    folder_list.append((d, f))
    return folder_list, root_dir


def init_FN_train(data_path,dir_list):
    # root_dir = '../../datasets/rawdata/ForgeryNet/Training/image/images/train_release/'
    root = '../../datasets/rawdata/ForgeryNet/'
    root_dir = root+data_path
    folder_list = []
    for d in sorted(os.listdir(root_dir)):
        if ','+d+',' in dir_list:
            print(d)
            for f1 in os.listdir(os.path.join(root_dir, d)):
                # if 'c551ffea7f0357066bffc719be5f4953' in f1:
                    frame_name = []
                    f2_file = True
                    for f2 in  sorted(os.listdir(os.path.join(root_dir, d, f1))):
                        if os.path.isdir(os.path.join(root_dir, d, f1, f2)):
                            f2_file = False
                            # print(os.path.join(root_dir, d, f1, f2))
                            # assert 0
                            frame_name = []
                            for f3 in sorted(os.listdir(os.path.join(root_dir, d, f1, f2))):
                                frame_name.append(f3)
                            tmp = (os.path.join(d, f1, f2),frame_name)
                            folder_list.append(tmp)
                        else:
                            f2_file = True
                            frame_name.append(f2)
                    if f2_file:
                        tmp = (os.path.join(d, f1),frame_name)
                        folder_list.append(tmp)
        # break
    return folder_list, root

