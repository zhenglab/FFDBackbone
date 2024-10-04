import os
import random
import shutil

def sample_images(source_path, target_path, num_samples):
    # 获取源路径下所有的图像文件
    image_files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f)) 
                   and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.mp4'))]
    # image_files = get_DFDC(0)
    # print(image_files)
    # 随机采样指定数量的图像文件
    sampled_images = random.sample(image_files, min(num_samples, len(image_files)))
    i = 0
    # 复制采样到的图像文件到目标路径
    for image in sampled_images:
        source_image_path = os.path.join(source_path, image)
        # target_image_path = os.path.join(target_path, image)
        target_image_path = os.path.join(target_path, str(1000+i)+'.mp4')
        
        shutil.copyfile(source_image_path, target_image_path)
        print(f"复制文件: {image} 到 {target_path}")
        i +=1
def find_mp4_files(root_dir='/SSD0/guozonghui/project/datasets/rawdata/FF++/manipulated_sequences/DeepFakeDetection/c23'):
    mp4_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    print(len(mp4_files))
    return mp4_files


def sample_images_v2(source_path, target_path, num_samples):
    image_files = find_mp4_files()
    # print(len(image_files))
    # assert 0
    sampled_images = random.sample(image_files, min(num_samples, len(image_files)))
    for i, image in enumerate(sampled_images):
        target_image_path = os.path.join(target_path, str(1000+i)+'.mp4')
        print(f"复制文件: {image} 到 {target_path}")
        shutil.copyfile(image, target_image_path)


import pandas as pd
def get_DFDC(choose_label=1):
    file_list = []
    root = '/SSD0/guozonghui/project/datasets/rawdata/DFDC/test_videos/'
    label=pd.read_csv('/SSD0/guozonghui/project/datasets/rawdata/DFDC/labels.csv',delimiter=',')
    dataset_info = [(video_name[:-4], label) for video_name, label in zip(label['filename'].tolist(), label['label'].tolist())]
    filtered_video_names = [video_name for video_name, label in dataset_info if label == choose_label]
    for i in range(len(filtered_video_names)):
        file_list.append(filtered_video_names[i]+'.mp4')
    return file_list
# dfdc_real = get_DFDC(0)
# dfdc_fake = get_DFDC(1)

import os






# # # 指定源路径、目标路径和要采样的图像数量
source_path = "/SSD0/guozonghui/project/datasets/rawdata/FFIW10K-v1-release/source/train"
target_path = "/SSD0/guozonghui/project/FFD/ffd_video_data/real/FFIW"
num_samples = 2500

# # 执行图像采样和复制
sample_images(source_path, target_path, num_samples)
# sample_images_v2(source_path, target_path, num_samples)
