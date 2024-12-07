## Face Forgery Detection with Elaborate Backbone

[Introduction](#introduction) |
[Preparation](#Preparation) |
[Get Started](#get-started) |
[Paper](https://arxiv.org/abs/2409.16945) |

### Introduction

Welcome to `FFDBackbone`, a comprehensive project to fintune various backbones for face forgery detection (FFD). This project offers robust support for loading backbones through diverse pre-training methods, fostering seamless integration of ideas and enabling a swift project initiation.

Its highlights are as follows:
-  Support the implementation of mainstream networks, including ResNet, XceptionNet, EfficientNet, ViT.
-  Support the transfer of mainstream self-supervised pre-trained backbone, including MoCo v2, MoCo v3, SimCLR, BYOL, SimSiam, MAE, BEiT v2, BEiT v3, SimMIM, DINO, ConMIM, etc.
-  Provided further pre-trained backbones on real faces, which can more effectively improve the performance of FFD.
-  Provided an effective training framework to improve fine-tuning efficiency. 

### Preparation

#### 1. Environment and Dependencies:

This project is implemented with Python version >= 3.10 and CUDA version >= 11.3.

It is recommended to follow the steps below to configure the environment:
```
conda create -n ffdbackbone python=3.10
conda activate ffdbackbone
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

#### 2.Data Preparation:

Before training, follow the steps below to prepare the data:
1. Download datasets and put them under  `../data/` .
   
2. Frame Extraction: Extract frames from the video files. It is recommended to extract 8 frames at regular intervals, which can control the training time cost while maintaining commendable performance.
   
3. Face Alignment and Cropping: Referring to the [FTCN](https://github.com/yinglinzheng/FTCN), RetinaFace was chosen for facial recognition, followed by cropping and alignment procedures. When multiple faces appear in the video, tracking the face with the longest appearance time for preservation.

### Quickly Inference

Download weights from [Baidu Cloud(code: fr6r)](https://pan.baidu.com/s/1EOgJeE4Gb4TAaxvSkhK4lw) or [Google Cloud](https://drive.google.com/drive/folders/1jFIpb4TftJiL82h69c3Roqy8uIzGQP4G?usp=sharing)  and put it into 'checkpoints/Ours/ckpt/'.

Infer a single image: Run the ```python Inference.py``` command and enter the path from the keyboard each time. If a folder path is entered, the program will automatically infer all the images under the path.

### Get Started

### 1.  Training

#### 1.1 Choose a pre-trained backbone

Backbone is crucial for the task of face forgery detection(FFD), as detailed in the [paper](https://arxiv.org/abs/2409.16945).Different Backbone can be loaded through the configuration files in `configs`.

Notice: It is recommended to use BEiT v2' as the backbone, which is pre-trained on large-scale real faces, exhibits significant promise, and is demonstrated to be highly effective for face forgery detection tasks. By fine-tuning BEiT v2' on FF++, the state-of-the-art (SOTA) performance on FFD task can be achieved:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">FF++</th>
<th valign="bottom">Celeb-DF</th>
<th valign="bottom">DFDC</th>
<th valign="bottom">FFIW</th>
<th valign="bottom">Checkpoints</th>

<!-- TABLE BODY -->
<tr><td align="left">BEiT_v2'</td>
<td align="center">99.14%</td>
<td align="center">89.18%</td>
<td align="center">84.02%</td>
<td align="center">86.48%</td>
<td align="center"><a href="https://pan.baidu.com/s/1EOgJeE4Gb4TAaxvSkhK4lw">Baidu(code: fr6r)</a>   <a href="https://drive.google.com/drive/folders/1jFIpb4TftJiL82h69c3Roqy8uIzGQP4G?usp=sharing">Google</a></td>
<tr><td align="left">Ours</td>
<td align="center">99.36%</td>
<td align="center">90.46%</td>
<td align="center">84.90%</td>
<td align="center">90.97%</td>
<td align="center"><a href="https://pan.baidu.com/s/1EOgJeE4Gb4TAaxvSkhK4lw">Baidu(code: fr6r)</a>   <a href="https://drive.google.com/drive/folders/1jFIpb4TftJiL82h69c3Roqy8uIzGQP4G?usp=sharing">Google</a></td>
</tbody></table>

#### 1.2 Choose a fine-tuning framework


Fine-tuning different backbones directly *(Backbone and FC with Cross-entropy loss)*:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 11023 train.py -c configs/backbones/**.yaml
```

Fine-tuning with competitive backbone Framework:
Download weights(BEiT-1k-Face-55w.tar) from [Baidu Cloud(code: fr6r)](https://pan.baidu.com/s/1EOgJeE4Gb4TAaxvSkhK4lw) or [Google Cloud](https://drive.google.com/drive/folders/1jFIpb4TftJiL82h69c3Roqy8uIzGQP4G?usp=sharing) and put it into 'pretrained_weight/' . 

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 11023 train_dualbranch.py -c configs/DBBF.yaml
```
### 4. Evaluation

Download weights from [Baidu Cloud(code: fr6r)](https://pan.baidu.com/s/1EOgJeE4Gb4TAaxvSkhK4lw) or [Google Cloud](https://drive.google.com/drive/folders/1jFIpb4TftJiL82h69c3Roqy8uIzGQP4G?usp=sharing) and put it into checkpoints/Ours/ckpt/ . Then run:

```
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 11011 eval.py -c checkpoints/Ours/DBBF.yaml
```

### Citation

```
@article{guo2024face,
  title={Face Forgery Detection with Elaborate Backbone},
  author={Guo, Zonghui and Liu, Yingjie and Zhang, Jie and Zheng, Haiyong and Shan, Shiguang},
  journal={arXiv preprint arXiv:2409.16945},
  year={2024}
}
```
