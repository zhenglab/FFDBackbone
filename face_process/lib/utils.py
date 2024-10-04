import numpy as np
import cv2, torch, os
import matplotlib.pyplot as plt
from pylab import *
from PIL import Image,ImageFont,ImageDraw
import matplotlib.font_manager as fm # to create font
import numpy as np
import cv2
import os
import platform
import json
import errno
import math


def weak_check(detect_res):
    return sum([len(faces) for faces in detect_res]) > len(detect_res) * 0.75


def get_crop_box(shape, box, scale=0.5):
    height, width = shape
    box = np.rint(box).astype(np.int)
    new_box = box.reshape(2, 2)
    size = new_box[1] - new_box[0]
    diff = scale * size
    diff = diff[None, :] * np.array([-1, 1])[:, None]
    new_box = new_box + diff
    new_box[:, 0] = np.clip(new_box[:, 0], 0, width - 1)
    new_box[:, 1] = np.clip(new_box[:, 1], 0, height - 1)
    new_box = np.rint(new_box).astype(np.int)
    return new_box.reshape(-1)


def get_fps(input_file):
    reader = cv2.VideoCapture(input_file)
    fps = reader.get(cv2.CAP_PROP_FPS)
    reader.release()
    return fps



def mkdir_p(dirname):
    """Like "mkdir -p", make a dir recursively, but do nothing if the dir exists
    这个是线程安全的, from Lingzhi Li
    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == "" or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def mkdir(*args):
    for folder in args:
        if not os.path.isdir(folder):
            mkdir_p(folder)


def make_join(*args):
    folder = os.path.join(*args)
    mkdir(folder)
    return folder


def list_dir(folder, condition=None, key=lambda x: x, reverse=False, co_join=[]):
    files = os.listdir(folder)
    if condition is not None:
        files = filter(condition, files)
    co_join = [folder] + co_join
    if key is not None:
        files = sorted(files, key=key, reverse=reverse)
    files = [(file, *[os.path.join(fold, file) for fold in co_join]) for file in files]
    return files

def get_jointer(file):
    def jointer(folder):
        return os.path.join(folder, file)

    return jointer

def flatten(l):
    return [item for sublist in l for item in sublist]


def is_win():
    return platform.system() == "Windows"


def get_postfix(post_fix):
    return lambda x: x.endswith(post_fix)


def partition(images, size):
    """
    Returns a new list with elements
    of which is a list of certain size.

        >>> partition([1, 2, 3, 4], 3)
        [[1, 2, 3], [4]]
    """
    return [
        images[i : i + size] if i + size <= len(images) else images[i:]
        for i in range(0, len(images), size)
    ]


def load_json(file):
    with open(file, "r") as f:
        res = json.load(f)
    return res


def save_json(file, obj):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def save_image(image_numpy, image_path, aspect_ratio=1.0):

    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path,quality=75) #added by Mia (quality)


def save_frames_video(image, idx,save_dir):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    t = image.size(1)
    image = image.transpose(1,0)
    # font=ImageFont.truetype('arial.ttf',10)
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    result = Image.new('RGB', (t*one_image_w-padding_num*2, one_image_w+20),"white")
            # print(image.size())
    for i in range(t):
        image_pil = Image.fromarray(tensor2im(image[i])).resize([one_image,one_image], Image.BICUBIC)
        result.paste(image_pil, box=(i*one_image_w, 0))
        drawer=ImageDraw.Draw(result)
        drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w),fill=(0,0,0),text=str(idx[i].item()),font=font)
    img_path = os.path.join(save_dir+'.jpg')
    result.save(img_path,  quality=75)

from PIL import Image,ImageDraw,ImageFont
import matplotlib.font_manager as fm # to create font

def save_sbi_train_images(video_path,imgs_r, imgs_f, masks,mean, std,sampled_idxs, save_dir=None, save_name=None, return_np=False):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    t = imgs_r.size(1)
    result = Image.new('RGB', (t*one_image_w-padding_num*2, one_image_w*3-padding_num+40),"white")
    for i in range(t):
        img_r = Image.fromarray(tensor2im(imgs_r[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        img_f = Image.fromarray(tensor2im(imgs_f[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        mask = Image.fromarray(tensor2im(masks[:,i,:,:,:])).resize([one_image,one_image], Image.BICUBIC)
        
        result.paste(img_r, box=(i*one_image_w, 0))
        result.paste(img_f, box=(i*one_image_w, one_image_w))
        result.paste(mask, box=(i*one_image_w, one_image_w*2))
        drawer=ImageDraw.Draw(result)
        drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w*3),fill=(0,0,0),text=str(sampled_idxs[0][i]),font=font)
    video_name = video_path[0].split("/")[-1][:-4]
    
    if return_np:
        return np.array(result)
    else:
        img_path = os.path.join(save_dir+'/'+video_name+'-'+save_name+'.jpg')
        result.save(img_path,  quality=75)


def save_real_data_images(video_path,imgs, labels, mean, std,sampled_idxs, save_dir=None, save_name=None, return_np=False):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    t = imgs.size(1)
    result = Image.new('RGB', (t*one_image_w-padding_num*2, one_image_w-padding_num+40),"white")
    for i in range(t):
        img_r = Image.fromarray(tensor2im(imgs[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        
        result.paste(img_r, box=(i*one_image_w, 0))
        drawer=ImageDraw.Draw(result)
        drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w),fill=(0,0,0),text=str(sampled_idxs[0][i].item()),font=font)
        drawer.text(xy=(0,one_image_w),fill=(255,0,0),text=str(labels[0].item()),font=font)
    video_name = video_path.split("/")[-1][:-4]
    
    if return_np:
        return np.array(result)
    else:
        img_path = os.path.join(save_dir+'/'+video_name+'-'+save_name+'.jpg')
        result.save(img_path,  quality=75)

def save_bi_images(video_name,imgs_r, imgs_f, sampled_idxs,landmarks_68,landmarks_5,bboxes,masks=None, file='bi'):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    t = len(imgs_f)
    result = Image.new('RGB', (t*one_image_w-padding_num*2, one_image_w*3-padding_num+40),"white")
    for i in range(t):
        img_r = visualize_landmark(imgs_r[i], landmarks_68[i],landmarks_5[i],bboxes[i])
        img_r = img_r.resize([one_image,one_image], Image.BICUBIC)
        img_f = visualize_landmark(imgs_f[i], landmarks_68[i],landmarks_5[i],bboxes[i])
        img_f = img_f.resize([one_image,one_image], Image.BICUBIC)
        if len(masks[i].shape) > 2:
            masks[i] = masks[i][:,:,0]
        mask = Image.fromarray(np.uint8(masks[i]*255)).convert('RGB').resize([one_image,one_image], Image.BICUBIC)

        result.paste(img_r, box=(i*one_image_w, 0))
        result.paste(img_f, box=(i*one_image_w, one_image_w))
        result.paste(mask, box=(i*one_image_w, one_image_w*2))
        drawer=ImageDraw.Draw(result)
        drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w*3),fill=(0,0,0),text=str(sampled_idxs[i]),font=font)
    img_path = 'images/'+str(video_name)+'_'+file+'.jpg'
    result.save(img_path,  quality=75)
    
    
def visualize_landmark(image_array, landmarks,landmarks_5,bboxes):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    if landmarks is not None:
        landmarks = np.array(landmarks)
        for landmark in landmarks:
            # print(landmark.shape)   #81,2
            draw.point((landmark[0],landmark[1]),fill = (255, 255, 0))
    if landmarks_5 is not None:
        for landmark in landmarks_5:
            draw.point((landmark[0],landmark[1]),fill = (0, 255, 0))
    # line = 5
    # x, y = 10, 10
    # width, height = 100, 50
    # for i in range(1, line + 1):
    #     draw.rectangle((x + (line - i), y + (line - i), x + width + i, y + height + i), outline='red')
    if bboxes is not None:
        draw.rectangle((bboxes[0][0],bboxes[0][1],bboxes[1][0],bboxes[1][1]), outline='red',width=2)

    # imshow(origin_img)
    return origin_img 
def save_multi_image(images,sampled_idxs, save_dir, mode='RGB'):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    result = Image.new(mode, (len(images)*one_image_w-padding_num*2, one_image_w+20),"white")
    for i, (image,idx) in enumerate(zip(images, sampled_idxs)):
        # image_pil = Image.fromarray(tensor2im(image[i])).resize([one_image,one_image], Image.BICUBIC)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = image.resize([one_image,one_image], Image.BICUBIC)
        result.paste(image, box=(i*one_image_w, 0))
        drawer=ImageDraw.Draw(result)
        drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w),fill=(0,0,0),text=str(idx),font=font)
    img_path = os.path.join(save_dir+'.jpg')
    result.save(img_path,  quality=75)
    
    
def extract_image_chips(img, points, desired_size=256, padding=0.37):
        """
        crop and align face
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
            desired_size: default 256
            padding: default 0
        Retures:
        -------
            crop_imgs: list, n
                cropped and aligned faces 
        """
        crop_imgs = []
        # originally for p in points:
        for p in [points]:
            shape  = []
            # for k in range(int(len(p)/2)):
            for k in range(points.shape[0]):
                shape.append(p[k][0])
                shape.append(p[k][1])
            
            if padding > 0:
                padding = padding
            else:
                padding = 0

            # average positions of face points
            mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
            mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]

            '''

            # 2021.1.13 Modifications
            mean_face_shape_x = np.array([89.3095, 169.3095, 127.8949, 96.8796, 159.1065])
            mean_face_shape_y = np.array([72.9025, 72.9025, 127.0441, 184.8907, 184.7601])
            scale = 0.75
            mean_face_shape_x = (mean_face_shape_x * scale + 128 * (1 - scale)) / 255
            mean_face_shape_y = (mean_face_shape_y * scale + 128 * (1 - scale) + 20) / 255
            '''

            from_points = []
            to_points = []

            for i in range(int(len(shape)/2)):
                x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * desired_size
                y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * desired_size
                to_points.append([x, y])
                from_points.append([shape[2*i], shape[2*i+1]])

            # convert the points to Mat
            from_mat = list2colmatrix(from_points)
            to_mat = list2colmatrix(to_points)

            # compute the similar transfrom
            tran_m, tran_b = find_tfrom_between_shapes(from_mat, to_mat)

            probe_vec = np.matrix([1.0, 0.0]).transpose()
            probe_vec = tran_m * probe_vec

            scale = np.linalg.norm(probe_vec)
        angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

        from_center = [(shape[0]+shape[2])/2.0, (shape[1]+shape[3])/2.0]
        to_center = [0, 0]
        to_center[1] = desired_size * 0.4
        to_center[0] = desired_size * 0.5

        ex = to_center[0] - from_center[0]
        ey = to_center[1] - from_center[1]

        rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1*angle, scale)
        rot_mat[0][2] += ex
        rot_mat[1][2] += ey

        chips = cv2.warpAffine(img, rot_mat, (desired_size, desired_size))
        crop_imgs.append(chips)

        return crop_imgs[0]
def list2colmatrix(pts_list):
    """
    convert list to column matrix
    Parameters:
    ----------
        pts_list:
            input list
    Retures:
    -------
        colMat: 

    """
    assert len(pts_list) > 0
    colMat = []
    for i in range(len(pts_list)):
        colMat.append(pts_list[i][0])
        colMat.append(pts_list[i][1])
    colMat = np.matrix(colMat).transpose()
    return colMat

def find_tfrom_between_shapes(from_shape, to_shape):
    """
        find transform between shapes
    Parameters:
    ----------
        from_shape: 
        to_shape: 
    Retures:
    -------
        tran_m:
        tran_b:
    """
    assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

    sigma_from = 0.0
    sigma_to = 0.0
    cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

    # compute the mean and cov
    from_shape_points = from_shape.reshape(int(from_shape.shape[0]/2), 2)
    to_shape_points = to_shape.reshape(int(to_shape.shape[0]/2), 2)
    mean_from = from_shape_points.mean(axis=0)
    mean_to = to_shape_points.mean(axis=0)

    for i in range(from_shape_points.shape[0]):
        temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
        sigma_from += temp_dis * temp_dis
        temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
        sigma_to += temp_dis * temp_dis
        cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

    sigma_from = sigma_from / to_shape_points.shape[0]
    sigma_to = sigma_to / to_shape_points.shape[0]
    cov = cov / to_shape_points.shape[0]

    # compute the affine matrix
    s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    u, d, vt = np.linalg.svd(cov)

    if np.linalg.det(cov) < 0:
        if d[1] < d[0]:
            s[1, 1] = -1
        else:
            s[0, 0] = -1
    r = u * s * vt
    c = 1.0
    if sigma_from != 0:
        c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

    tran_b = mean_to.transpose() - c * r * mean_from.transpose()
    tran_m = c * r

    return tran_m, tran_b

def crop_face_sbi(img,landmark=None,bbox=None,lm5=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
    assert phase in ['train','val','test']

    #crop face------------------------------------------
    H,W=len(img),len(img[0])

    assert landmark is not None or bbox is not None

    H,W=len(img),len(img[0])

    if crop_by_bbox:
        x0,y0=bbox[0]
        x1,y1=bbox[1]
        w=x1-x0
        h=y1-y0
        w0_margin=w/4#0#np.random.rand()*(w/8)
        w1_margin=w/4
        h0_margin=h/4#0#np.random.rand()*(h/5)
        h1_margin=h/4
    else:
        x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
        x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
        w=x1-x0
        h=y1-y0
        w0_margin=w/8#0#np.random.rand()*(w/8)
        w1_margin=w/8
        h0_margin=h/2#0#np.random.rand()*(h/5)
        h1_margin=h/5



    if margin:
        w0_margin*=4
        w1_margin*=4
        h0_margin*=2
        h1_margin*=2
    elif phase=='train':
        w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()	
    else:
        w0_margin*=0.5
        w1_margin*=0.5
        h0_margin*=0.5
        h1_margin*=0.5
            
    y0_new=max(0,int(y0-h0_margin))
    y1_new=min(H,int(y1+h1_margin)+1)
    x0_new=max(0,int(x0-w0_margin))
    x1_new=min(W,int(x1+w1_margin)+1)

    img_cropped=img[y0_new:y1_new,x0_new:x1_new]
    if landmark is not None:
        landmark_cropped=np.zeros_like(landmark)
        for i,(p,q) in enumerate(landmark):
            landmark_cropped[i]=[p-x0_new,q-y0_new]
    else:
        landmark_cropped=None
    if lm5 is not None:
        lm5_cropped=np.zeros_like(lm5)
        for i,(p,q) in enumerate(lm5):
            lm5_cropped[i]=[p-x0_new,q-y0_new]
    else:
        lm5_cropped=None
    if bbox is not None:
        bbox_cropped=np.zeros_like(bbox)
        for i,(p,q) in enumerate(bbox):
            bbox_cropped[i]=[p-x0_new,q-y0_new]
    else:
        bbox_cropped=None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped,landmark_cropped,bbox_cropped,lm5_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
    else:
        return img_cropped,landmark_cropped,bbox_cropped,lm5_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)