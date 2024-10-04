import numpy as np
from lib.ct.detection import FaceDetector
import cv2
from lib.utils import flatten,partition
from tqdm import tqdm
from imutils import face_utils
import custom_data_list
import os
import face_utils
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
detector = FaceDetector(0)

def split_array(array, segment_size):
    segmented_array = []
    for i in range(0, len(array), segment_size):
        segment = array[i:i+segment_size]
        segmented_array.append(segment)
    return segmented_array

def process(method):
    root = f'/SSD0/guozonghui/project/FFD/Real_video/Frame/{method}'
    data_list = custom_data_list.get_image_files(root)
    print(len(data_list))
    print(data_list[0])
    frames = []
    data_list = sorted(data_list)
    data_list_split = split_array(data_list, 1)
    target_width = 512
    target_height = 512
    image_size = 224
    for clip in tqdm(data_list_split):
        frames = []
        for frame_name in clip:
            tmp = frame_name.replace('Frame', 'Frame_align2')
            if not os.path.isfile(tmp):
                frame=cv2.imread(os.path.join(frame_name))
            # frame = cv2.resize(frame, (512, 512))
            # # 获取图像的原始尺寸
            # height, width = frame.shape[:2]
            
            # # 计算裁剪的起始位置
            # start_x = (width - target_width) // 2
            # start_y = (height - target_height) // 2
            
            # # 执行中心裁剪
            # frame = frame[start_y:start_y+target_height, start_x:start_x+target_width]
            frames.append(frame)
        if len(frames) > 0:
            detect_res = flatten(
                [detector.detect(item) for item in partition(frames, 1)]
            )
            detect_res = get_valid_faces(detect_res, thres=0.5)
            
            for faces, frame, frame_name in zip(detect_res, frames, clip):
                # for i, (bbox, lm5, score) in enumerate(faces):
                if len(faces) > 0:
                    bbox, lm5, score = faces[0]
                    frame, landmark, bbox=face_utils.crop_aligned(frame,lm5,landmarks_68=None,bboxes=bbox,aligned_image_size=image_size)
                    bbox = np.array([[bbox[0],bbox[1]],[bbox[2],bbox[3]]])
                    # frame_croped = crop_face_sbi(frame, bbox=bbox, margin=False)
                    frame_croped = crop_face_sbi(frame,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase='test')
                    
                    frame_croped = cv2.resize(frame_croped,(224,224),interpolation=cv2.INTER_LINEAR)

                    frame_name = frame_name.replace('Frame', 'Frame_align')
                    directory_path = os.path.dirname(frame_name)
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                    try:
                        cv2.imwrite(frame_name, frame_croped)
                    except Exception as e:
                        # Catch-all exception handling
                        print("An error occurred:", str(e))
            # assert 0

def get_valid_faces(detect_results, max_count=10, thres=0.5, at_least=False):
    new_results = []
    for i, faces in enumerate(detect_results):
        # faces = sorted(faces, key=lambda x: bbox_range(x[0]), reverse=True)
        # print(len(faces))
        # assert 0
        if len(faces) > max_count:
            faces = faces[:max_count]
        l = []
        for j, face in enumerate(faces):
            if face[-1] < thres and not (j == 0 and at_least):
                continue
            box, lm, score = face
            box = box.astype(np.float)
            lm = lm.astype(np.float)
            l.append((box, lm, score))
        new_results.append(l)
    return new_results


def crop_face_sbi(img,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
    assert phase in ['train','val','test']

    #crop face------------------------------------------
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

    return img_cropped


if __name__=='__main__':
    methods = ['Celeb-DF', 'DFDC', 'DiffHead', 'FF', 'FFIW', 'hrfae', 'iplap', 'makeittalker', 'mobileswap', 'sadtalker', 'styleHEAT', 'VIPL', 'wav2lip']
    methods = [  'DFv1', 'FF', 'FFIW', 'ROSE-Youtu','DFDC',]
    methods = ['DFDC']
    for method in methods:
        process(method)

