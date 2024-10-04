"""Crop face from image via landmarks
"""
import cv2
import numpy as np
from skimage.transform import SimilarityTransform
import albumentations as alb
import math
def add_face_margin(x, y, w, h, margin=0.5):
    """Add marigin to face bounding box
    """
    x_marign = int(w * margin / 2)
    y_marign = int(h * margin / 2)

    x1 = x - x_marign
    x2 = x + w + x_marign
    y1 = y - y_marign
    y2 = y + h + y_marign

    return x1, x2, y1, y2


def get_face_box(img, landmarks, margin):
    """Get faca bounding box from landmarks

    Args:
        img (np.array): input image
        landmarks (np.array): face landmarks
        margin (float): margin for face box

    Returns:
        list: face bouding box
    """
    # load the positions of five landmarks
    x_list = [
        int(float(landmarks[6])),
        int(float(landmarks[8])),
        int(float(landmarks[10])),
        int(float(landmarks[12])),
        int(float(landmarks[14]))
    ]
    y_list = [
        int(float(landmarks[7])),
        int(float(landmarks[9])),
        int(float(landmarks[11])),
        int(float(landmarks[13])),
        int(float(landmarks[15]))
    ]

    x, y = min(x_list), min(y_list)
    w, h = max(x_list) - x, max(y_list) - y

    side = w if w > h else h

    # add margin
    x1, x2, y1, y2 = add_face_margin(x, y, side, side, margin)
    max_h, max_w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(max_w, x2)
    y2 = min(max_h, y2)

    return x1, x2, y1, y2


def crop_aligned(img,landmarks, landmarks_68=None,bboxes=None,aligned_image_size=None,zoom_in=1):
    aligned,RotationMatrix = norm_crop(img, landmarks, aligned_image_size,zoom_in)
    # aligned = Image.fromarray(aligned[:, :, ::-1])
    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmarks.shape[0]):
        pts = []    
        pts.append(RotationMatrix[0,0]*landmarks[i,0]+RotationMatrix[0,1]*landmarks[i,1]+RotationMatrix[0,2])
        pts.append(RotationMatrix[1,0]*landmarks[i,0]+RotationMatrix[1,1]*landmarks[i,1]+RotationMatrix[1,2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    if landmarks_68 is not None:
        new_landmark_68 = []
        for i in range(landmarks_68.shape[0]):
            pts = []    
            pts.append(RotationMatrix[0,0]*landmarks_68[i,0]+RotationMatrix[0,1]*landmarks_68[i,1]+RotationMatrix[0,2])
            pts.append(RotationMatrix[1,0]*landmarks_68[i,0]+RotationMatrix[1,1]*landmarks_68[i,1]+RotationMatrix[1,2])
            new_landmark_68.append(pts)

        new_landmark_68 = np.array(new_landmark_68)

    new_bbox = []
    # for i in [0,2]:
    #     new_bbox.append(RotationMatrix[0,0]*bboxes[i]+RotationMatrix[0,1]*bboxes[i+1]+RotationMatrix[0,2])  #boxes  [左上角x坐标，左上角y坐标，右下角x坐标，右下角y坐标]
    #     new_bbox.append(RotationMatrix[1,0]*bboxes[i]+RotationMatrix[1,1]*bboxes[i+1]+RotationMatrix[1,2])
    # for i in range(bboxes.shape[0]):
    #     pts = []    
    #     print(bboxes[i])
    #     pts.append(RotationMatrix[0,0]*bboxes[i,0]+RotationMatrix[0,1]*bboxes[i,1]+RotationMatrix[0,2])
    #     pts.append(RotationMatrix[1,0]*bboxes[i,0]+RotationMatrix[1,1]*bboxes[i,1]+RotationMatrix[1,2])
    #     print(pts)
    #     print("-------")
    #     new_bbox.append(pts)
    new_bbox = bbox_rotate(bboxes,RotationMatrix,(img.shape[1], img.shape[0]))

    new_bbox = np.array(new_bbox)

    if landmarks_68 is not None:
        return aligned,new_landmark,new_landmark_68,new_bbox
    else:
        return aligned,new_landmark,new_bbox

ARCFACE_SRC = np.array([[
    [122.5, 141.25],
    [197.5, 141.25],
    [160.0, 178.75],
    [137.5, 225.25],
    [182.5, 225.25]
]], dtype=np.float32)



def estimate_norm(lmk,image_size,zoom_in=1):
    lmk = np.array(lmk)
    assert lmk.shape == (5, 2)
    tform = SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = np.inf
    src = ARCFACE_SRC*(image_size/320)
    src=src/zoom_in-image_size/2*(1/zoom_in-1)
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
    M = tform.params[0:2, :]
    results = np.dot(M, lmk_tran.T)
    results = results.T
    error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
    if error < min_error:
        min_error = error
        min_M = M
        min_index = i
    return min_M, min_index

def norm_crop(img, landmark, image_size=320,zoom_in=1):
    M, pose_index = estimate_norm(landmark,image_size,zoom_in)
    warped = cv2.warpAffine(img, M, (image_size, image_size),flags=cv2.INTER_CUBIC, borderValue=0.0)
    return warped, M


def align_face(image_array, landmarks, bboxes):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye_center = landmarks[0]
    right_eye_center = landmarks[1]
    # calculate the mean point of landmarks of left and right eye
    # left_eye_center = np.mean(left_eye, axis=0).astype("int")
    # right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    landmarks = rotate_landmarks(landmarks,eye_center,angle,rotated_img.shape[0])
    bboxes = bbox_rotate(bboxes,rotate_matrix,(image_array.shape[1], image_array.shape[0]))
    return rotated_img, landmarks,bboxes

def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)

def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = []
    for i in range(landmarks.shape[0]):
        rotated_landmark = rotate(origin=eye_center, point=landmarks[i], angle=angle, row=row)
        rotated_landmarks.append(rotated_landmark)
    return rotated_landmarks

def bbox_rotate(bbox, M, img_shape):
    """Flip bboxes horizontally.
    Args:
        bbox(list): [left, right, up, down]
        img_shape(tuple): (height, width)
    """
    assert len(bbox) == 4
    a = M[:, :2]  ##a.shape (2,2)
    b = M[:, 2:]  ###b.shape(2,1)
    b = np.reshape(b, newshape=(1, 2))
    a = np.transpose(a)

    # [left, right, up, down] = bbox
    [left, up, right, down] = bbox
    corner_point = np.array([[left, up], [right, up], [left, down], [right, down]])
    corner_point = np.dot(corner_point, a) + b
    min_left = max(int(np.min(corner_point[:, 0])), 0)
    max_right = min(int(np.max(corner_point[:, 0])), img_shape[1])
    min_up = max(int(np.min(corner_point[:, 1])), 0)
    max_down = min(int(np.max(corner_point[:, 1])), img_shape[0])

    return [min_left, min_up, max_right, max_down]



def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def crop_face(img,landmark=None,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
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
    
    if bbox is not None:
        bbox_cropped=np.zeros_like(bbox)
        for i,(p,q) in enumerate(bbox):
            bbox_cropped[i]=[p-x0_new,q-y0_new]
    else:
        bbox_cropped=None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
    else:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)

class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self,img,**params):
        return self.randomdownscale(img)

    def randomdownscale(self,img):
        keep_ratio=True
        keep_input_shape=True
        H,W,C=img.shape
        ratio_list=[2,4]
        r=ratio_list[np.random.randint(len(ratio_list))]
        img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

        return img_ds
