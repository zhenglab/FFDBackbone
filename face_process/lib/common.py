import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from lib.ct.detection.utils import grab_all_frames, get_valid_faces, sample_chunks,grab_spase_frames
from lib.ct.operations import multiple_tracking
import numpy as np
from lib.ct.face_alignment import LandmarkPredictor
from lib.ct.detection import FaceDetector
import cv2
from lib.utils import flatten,partition
from tqdm import tqdm
from imutils import face_utils

detector = FaceDetector(0)
predictor = LandmarkPredictor(0)


def get_five(ldm68):
    groups = [range(36, 42), range(42, 48), [30], [48], [54]]
    points = []
    for group in groups:
        points.append(ldm68[group].mean(0))
    return np.array(points)


def get_bbox(mask):
    try:
        y, x = np.nonzero(mask[..., 0])
        return x.min() - 1, y.min() - 1, x.max() + 1, y.max() + 1
    except:
        return None


def get_bigger_box(image, box, scale=0.5):
    height, width = image.shape[:2]
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


def process_bigger_clips(clips, dete_res, clip_size, step, scale=0.5):
    assert len(clips) % clip_size == 0
    detect_results = sample_chunks(dete_res, clip_size, step)
    clips = sample_chunks(clips, clip_size, step)
    new_clips = []
    for i, (frame_clip, record_clip) in enumerate(zip(clips, detect_results)):
        tracks = multiple_tracking(record_clip)
        for j, track in enumerate(tracks):
            new_images = []
            for (box, ldm, _), frame in zip(track, frame_clip):
                big_box = get_bigger_box(frame, box, scale)
                x1, y1, x2, y2 = big_box
                top_left = big_box[:2][None, :]
                new_ldm5 = ldm - top_left
                box = np.rint(box).astype(np.int)
                new_box = (box.reshape(2, 2) - top_left).reshape(-1)
                feed = LandmarkPredictor.prepare_feed(frame, box)
                ldm68 = predictor(feed) - top_left
                new_images.append(
                    (frame[y1:y2, x1:x2], big_box, new_box, new_ldm5, ldm68)
                )
            new_clips.append(new_images)
    return new_clips


def post(detected_faces):
    return [[face[:4], None, face[-1]] for face in detected_faces]


def check(detect_res):
    return min([len(faces) for faces in detect_res]) != 0

import dlib
dlib.DLIB_USE_CUDA
def detect_frames_from_video(file, sfd_only=False, return_frames=False, max_size=None, n_frames=16, lm68=False, lm81=False):
    frames,frames_ids = grab_spase_frames(file, max_size=max_size, cvt=False, n_frames=n_frames)
    if not sfd_only:
        detect_res = flatten(
            [detector.detect(item) for item in partition(frames, 256)]
        )
        detect_res = get_valid_faces(detect_res, thres=0.5)
    else:
        raise NotImplementedError
    all_68 = None
    if lm68:
        all_68 = get_lm68(frames, detect_res)
    all_lm81 = None
    if lm81:
        face_detector = None
        predictor_path = './lib/shape_predictor_81_face_landmarks.dat'
        face_predictor = dlib.shape_predictor(predictor_path)
        all_lm81 = generate_landmark_81(face_detector, face_predictor, frames, detect_res,file)
    if not return_frames:
        return detect_res, all_68
    else:
        return detect_res, all_68, frames, all_lm81, frames_ids


def detect_frames(frame_list, path=False, return_frames=False, lm68=False, lm81=False):
    # frames = grab_all_frames(file, max_size=max_size, cvt=True)
    frames = []
    for frame_name in frame_list:
        frame=cv2.imread(os.path.join(path,frame_name))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    detect_res = flatten(
        [detector.detect(item) for item in partition(frames, 256)]
    )
    detect_res = get_valid_faces(detect_res, thres=0.5)

    all_68 = None
    if lm68:
        all_68 = get_lm68(frames, detect_res)
    all_lm81 = None
    if lm81:
        face_detector = None
        predictor_path = './lib/shape_predictor_81_face_landmarks.dat'
        face_predictor = dlib.shape_predictor(predictor_path)
        all_lm81 = generate_landmark_81(face_detector, face_predictor, frames, detect_res,path)
    if not return_frames:
        return detect_res, all_68
    else:
        return detect_res, all_68, frames, all_lm81, frame_list

def get_lm68(frames, detect_res):
    assert len(frames) == len(detect_res)
    frame_count = len(frames)
    all_68 = []
    for i in range(frame_count):
        frame = frames[i]
        faces = detect_res[i]
        if len(faces) == 0:
            res_68 = []
        else:
            feeds = []
            for face in faces:
                assert len(face) == 3
                box = face[0]
                feed = LandmarkPredictor.prepare_feed(frame, box)
                feeds.append(feed)
            res_68 = predictor(feeds)
            assert len(res_68) == len(faces)
            for face, l_68 in zip(faces, res_68):
                if face[1] is None:
                    face[1] = get_five(l_68)
        all_68.append(res_68)

    assert len(all_68) == len(detect_res)
    return all_68

def generate_landmark_81(face_detector, face_predictor, frames, detect_res, org_path):
    video_landmarks = []
    all_81 = []
    assert len(frames) == len(detect_res)
    frame_count = len(frames)
    for i in range(frame_count):
        frame = frames[i]
        faces = detect_res[i]
        if len(faces) == 0:
            res_81 = []
        else:
            res_81 = []
            for face in faces:
                bbox = face[0]
                bbox_rectangle = dlib.rectangle(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
                landmark = face_predictor(frame, bbox_rectangle)
                landmark = face_utils.shape_to_np(landmark)
                res_81.append(landmark)
            assert len(res_81) == len(faces)
            for face, l_68 in zip(faces, res_81):
                if face[1] is None:
                    face[1] = get_five(l_68)
        all_81.append(res_81)

    assert len(all_81) == len(detect_res)
    return all_81
    
    
    for cnt_frame, frame in enumerate(frames): 
        # faces = face_detector(frame, 1)
        faces = detect_res[cnt_frame]
        if len(faces)==0:
            tqdm.write('No faces in {}:{}'.format(cnt_frame,os.path.basename(org_path)))
            continue
        face_s_max=-1
        landmarks=[]
        size_list=[]
        for face_idx in range(len(faces)):
            bbox = faces[face_idx][0]
            bbox_rectangle = dlib.rectangle(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
            # faces_tmp = dlib.rectangles()
            # faces_tmp.append(bbox_rectangle)
            landmark = face_predictor(frame, bbox_rectangle)
            
            landmark = face_utils.shape_to_np(landmark)
            x0,y0=landmark[:,0].min(),landmark[:,1].min()
            x1,y1=landmark[:,0].max(),landmark[:,1].max()
            face_s=(x1-x0)*(y1-y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
        landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]
        # video_landmarks.append((cnt_frame,landmarks))
        video_landmarks.append(landmarks)
    return video_landmarks


def detect_all(file, sfd_only=False, return_frames=False, max_size=None):
    frames = grab_all_frames(file, max_size=max_size, cvt=True)
    if not sfd_only:
        detect_res = flatten(
            [detector.detect(item) for item in partition(frames, 256)]
        )
        detect_res = get_valid_faces(detect_res, thres=0.5)
    else:
        raise NotImplementedError

    all_68 = get_lm68(frames, detect_res)
    if not return_frames:
        return detect_res, all_68
    else:
        return detect_res, all_68, frames
