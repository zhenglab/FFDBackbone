import cv2
import numpy as np

def write_img(file, img):
    cv2.imwrite(file, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
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