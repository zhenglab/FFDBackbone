import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_random_frames(video_num, video_path, output_dir, num_frames=1):
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate five random frame numbers
    frame_nums = np.sort(np.random.randint(0, total_frames, num_frames))

    for frame_num in frame_nums:
        # Set the position of the video file to the frame number we want to capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        ret, frame = cap.read()
        
        # If the frame was read successfully, save it
        if ret:
            output_dir_ = os.path.join(output_dir, f'{video_num}')
            os.makedirs(output_dir_, exist_ok=True)
            output_path = os.path.join(output_dir_, f'{frame_num}.png')
            print(output_path)
            cv2.imwrite(output_path, frame)
            
    cap.release()

def main(input_dir, output_dir):
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
 
    # Iterate over all files in the input directory
    print(len(os.listdir(input_dir)))
    video_num = 0
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            try: 
                extract_random_frames(video_num, video_path, output_dir, num_frames=num_frames)
                video_num +=1
            except:
                print(filename)


if __name__ == "__main__":
    num_frames = 8
    # methods = ['Celeb-DF', 'DFDC', 'DiffHead', 'FF', 'FFIW', 'hrfae', 'iplap', 'makeittalker', 'mobileswap', 'sadtalker', 'styleHEAT', 'VIPL', 'wav2lip']
    methods = [  'DFDC']
    
    for method in methods:
        main(f'/SSD0/guozonghui/project/FFD/Real_video/Video/{method}', \
            f'/SSD0/guozonghui/project/FFD/Real_video/Frame/{method}')