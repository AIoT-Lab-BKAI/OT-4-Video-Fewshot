import os
import glob
import cv2
from argparse import ArgumentParser

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_src', required=True, type=str, help='path to folder containe class data ex: ApplyEyeMakeup')
    args = parser.parse_args()
    
    dest_dir = "data/ucf101"
    
    class_names = os.listdir(args.data_src)
    class_names.sort()
    class_names = [class_name for class_name in class_names if class_name != ".DS_Store"]
    class_paths = [os.path.join(args.data_src, class_name) for class_name in class_names]
    for class_name in class_names:
        os.makedirs(os.path.join(dest_dir, class_name), exist_ok=True)
    
    for class_path in class_paths:
        class_name = os.path.basename(class_path)
        video_paths = glob.glob(os.path.join(class_path, "*.avi"))
        for video_path in video_paths:
            video_frames = read_video(video_path)
            video_name = os.path.basename(video_path).replace(".avi", "")
            save_frame_dir = os.path.join(dest_dir, class_name, video_name)
            os.makedirs(save_frame_dir, exist_ok=True)
            for i, frame in enumerate(video_frames):
                # frame id has 5 digits
                frame_id = str(i).zfill(5)
                cv2.imwrite(os.path.join(save_frame_dir, f"{frame_id}.jpg"), frame)

if __name__ == '__main__':
    main()