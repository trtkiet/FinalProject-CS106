import numpy as np
import cv2
import h5py
from tqdm import tqdm
import os

def create_summary_video(source_video_folder, videos_summaries_path, output_folder):
    dict_frame_states = {}
    with h5py.File(videos_summaries_path, 'r') as hf:
        for video_name in hf.keys():
            frame_state = hf[video_name][:]
            dict_frame_states[video_name] = frame_state
            
    # Loop through each video in source_video_folder
    for filename in os.listdir(source_video_folder):
        if not filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
            continue
        
        video_path = os.path.join(source_video_folder, filename)
        video_name = os.path.splitext(filename)[0]
        if video_name not in dict_frame_states:
            print(f"Skipping {video_name} as it has no summary data.")
            continue
        frame_state = dict_frame_states[video_name]
        print(frame_state)
        output_video_path = os.path.join(output_folder, f"{video_name}_summary.mp4")
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}. Skipping.")
            continue
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Check if the frame is selected in the summary
            if frame_state[frame_count]:
                out.write(frame)
            frame_count += 1
        cap.release()
        out.release()
    