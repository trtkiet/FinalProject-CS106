from torchvision.models import googlenet, GoogLeNet_Weights
import torchvision.transforms as transforms
from torch import nn
import torch
import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

def extract_frames(video_path, fps=2):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(round(orig_fps / fps)) if orig_fps > 0 else 15
    frames = []
    frame_idx = 0
    frame_ids = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            frames.append(img)
            frame_ids.append(frame_idx)
        frame_idx += 1
    cap.release()
    return orig_fps, frames, frame_idx, frame_ids


def extract_features(frames):
    model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
    model.eval()  # Set to evaluation mode
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # GoogLeNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    features = []
    for img in frames:
        input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = feature_extractor(input_tensor)
        features.append(output.squeeze().numpy())
    return features
        
def extract_video_features(video_dir, output_path, fps=2):
    with h5py.File(output_path, 'w') as h5f:
        for filename in tqdm(os.listdir(video_dir)):
            if not filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
                continue

            video_path = os.path.join(video_dir, filename)
            video_name = os.path.splitext(filename)[0]
            
            data = {}

            orig_fps, frames, total_frames, frame_ids = extract_frames(video_path, fps=fps)
            if not frames:
                continue

            features = extract_features(frames)  # List of np.arrays
            
            data['Features'] = features
            data['N_FPS_OG'] = [orig_fps]
            data['N_FPS_EXTRACTED'] = [fps]
            data['N_FRAMES'] = [total_frames]
            data['Choosen_Frame_IDs'] = frame_ids
            

            h5f.create_group(video_name)
            for key, value in data.items():
                h5f[video_name].create_dataset(key, data=value)

def read_datas(features_path):
    datas = []
    with h5py.File(features_path, 'r') as h5f:
        for video_name in h5f.keys():
            data = {'Video_Name': video_name}
            for key in h5f[video_name].keys():
                data[key] = h5f[video_name][key][:]
            datas.append(data)
    print(f"Read {len(datas)} video datas from {features_path}.")
    return datas
    
def main():
    video_dir = 'input_extractor'
    output_path = 'output_extractor/video_features.h5'
    extract_video_features(video_dir, output_path)
    
    features = {}
    with h5py.File(output_path, 'r') as h5f:
        features = {k: v[:] for k, v in h5f.items()}
    print(f"Extracted features for {len(features)} videos.")
    
if __name__ == "__main__":
    main()