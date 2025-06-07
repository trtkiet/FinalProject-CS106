import argparse
from extract_feature import extract_video_features, read_datas
import torch
from models import DSN
from KTS import Kernel_temporal_segmentation as KTS
import h5py
import numpy as np
from Knapsack import Knapsack
from summary2video import create_summary_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test script for feature extraction and model testing')
    
    parser.add_argument('-e', '--extract_features', action='store_true',
                        help='Extract features from videos')
    parser.add_argument('-i', '--input_dir', type=str, default='input_extractor',
                        help='Directory containing input videos')
    parser.add_argument('-ft', '--output_extractor', type=str, default='output_extractor/video_features.h5',
                        help='Path to save the extracted features')
    parser.add_argument('--pretrained_model', type=str, default='weights/model.pth',
                        help='Path to the pretrained model weights')
    parser.add_argument('-o', '--save_summary', type=str, default='summaries/summary.h5',
                        help='Path to save the summary of selected frames')
    parser.add_argument('-v', '--save_video', type=str, default='output_videos',
                        help='Directory to save the summary videos')
    
    args = parser.parse_args()
    
    if args.extract_features:
        extract_video_features(args.input_dir, args.output_extractor)
        
    datas = read_datas(args.output_extractor)
    print()
    # Temporal Segmentation using KTS
    list_change_points = []
    for data in datas:
        print(data)
        list_change_points.append(KTS(data['Features'], max_change_points=20, penalty_factor=0.05))
    
    model = DSN()
    model.load_state_dict(torch.load(args.pretrained_model, weights_only=True))
    
    hd5 = h5py.File(args.save_summary, 'w')
    for i, data in enumerate(datas):
        change_points = list_change_points[i]
        video_name = data['Video_Name']
        features = np.array(data['Features'])
        features = torch.tensor(features, dtype=torch.float32)
        frame_ids = np.array(data['Choosen_Frame_IDs'])
        n_frames = data['N_FRAMES'][0]
        
        importance_scores = model.forward(features)
        importance_scores = importance_scores.detach().numpy()
        
        last = 0
        seg_weights = []
        seg_values = []
        for cp in change_points:
            seg_weights.append(cp - last)
            seg_values.append(np.mean(importance_scores[last:cp]))
            last = cp
        
        _, segments = Knapsack(seg_weights, seg_values, int(features.shape[0] * 0.15))
        print(f"Selected segments for {video_name}: {segments}")
        
        frame_state = np.zeros((n_frames), dtype=bool)
        for seg_id in segments:
            if seg_id == 0:
                start = 0
            else:
                start = change_points[seg_id - 1]
            end = change_points[seg_id] - 1
            print('Segment:', start, end)
            start = frame_ids[start]
            end = frame_ids[end]
            print('Selected segment:', start, end)
            frame_state[start:end + 1] = True
        hd5.create_dataset(f"{video_name}", data=frame_state, compression='gzip')
    hd5.close()
    
    create_summary_video(args.input_dir, args.save_summary, args.save_video)
    print("Summary videos created successfully.")