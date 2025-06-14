import argparse
from extract_feature import extract_video_features, read_datas
import torch
from models import DSN
from KTS import Kernel_temporal_segmentation as KTS
import h5py
import numpy as np
from Knapsack import Knapsack
from summary2video import create_summary_video
import os
from evaluate import evaluate_summary
import scipy 


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
    parser.add_argument('--user_summary', type=str, default=None,
                        help='Path to user-provided summary for evaluation (optional)')
    parser.add_argument('--save_results', type=str, default='results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    if args.extract_features:
        extract_video_features(args.input_dir, args.output_extractor)
        
    datas = read_datas(args.output_extractor)
    # Temporal Segmentation using KTS
    list_change_points = []
    for data in datas:
        list_change_points.append(KTS(data['Features'], max_change_points=len(data['Features'])//2, penalty_factor=0.05))
    
    model = DSN()
    model.load_state_dict(torch.load(args.pretrained_model, weights_only=True))
    
    summaries_folder = os.path.dirname(args.save_summary)
    if not os.path.exists(summaries_folder):
        os.makedirs(summaries_folder)
    
    hd5 = h5py.File(args.save_summary, 'w')
    for i, data in enumerate(datas):
        change_points = list_change_points[i]
        video_name = data['Video_Name']
        features = np.array(data['Features'])
        features = torch.tensor(features, dtype=torch.float32)
        frame_ids = np.array(data['Choosen_Frame_IDs'])
        n_frames_original = data['N_FRAMES'][0]
        
        
        
        importance_scores = model.forward(features)
        importance_scores = importance_scores.detach().numpy()
        
        last = 0
        seg_weights = []
        seg_values = []
        for cp in change_points:
            seg_weights.append(frame_ids[cp] - frame_ids[last])
            seg_values.append(np.mean(importance_scores[last:cp]))
            last = cp
        
        # print('Original number of frames:', n_frames_original)
        last = 0
        
        _, segments = Knapsack(seg_weights, seg_values, int(n_frames_original * 0.15))
        # print(f"Selected segments for {video_name}: {segments}")
        
        frame_state = np.zeros((n_frames_original), dtype=bool)
        for seg_id in segments:
            if seg_id == 0:
                start = 0
            else:
                start = change_points[seg_id - 1]
            end = change_points[seg_id]
            
            start = frame_ids[start]
            end = frame_ids[end] - 1
            
            frame_state[start:end + 1] = True
        print(f"% of frames selected for {video_name}: {np.sum(frame_state) / n_frames_original * 100:.2f}%")
        print(f"Number of frames selected for {video_name}: {np.sum(frame_state)} / {n_frames_original}")
        hd5.create_dataset(f"{video_name}", data=frame_state, compression='gzip')
    hd5.close()
    
    create_summary_video(args.input_dir, args.save_summary, args.save_video)
    print("Summary videos created successfully.")
    
    summaries = {}
    with h5py.File(args.save_summary, 'r') as hf:
        for video_name in hf.keys():
            summaries[video_name] = hf[video_name][:]
    print(f"Summaries extracted for {len(summaries)} videos.")
    
    # Read user-provided summary
    f_scores = []
    precisions = []
    recalls = []
    if args.user_summary is None:
        print("No user summary provided for evaluation. Skipping evaluation.")
        exit(0)
    for filename in summaries.keys():
        path_to_user_summary = os.path.join(args.user_summary, f"{filename}.mat")
        if not os.path.exists(path_to_user_summary):
            print(f"User summary for {filename} not found. Skipping evaluation.")
            continue
        data = scipy.io.loadmat(path_to_user_summary)
        user_score = data.get('user_score').T
        f_score, precision, recall = evaluate_summary(summaries[filename], user_score)
        f_scores.append(f_score)
        precisions.append(precision)
        recalls.append(recall)
    print(f"Average F-score: {np.mean(f_scores):.3f}")
    print(f"Average Precision: {np.mean(precisions):.3f}")
    print(f"Average Recall: {np.mean(recalls):.3f}")
    print("Evaluation completed.")
    
    save_results_folder = os.path.dirname(args.save_results)
    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)
    
    with open(args.save_results, 'w') as f:
        f.write(f"Average F-score: {np.mean(f_scores):.3f}\n")
        f.write(f"Average Precision: {np.mean(precisions):.3f}\n")
        f.write(f"Average Recall: {np.mean(recalls):.3f}\n")
        