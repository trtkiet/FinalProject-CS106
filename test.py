import argparse
from extract_feature import extract_video_features, read_features
import torch
from models import DSN
from KTS import Kernel_temporal_segmentation as KTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test script for feature extraction and model testing')
    
    parser.add_argument('-e', '--extract_features', action='store_true',
                        help='Extract features from videos')
    parser.add_argument('-inp', '--input_dir', type=str, default='input_extractor',
                        help='Directory containing input videos')
    parser.add_argument('-ft', '--output_extractor', type=str, default='output_extractor/video_features.h5',
                        help='Path to save the extracted features')
    parser.add_argument('--pretrained_model', type=str, default='weights/model.pth',
                        help='Path to the pretrained model weights')
    parser.add_argument('--save_summary', type=str, default='output_extractor/summary.h5',
                        help='Path to save the summary of selected frames')
    
    args = parser.parse_args()
    
    if args.extract_features:
        extract_video_features(args.input_dir, args.output_extractor)
        
    features = read_features(args.output_extractor)
    # Temporal Segmentation using KTS
    for feature in features:
        change_points = KTS(feature)
        continue
    exit(0)
    
    model = DSN()
    model.load_state_dict(torch.load(args.pretrained_model))
    
    features = torch.tensor(features, dtype=torch.float32)
    frame_importance = model.forward(features)
    frame_importance = frame_importance.squeeze().cpu().numpy()
    
    
    