from extract_feature import extract_video_features, read_datas
import argparse
from models import DSN
import torch 
import h5py
import os
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the feature extractor')
    
    parser.add_argument('-e', '--extract_features', action='store_true',
                        help='Extract features from videos')
    parser.add_argument('-i', '--input_dir', type=str, default='input_folder',
                        help='Directory containing input videos')
    parser.add_argument('-o', '--output_extractor', type=str, default='features_folder/video_features.h5',
                        help='Path to save the extracted features')
    parser.add_argument('--save_model', type=str, default='weights/model.pth',
                        help='Path to save the trained model')
    args = parser.parse_args()
    extract_features = args.extract_features
    input_dir = args.input_dir
    output_extractor = args.output_extractor
    
    if extract_features:
        extract_video_features(input_dir, output_extractor)
        
    if os.path.exists(args.save_model):
        print(f"Model already exists at {args.save_model}. Skipping training.")
        exit(0)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datas = read_datas(output_extractor)
    features = []
    for data in datas:
        features.append(data['Features'])
    model = DSN().to(device)
    
    model.train(features, device='cuda')  # Set the model to training mode
    output_folder = os.path.dirname(args.save_model)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, args.save_model)