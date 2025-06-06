from extract_feature import extract_video_features, read_features
import argparse
from models import DSN
import torch 
import h5py
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the feature extractor')
    
    parser.add_argument('-e', '--extract_features', action='store_true',
                        help='Extract features from videos')
    parser.add_argument('--input_dir', type=str, default='input_extractor',
                        help='Directory containing input videos')
    parser.add_argument('--output_extractor', type=str, default='output_extractor/video_features.h5',
                        help='Path to save the extracted features')
    parser.add_argument('--save_model', type=str, default='weights/model.pth',
                        help='Path to save the trained model')
    args = parser.parse_args()
    extract_features = args.extract_features
    input_dir = args.input_dir
    output_extractor = args.output_extractor
    
    if extract_features:
        extract_video_features(input_dir, output_extractor)
    
    features = read_features(output_extractor)
    model = DSN()
    
    model.train(features)  # Set the model to training mode
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, args.save_model)