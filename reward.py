import numpy as np
import torch

def dissimilarity(x_t, x_t_prime):
    """
    Calculate differentiable dissimilarity between two feature vectors
    """
    # Calculate dot product
    dot_product = torch.dot(x_t, x_t_prime)
    # Calculate L2 norms
    norm_t = torch.linalg.norm(x_t)
    norm_t_prime = torch.linalg.norm(x_t_prime)
    # Calculate cosine similarity and convert to dissimilarity
    return 1.0 - (dot_product / (norm_t * norm_t_prime))

def div_reward(actions, features, temporal_distance=20):
    """
    Calculate differentiable diversity reward
    """
    Y = features[actions.squeeze() == 1]  # Select features corresponding to selected frames
    
    if len(Y) <= 1:  # Need at least 2 frames to calculate diversity
        return torch.tensor(0.0, device=features.device)
    
    sum_distances = torch.tensor(0.0, device=features.device)
    for t in range(len(Y)):
        for t_prime in range(len(Y)):
            if abs(t - t_prime) <= temporal_distance:
                sum_distances += 1.0
                continue
            if t != t_prime:  # Skip when t' = t
                sum_distances += dissimilarity(Y[t], Y[t_prime])
    
    # Normalize by number of pairs
    num_pairs = len(Y) * (len(Y) - 1)
    r_div = sum_distances / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=features.device)
    
    return r_div

def rep_reward(actions, features):
    """
    Calculate differentiable repetition reward
    """
    T = len(features)
    selected_features = features[actions.squeeze() == 1]  # Y in the formula
    
    if len(selected_features) == 0:
        return torch.tensor(0.0, device=features.device)
        
    sum_min_distances = torch.tensor(0.0, device=features.device)
    for t in range(T):
        # Calculate distances between current frame and all selected frames
        distances = torch.linalg.norm(features[t].unsqueeze(0) - selected_features, dim=1)
        # Get minimum distance
        min_distance = torch.min(distances)
        sum_min_distances += min_distance
    
    # Calculate final reward
    r_rep = torch.exp(-sum_min_distances / T)
    
    return r_rep
    
        