import numpy as np
import torch
import torch.nn.functional as F

def dissimilarity_batch(x1, x2):
    """
    Calculate differentiable dissimilarity between batches of feature vectors
    More efficient than pairwise computation
    """
    # Normalize vectors for cosine similarity
    x1_norm = F.normalize(x1, p=2, dim=-1)
    x2_norm = F.normalize(x2, p=2, dim=-1)
    
    # Compute cosine similarity and convert to dissimilarity
    cos_sim = torch.sum(x1_norm * x2_norm, dim=-1)
    return 1.0 - cos_sim

def div_reward_optimized(actions, features, temporal_distance=20):
    """
    Optimized diversity reward with vectorized operations
    """
    # Get selected indices
    selected_mask = actions.squeeze() == 1
    selected_indices = torch.nonzero(selected_mask, as_tuple=True)[0]
    
    if len(selected_indices) <= 1:
        return torch.tensor(0.0, device=features.device)
    
    Y = features[selected_indices]  # Selected features
    n_selected = len(Y)
    
    # Create pairwise distance matrix using broadcasting
    # Y.unsqueeze(0) -> (1, n_selected, feature_dim)
    # Y.unsqueeze(1) -> (n_selected, 1, feature_dim)
    Y_expanded_1 = Y.unsqueeze(0)  # (1, n_selected, feature_dim)
    Y_expanded_2 = Y.unsqueeze(1)  # (n_selected, 1, feature_dim)
    
    # Compute all pairwise dissimilarities at once
    dissim_matrix = dissimilarity_batch(Y_expanded_1, Y_expanded_2)  # (n_selected, n_selected)
    
    # Create temporal distance mask
    indices_expanded_1 = selected_indices.unsqueeze(0)  # (1, n_selected)
    indices_expanded_2 = selected_indices.unsqueeze(1)  # (n_selected, 1)
    temporal_diff = torch.abs(indices_expanded_1 - indices_expanded_2)
    
    # Mask for temporal constraint and self-pairs
    temporal_mask = temporal_diff > temporal_distance
    self_mask = torch.eye(n_selected, device=features.device).bool()
    
    # Apply masks: use dissimilarity for valid pairs, 1.0 for temporal pairs, 0 for self
    final_matrix = torch.where(self_mask, 
                              torch.zeros_like(dissim_matrix),
                              torch.where(temporal_mask, dissim_matrix, torch.ones_like(dissim_matrix)))
    
    # Sum all non-diagonal elements
    sum_distances = final_matrix.sum() - final_matrix.diagonal().sum()
    num_pairs = n_selected * (n_selected - 1)
    
    return sum_distances / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=features.device)

def rep_reward_optimized(actions, features):
    """
    Optimized repetition reward with vectorized operations
    """
    selected_mask = actions.squeeze() == 1
    selected_features = features[selected_mask]
    
    if len(selected_features) == 0:
        return torch.tensor(0.0, device=features.device)
    
    T = len(features)
    
    # Vectorized distance computation
    # features: (T, feature_dim)
    # selected_features: (n_selected, feature_dim)
    # Use broadcasting to compute all distances at once
    features_expanded = features.unsqueeze(1)  # (T, 1, feature_dim)
    selected_expanded = selected_features.unsqueeze(0)  # (1, n_selected, feature_dim)
    
    # Compute all pairwise L2 distances
    distances = torch.linalg.norm(features_expanded - selected_expanded, dim=2)  # (T, n_selected)
    
    # Get minimum distance for each frame
    min_distances = torch.min(distances, dim=1)[0]  # (T,)
    
    # Calculate final reward
    sum_min_distances = torch.sum(min_distances)
    r_rep = torch.exp(-sum_min_distances / T)
    
    return r_rep

# Even faster versions for when you need maximum speed
def div_reward_fast(actions, features, temporal_distance=20):
    """
    Ultra-fast diversity reward with approximations
    """
    selected_mask = actions.squeeze() == 1
    selected_indices = torch.nonzero(selected_mask, as_tuple=True)[0]
    
    if len(selected_indices) <= 1:
        return torch.tensor(0.0, device=features.device)
    
    Y = features[selected_indices]
    n_selected = len(Y)
    
    # Use matrix multiplication for cosine similarity (much faster)
    Y_norm = F.normalize(Y, p=2, dim=1)
    cos_sim_matrix = torch.mm(Y_norm, Y_norm.t())
    dissim_matrix = 1.0 - cos_sim_matrix
    
    # Simple temporal masking
    indices_diff = selected_indices[:, None] - selected_indices[None, :]
    temporal_mask = torch.abs(indices_diff) > temporal_distance
    
    # Apply masks and compute mean
    valid_mask = temporal_mask & ~torch.eye(n_selected, device=features.device, dtype=bool)
    
    if valid_mask.sum() == 0:
        return torch.tensor(1.0, device=features.device)  # All pairs are temporal
    
    return dissim_matrix[valid_mask].mean()

def rep_reward_fast(actions, features):
    """
    Ultra-fast repetition reward
    """
    selected_mask = actions.squeeze() == 1
    selected_features = features[selected_mask]
    
    if len(selected_features) == 0:
        return torch.tensor(0.0, device=features.device)
    
    T = len(features)
    
    # Use cdist for efficient pairwise distance computation
    distances = torch.cdist(features.unsqueeze(0), selected_features.unsqueeze(0)).squeeze(0)
    min_distances = torch.min(distances, dim=1)[0]
    
    return torch.exp(-min_distances.mean())

# Convenience function to choose the right version
def get_reward_functions(mode='balanced'):
    """
    Get reward functions based on speed/accuracy trade-off
    
    Args:
        mode: 'accurate' (original), 'balanced' (optimized), 'fast' (approximated)
    """
    if mode == 'fast':
        return div_reward_fast, rep_reward_fast
    elif mode == 'balanced':
        return div_reward_optimized, rep_reward_optimized
    else:
        # Original functions for reference
        return div_reward, rep_reward