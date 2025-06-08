import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from reward import div_reward_optimized, rep_reward_optimized
from tqdm import tqdm
import time

class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1):
        super(DSN, self).__init__()
        self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)
    
    def forward(self, x):
        h, _ = self.rnn(x)
        p = torch.sigmoid(self.fc(h))
        return p
    
    def train(self, videos_features, epochs=60, early_stopping=10, alpha=20, epsilon=0.5, episodes=5, 
              lr=1e-4, beta=0.01, weight_decay=1e-5, device='cpu', batch_size=4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        print("Training model...")
        baseline = torch.zeros(len(videos_features), dtype=torch.float32, device=device)
        epoch_rewards = []
        print(len(videos_features), "videos for training.")
        
        for epoch in tqdm(range(epochs), desc="Epochs"):
            idxs = torch.randperm(len(videos_features))
            sum_reward = 0.0
            
            # Process in batches for better efficiency
            for batch_start in range(0, len(videos_features), batch_size):
                batch_end = min(batch_start + batch_size, len(videos_features))
                batch_idxs = idxs[batch_start:batch_end]
                
                batch_cost = 0.0
                batch_reward = 0.0
                
                for idx in batch_idxs:
                    video_features = videos_features[idx]
                    video_features = torch.tensor(video_features, dtype=torch.float32).to(device)
                    
                    p = self(video_features)
                    cost = beta * (p.mean() - epsilon) ** 2
                    
                    # Reduced episodes for faster training
                    bernouli = Bernoulli(p)
                    episode_rewards = []
                    
                    for _ in range(episodes):
                        actions = bernouli.sample()
                        log_p = bernouli.log_prob(actions)
                        reward = div_reward_optimized(actions, video_features, alpha) + rep_reward_optimized(actions, video_features)
                        expected_reward = log_p.mean() - (reward - baseline[idx])
                        cost -= expected_reward 
                        episode_rewards.append(reward.item())
                    
                    avg_reward = sum(episode_rewards) / episodes
                    batch_reward += avg_reward
                    batch_cost += cost
                    
                    # Update baseline
                    with torch.no_grad():
                        baseline[idx] = 0.9 * baseline[idx] + 0.1 * avg_reward
                # print(f"Batch {batch_start // batch_size + 1}, Batch Reward: {batch_reward:.4f}, Batch Cost: {batch_cost.item():.4f}")
                # Batch optimization
                optimizer.zero_grad()
                batch_cost.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                optimizer.step()
                
                sum_reward += batch_reward
            
            epoch_reward = sum_reward / len(videos_features)
            epoch_rewards.append(epoch_reward)
            
            # Less frequent printing
            if epoch % 1 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}, Reward: {epoch_reward:.4f}")
            
            # Early stopping
            if epoch > early_stopping and epoch_reward - min(epoch_rewards[-early_stopping:]) < 1e-5:
                print(f"Early stopping at epoch {epoch+1}")
                break