import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from reward import div_reward, rep_reward
from tqdm import tqdm

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
    
    def train(self, videos_features, epochs=60, early_stopping=10, alpha=20, epsilon=0.5, episodes=5, lr=1e-5, beta=0.01, weight_decay=1e-5, device='cpu'):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        if device == 'cuda':
            self = nn.DataParallel(self).cuda()
        
        print("Training model...")
        baseline = torch.zeros(len(videos_features), dtype=torch.float32, device=device)
        epoch_rewards = []
        for epoch in tqdm(range(epochs)):
            idxs = torch.randperm(len(videos_features))
            sum_reward = 0.0
            for idx in idxs:
                video_features = videos_features[idx]
                video_features = torch.tensor(video_features, dtype=torch.float32).to(device)
                n = video_features.shape[0]
                
                p = self(video_features)
                
                cost = beta * (p.mean() - epsilon) ** 2
                bernouli = Bernoulli(p)
                rewards = []    
                for _ in range(episodes):
                    actions = bernouli.sample()
                    log_p = bernouli.log_prob(actions)
                    reward = div_reward(actions, video_features, alpha) + rep_reward(actions, video_features)
                    expected_reward = log_p.mean() - (reward - baseline[idx])
                    cost -= expected_reward 
                    rewards.append(reward.item())
                    sum_reward += reward / episodes
                    
                optimizer.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                optimizer.step()
                with torch.no_grad():
                    baseline[idx] = 0.9 * baseline[idx] + 0.1 * reward.mean()                
            epoch_reward = sum_reward.item() / len(videos_features)
            epoch_rewards.append(epoch_reward)
            if epoch > early_stopping and epoch_reward - min(epoch_rewards[-early_stopping:]) < 1e-5:
                print(f"Early stopping at epoch {epoch+1}")
                break
            print(f"Epoch {epoch+1}/{epochs}, Reward: {epoch_reward:.4f}")