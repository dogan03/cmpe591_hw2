import random
from collections import deque

# Add to imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta    # Importance sampling weight (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Normalize priorities to probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Debug information
        state_shapes = [s.shape for s in states]
        if len(set(str(s) for s in state_shapes)) > 1:
            print(f"Found inconsistent shapes: {set(state_shapes)}")
        
        # Ensure all states have the same shape
        first_shape = states[0].shape
        valid_indices = []
        valid_states = []
        valid_actions = []
        valid_rewards = []
        valid_next_states = []
        valid_dones = []
        
        for i, (s, a, r, ns, d) in enumerate(zip(states, actions, rewards, next_states, dones)):
            if s.shape == first_shape and ns.shape == first_shape:
                valid_indices.append(indices[i])
                valid_states.append(s)
                valid_actions.append(a)
                valid_rewards.append(r)
                valid_next_states.append(ns)
                valid_dones.append(d)
        
        # If we have enough valid samples, use them
        if len(valid_states) >= batch_size // 2:  # Accept at least half the batch size
            states_arr = np.array(valid_states)
            next_states_arr = np.array(valid_next_states)
            new_weights = weights[np.array([list(indices).index(idx) for idx in valid_indices])]
            
            return ((states_arr, np.array(valid_actions), np.array(valid_rewards),
                    next_states_arr, np.array(valid_dones)), 
                    valid_indices, new_weights[:len(valid_indices)])
        
        # Otherwise, skip this batch
        print(f"Not enough valid samples, skipping batch ({len(valid_states)}/{batch_size})")
        return None, None, None
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions, dropout_rate=0.2):
        super(DuelingDQN, self).__init__()
        
        # Deeper convolutional network (feature extractor)
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Additional conv blocks
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate the size of the output from conv layers
        conv_out_size = self._get_conv_out(input_shape)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        features = self.features(x).view(x.size()[0], -1)
        
        # Calculate value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DQNAgent:
    def __init__(self, state_shape, n_actions, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        # Using DuelingDQN instead of DQN
        self.policy_net = DuelingDQN(state_shape, n_actions).to(device)
        self.target_net = DuelingDQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        # Use prioritized experience replay
        self.memory = PrioritizedReplayBuffer(capacity=50_000)
        
        self.batch_size = 64  # Increased batch size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Lower min epsilon
        self.epsilon_decay = 0.9995  # Slower decay
        self.target_update = 1000
        self.steps = 0
        self.replay_freq = 4
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size or self.steps % self.replay_freq != 0:
            return
        
        # Get batch with importance sampling weights
        batch, indices, weights = self.memory.sample(self.batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Double DQN: use policy net to select actions, target net to evaluate them
        next_actions = self.policy_net(next_states).max(1)[1].detach()
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # TD errors for updating priorities
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)  # small constant for stability
        
        # Weighted MSE loss for prioritized replay
        loss = (weights * (current_q_values - target_q_values) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()