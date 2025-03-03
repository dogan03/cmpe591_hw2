import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from homework2 import Hw2Env

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

 
BATCH_SIZE = 128             
LEARNING_RATE = 3e-4         
GAMMA = 0.99                 
TAU = 0.005                  
EPSILON_START = 1.0
EPSILON_END = 0.05           
EPSILON_DECAY = 0.999        
MEMORY_SIZE = 100_000          
HIDDEN_DIM = 256             
UPDATE_EVERY = 4             
PRIORITIZED_REPLAY = True    
ALPHA = 0.6                  
BETA_START = 0.4             
BETA_FRAMES = 100000         

 
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'priority'))

 
class ReplayMemory(object):
    def __init__(self, capacity, use_priorities=False):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.use_priorities = use_priorities
        self.beta = BETA_START
        self.beta_increment = (1.0 - BETA_START) / BETA_FRAMES
        self.max_priority = 1.0

    def push(self, batch):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
         
        if self.use_priorities:
            priority = self.max_priority
            idx = self.position
            self.priorities[idx] = priority
        
        self.memory[self.position] = batch
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.use_priorities:
             
            self.beta = min(1.0, self.beta + self.beta_increment)
            
             
            priorities = self.priorities[:len(self.memory)]
            probabilities = priorities ** ALPHA
            probabilities /= probabilities.sum()
            
             
            indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
            samples = [self.memory[idx] for idx in indices]
            
             
            weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()   
            
            return samples, indices, torch.FloatTensor(weights).to(device)
        else:
             
            return random.sample(self.memory, batch_size), None, None

    def update_priorities(self, indices, priorities):
        if self.use_priorities and indices is not None:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority.item()
                self.max_priority = max(self.max_priority, priority.item())

    def __len__(self):
        return len(self.memory)

 
class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DuelingQNetwork, self).__init__()
        
         
        print(f"Creating DuelingQNetwork with input_dim={input_dim}, output_dim={output_dim}")
        
         
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
         
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
         
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
         
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
         
        if x.shape[1] != self.feature_layer[0].in_features:
            raise ValueError(f"Input shape {x.shape} doesn't match expected input dim {self.feature_layer[0].in_features}")
            
        features = self.feature_layer(x)
        
         
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
         
         
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

 
class Agent(object):
    def __init__(self, n_states, n_actions, hidden_dim):
         
        self.q_local = DuelingQNetwork(n_states, n_actions, hidden_dim).to(device)
        self.q_target = DuelingQNetwork(n_states, n_actions, hidden_dim).to(device)
        
         
        self.loss_fn = nn.SmoothL1Loss(reduction='none')   
        self.optim = optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE)
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.replay_memory = ReplayMemory(MEMORY_SIZE, use_priorities=PRIORITIZED_REPLAY)
        self.t_step = 0
        
         
        self.hard_update(self.q_local, self.q_target)
        
    def act(self, state):
        """Select action in eval mode (no exploration)"""
        self.q_local.eval()
        with torch.no_grad():
            action_values = self.q_local(state)
        self.q_local.train()
        return action_values.argmax(dim=1).item()

    def get_action(self, state, eps, check_eps=True):
        """Select action (with optional exploration)"""
        self.q_local.eval()
        
        if check_eps and random.random() < eps:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device)
        
        with torch.no_grad():
            action_values = self.q_local(Variable(state).type(FloatTensor))
        
        self.q_local.train()
        return action_values.max(1)[1].view(1, 1)

    def step(self, state, action, reward, next_state, done):
         
        priority = torch.tensor([self.replay_memory.max_priority], device=device)
        
         
        done_tensor = torch.tensor([float(done)], device=device)
        self.replay_memory.push(Transition(state, action, reward, next_state, done_tensor, priority))
        
         
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
             
            if len(self.replay_memory) > BATCH_SIZE:
                self.learn(GAMMA)

    def learn(self, gamma):
        if len(self.replay_memory) < BATCH_SIZE:
            return
        
         
        experiences, indices, weights = self.replay_memory.sample(BATCH_SIZE)
        transitions = Transition(*zip(*experiences))
        
        states = torch.cat(transitions.state)
        actions = torch.cat(transitions.action)
        rewards = torch.cat(transitions.reward)
        next_states = torch.cat(transitions.next_state)
        dones = torch.cat(transitions.done).float()
        
         
         
        with torch.no_grad():
             
            next_actions = self.q_local(next_states).max(1)[1].unsqueeze(1)
             
            Q_targets_next = self.q_target(next_states).gather(1, next_actions).squeeze(1)
             
            Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        
         
        self.q_local.train()
        Q_expected = self.q_local(states).gather(1, actions).squeeze(1)
        
         
        td_errors = Q_targets - Q_expected
        if weights is not None:
            loss = (self.loss_fn(Q_expected, Q_targets) * weights).mean()
            
             
            priorities = td_errors.abs() + 1e-5   
            self.replay_memory.update_priorities(indices, priorities)
        else:
            loss = self.loss_fn(Q_expected, Q_targets).mean()
        
         
        self.optim.zero_grad()
        loss.backward()
         
        torch.nn.utils.clip_grad_norm_(self.q_local.parameters(), 1.0)
        self.optim.step()
        
         
        self.soft_update(self.q_local, self.q_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def hard_update(self, local, target):
        target.load_state_dict(local.state_dict())