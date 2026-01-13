"""
Deep Q-Network (DQN) Agent
As described in the paper for defect detection.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from .replay_buffer import ReplayBuffer
import sys
sys.path.append('..')
from config import (
    FEATURE_DIM, NUM_ACTIONS, GAMMA, 
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    REPLAY_BUFFER_SIZE, BATCH_SIZE_RL, RL_LR
)


class QNetwork(nn.Module):
    """
    Q-Network for approximating action-value function.
    
    Q(s, a; θ) where s is the 512-d feature vector
    """
    def __init__(self, state_dim=FEATURE_DIM, action_dim=NUM_ACTIONS):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        """
        Args:
            state: State tensor of shape (B, state_dim)
            
        Returns:
            Q-values for all actions, shape (B, action_dim)
        """
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network Agent for defect detection.
    
    From paper:
    - ε-greedy exploration
    - Experience replay
    - Target network with periodic sync
    - TD loss minimization
    """
    def __init__(self, device, state_dim=FEATURE_DIM, action_dim=NUM_ACTIONS):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=RL_LR)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # Exploration parameters
        self.epsilon = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        
        # Training parameters
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE_RL
        
    def select_action(self, state, training=True):
        """
        Select action using ε-greedy policy.
        
        Args:
            state: State tensor
            training: If True, use exploration
            
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step.
        
        Returns:
            Loss value or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values (using target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # TD loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Sync target network weights with main network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        """Save agent state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
