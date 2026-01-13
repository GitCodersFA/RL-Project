"""
Experience Replay Buffer for DQN
"""
import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer for storing transitions.
    
    Stores (state, action, reward, next_state, done) tuples.
    """
    def __init__(self, capacity):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a transition.
        
        Args:
            state: Current state (feature vector)
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode terminated flag
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.stack([t[0] for t in batch])
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.stack([t[3] for t in batch])
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
