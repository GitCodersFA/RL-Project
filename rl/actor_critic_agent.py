"""
Advantage Actor-Critic (A2C) Agent

Implements a simple on-policy actor-critic (policy gradient + baseline)
that operates on 512-d feature vectors produced by the frozen feature extractor.

Design choices (simple, easy to integrate):
- Actor produces action logits for 6 discrete actions
- Critic estimates state-value V(s)
- On each episode, collect trajectory, compute discounted returns,
  compute advantages: A = R - V(s)
- Update policy and value networks jointly using combined loss
- Include entropy bonus for exploration
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import sys
sys.path.append('..')
from config import (
    FEATURE_DIM, NUM_ACTIONS, GAMMA,
    RL_LR, DEVICE
)

# Additional hyperparameters with sensible defaults
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5


class ActorNetwork(nn.Module):
    def __init__(self, state_dim=FEATURE_DIM, action_dim=NUM_ACTIONS):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim=FEATURE_DIM):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ActorCriticAgent:
    def __init__(self, device, state_dim=FEATURE_DIM, action_dim=NUM_ACTIONS):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)

        # Single optimizer for both networks
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=RL_LR)

        # Trajectory buffers for on-policy updates
        self.reset_episode_buffer()

    def reset_episode_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []

    def select_action(self, state, training=True):
        """Select an action given state.
        Returns action index. When training, sample from policy; otherwise take argmax.
        Also stores log_prob and value for training.
        """
        state_t = state.to(self.device).float().unsqueeze(0)

        logits = self.actor(state_t)
        dist = Categorical(logits=logits)

        value = self.critic(state_t).squeeze(0)

        if training:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=1)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Store for training
        self.states.append(state_t.squeeze(0).cpu())
        self.actions.append(action.item())
        self.log_probs.append(log_prob.squeeze(0).cpu())
        self.values.append(value.squeeze(0).cpu())
        self.entropies.append(entropy.squeeze(0).cpu())

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def train_episode(self, gamma=GAMMA, entropy_coef=ENTROPY_COEF, value_loss_coef=VALUE_LOSS_COEF):
        """Compute returns and perform a single on-policy update over the collected episode."""
        if len(self.rewards) == 0:
            return None

        # Compute discounted returns
        returns = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Convert lists to tensors
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # Advantage
        advantages = returns - values

        # Losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = -entropies.mean()

        loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.reset_episode_buffer()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropies.mean().item(),
            'total_loss': loss.item()
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


__all__ = ['ActorCriticAgent']
