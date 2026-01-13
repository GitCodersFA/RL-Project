"""
Defect Detection Environment for RL
Formalized as Markov Decision Process as described in the paper.
"""
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append('..')
from config import (
    PATCH_SIZE, IMAGE_SIZE, INPUT_SIZE,
    REWARD_CORRECT, REWARD_INCORRECT, REWARD_MOVEMENT,
    MAX_STEPS, NUM_ACTIONS, ACTION_NAMES
)
from dataset import PatchExtractor


class DefectDetectionEnv:
    """
    MDP Environment for Defect Detection.
    
    From paper:
    - State s_t = f_t (512-d feature vector from current patch)
    - Actions: {0: ↑, 1: ↓, 2: ←, 3: →, 4: no-defect, 5: defect}
    - Transition: Movement actions shift by half-patch, classification terminates
    - Reward: +1 correct, -1 incorrect, -0.01 movement cost
    """
    def __init__(self, feature_extractor, device):
        """
        Args:
            feature_extractor: Pre-trained feature extractor model
            device: torch device
        """
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        self.device = device
        
        self.patch_extractor = PatchExtractor(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE
        )
        
        # Movement step size (half-patch as per paper)
        self.step_size = PATCH_SIZE // 2
        
        # Current state
        self.current_image = None
        self.current_label = None
        self.current_u = 0  # Patch top-left x
        self.current_v = 0  # Patch top-left y
        self.steps = 0
        self.done = False
        
    def reset(self, image, label):
        """
        Reset environment with new image.
        
        Args:
            image: PIL Image (original 512x512)
            label: Ground truth (0: no-defect, 1: defect)
            
        Returns:
            Initial state (feature vector)
        """
        self.current_image = image
        self.current_label = label
        
        # Start at center patch
        self.current_u, self.current_v = self.patch_extractor.get_center_position()
        self.steps = 0
        self.done = False
        
        # Get initial state (feature of center patch)
        state = self._get_state()
        
        return state
    
    def _get_state(self):
        """Extract feature vector for current patch"""
        patch = self.patch_extractor.extract_patch(
            self.current_image,
            self.current_u,
            self.current_v
        )
        
        # Add batch dimension and move to device
        patch = patch.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features, _ = self.feature_extractor(patch)
        
        return features.squeeze(0).cpu()
    
    def step(self, action):
        """
        Take action in environment.
        
        Args:
            action: Action index (0-5)
            
        Returns:
            next_state: Next state feature vector
            reward: Reward received
            done: Episode terminated flag
            info: Additional information
        """
        self.steps += 1
        info = {'action_name': ACTION_NAMES[action]}
        
        if action < 4:  # Movement action
            # Update position based on action
            if action == 0:  # Up
                self.current_v = max(0, self.current_v - self.step_size)
            elif action == 1:  # Down
                self.current_v = min(IMAGE_SIZE - PATCH_SIZE, self.current_v + self.step_size)
            elif action == 2:  # Left
                self.current_u = max(0, self.current_u - self.step_size)
            elif action == 3:  # Right
                self.current_u = min(IMAGE_SIZE - PATCH_SIZE, self.current_u + self.step_size)
            
            reward = REWARD_MOVEMENT
            self.done = self.steps >= MAX_STEPS
            
            if self.done:
                # Force classification at max steps
                info['timeout'] = True
                
        else:  # Classification action (4: no-defect, 5: defect)
            predicted = action - 4  # 0 or 1
            
            if predicted == self.current_label:
                reward = REWARD_CORRECT
                info['correct'] = True
            else:
                reward = REWARD_INCORRECT
                info['correct'] = False
            
            info['predicted'] = predicted
            info['ground_truth'] = self.current_label
            self.done = True
        
        # Get next state
        next_state = self._get_state() if not self.done else torch.zeros(512)
        
        return next_state, reward, self.done, info
    
    def get_position(self):
        """Get current patch position"""
        return self.current_u, self.current_v
