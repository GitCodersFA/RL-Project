"""
Configuration file for RL-Driven Garment Defect Detection
Based on the paper methodology
"""
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configuration
DATA_DIR = 'data/DAGM_KaggleUpload'
IMAGE_SIZE = 512  # Original DAGM image size
PATCH_SIZE = 128  # Patch size for agent observation
INPUT_SIZE = 224  # ResNet input size

# Data augmentation
AUGMENTATION = {
    'rotation': 15,
    'horizontal_flip': 0.5,
    'vertical_flip': 0.5,
    'brightness': 0.2,
    'contrast': 0.2
}

# Feature extractor configuration
FEATURE_DIM = 512  # ResNet-34 feature dimension
ATTENTION_KERNEL = 7  # 7x7 convolution for attention

# Supervised training configuration
SUPERVISED_EPOCHS = 20
SUPERVISED_LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5
BATCH_SIZE = 32

# RL configuration
RL_EPISODES = 400
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE_RL = 64
TARGET_UPDATE_FREQ = 10  # Episodes between target network updates
RL_LR = 1e-4

# Actor-Critic specific hyperparameters
ENTROPY_COEF = 0.01  # Entropy bonus coefficient for exploration
VALUE_LOSS_COEF = 0.5  # Coefficient for value loss in combined loss

# Actions
NUM_ACTIONS = 6  # up, down, left, right, no-defect, defect
ACTION_NAMES = ['up', 'down', 'left', 'right', 'no-defect', 'defect']

# Rewards (as per paper)
REWARD_CORRECT = 1.0
REWARD_INCORRECT = -1.0
REWARD_MOVEMENT = -0.01

# Maximum steps per episode
MAX_STEPS = 20
