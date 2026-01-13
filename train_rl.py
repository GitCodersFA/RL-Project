"""
Reinforcement Learning Training Script
Train the DQN agent for defect detection.

Usage:
    python train_rl.py --episodes 400
    python train_rl.py --episodes 100 --target_update 5
"""
import os
import argparse
import torch
import random
from PIL import Image
from tqdm import tqdm
import numpy as np

from config import (
    DEVICE, DATA_DIR, RL_EPISODES, TARGET_UPDATE_FREQ,
    FEATURE_DIM, MAX_STEPS
)
from models.feature_extractor import DefectClassifier
from rl.environment import DefectDetectionEnv
from rl.actor_critic_agent import ActorCriticAgent
from utils import plot_rl_rewards


def parse_args():
    parser = argparse.ArgumentParser(description='RL Training for Defect Detection')
    parser.add_argument('--episodes', type=int, default=RL_EPISODES,
                        help=f'Number of training episodes (default: {RL_EPISODES})')
    parser.add_argument('--target_update', type=int, default=TARGET_UPDATE_FREQ,
                        help=f'Target network update frequency (default: {TARGET_UPDATE_FREQ})')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting (useful for headless servers)')
    return parser.parse_args()


def load_images(data_dir):
    """Load all images and labels for RL training"""
    images = []
    labels = []
    
    class_dir = os.path.join(data_dir, 'Class10', 'Train')
    labels_file = os.path.join(class_dir, 'labels.txt')
    
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_path = os.path.join(class_dir, parts[0])
                    label = int(parts[1])
                    
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert('RGB')
                        images.append(img)
                        labels.append(label)
    else:
        # Auto-detect from directory structure
        label_dir = os.path.join(class_dir, 'Label')
        
        for img_file in sorted(os.listdir(class_dir)):
            if img_file.endswith(('.png', '.PNG', '.jpg', '.bmp')):
                img_path = os.path.join(class_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                
                label = 0
                for ext in ['.png', '.PNG', '.bmp']:
                    if os.path.exists(os.path.join(label_dir, f'{base_name}_label{ext}')):
                        label = 1
                        break
                
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                labels.append(label)
    
    print(f"Loaded {len(images)} images")
    print(f"Defects: {sum(labels)}, Non-defects: {len(labels) - sum(labels)}")
    
    return images, labels


def main():
    args = parse_args()
    
    episodes = args.episodes
    target_update_freq = args.target_update
    
    print("=" * 60)
    print("Reinforcement Learning Training Stage")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Episodes: {episodes}")
    print(f"Target Update Frequency: {target_update_freq}")
    print("=" * 60)
    
    # Check if pre-trained classifier exists
    classifier_path = 'checkpoints/best_classifier.pth'
    if not os.path.exists(classifier_path):
        print("\nPre-trained classifier not found!")
        print("Please run train_supervised.py first.")
        return
    
    # Load pre-trained feature extractor
    print("\nLoading pre-trained feature extractor...")
    # Use local weights (pretrained flag disabled to avoid network downloads during tests)
    model = DefectClassifier(pretrained=False).to(DEVICE)
    checkpoint = torch.load(classifier_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze feature extractor (as per paper)
    model.freeze_feature_extractor()
    feature_extractor = model.feature_extractor
    feature_extractor.eval()
    print("Feature extractor loaded and frozen")
    
    # Load images
    print("\nLoading images...")
    if not os.path.exists(os.path.join(DATA_DIR, 'Class10')):
        print("Dataset not found! Running download script...")
        from download_dataset import download_dagm_dataset
        download_dagm_dataset()
    
    images, labels = load_images(DATA_DIR)
    
    if len(images) == 0:
        print("No images loaded! Please check the dataset.")
        return
    
    # Create environment and agent
    print("\nInitializing RL components...")
    env = DefectDetectionEnv(feature_extractor, DEVICE)
    agent = ActorCriticAgent(DEVICE)
    
    # Training loop
    print("\nStarting RL training...")
    episode_rewards = []
    episode_lengths = []
    correct_classifications = 0
    total_classifications = 0
    
    pbar = tqdm(range(1, episodes + 1), desc='RL Training')
    # Variables to track last update losses
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for episode in pbar:
        # Sample random image
        idx = random.randint(0, len(images) - 1)
        image = images[idx]
        label = labels[idx]
        
        # Reset environment
        state = env.reset(image, label)
        
        # Reset per-episode buffers for on-policy agent
        agent.reset_episode_buffer()

        episode_reward = 0
        step = 0
        
        while True:
            # Select action (actor-critic samples from policy)
            action = agent.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Store reward for on-policy update
            agent.store_reward(reward)
            
            episode_reward += reward
            step += 1
            state = next_state
            
            if done:
                if 'correct' in info:
                    total_classifications += 1
                    if info['correct']:
                        correct_classifications += 1
                break
        
        # After episode completes, update policy (on-policy A2C-style)
        loss_dict = agent.train_episode()
        if loss_dict is not None:
            last_policy_loss = loss_dict['policy_loss']
            last_value_loss = loss_dict['value_loss']
            last_entropy = loss_dict['entropy']

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Update progress bar
        avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        accuracy = 100 * correct_classifications / total_classifications if total_classifications > 0 else 0
        pbar.set_postfix({
            'avg_reward': f'{avg_reward:.2f}',
            'pol_loss': f'{last_policy_loss:.3f}',
            'val_loss': f'{last_value_loss:.3f}',
            'ent': f'{last_entropy:.3f}',
            'acc': f'{accuracy:.1f}%'
        })
    
    # Final statistics
    print("\n" + "=" * 60)
    print("RL Training Complete!")
    print("=" * 60)
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average Episode Reward: {avg_reward:.4f}")
    print(f"Classification Accuracy: {100*correct_classifications/total_classifications:.2f}%")
    
    # Save agent
    os.makedirs('checkpoints', exist_ok=True)
    agent.save('checkpoints/actor_critic_agent.pth')
    print(f"Agent saved to: checkpoints/actor_critic_agent.pth")
    
    # Plot rewards
    if not args.no_plot:
        try:
            plot_rl_rewards(episode_rewards, 'checkpoints/rl_rewards.png')
        except Exception as e:
            print(f"Could not generate plot: {e}")
    
    print("\n" + "=" * 60)
    print("RL training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
