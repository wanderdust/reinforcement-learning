#!/usr/bin/env python3
"""Play script to watch the trained agent play Space Invaders"""
import argparse
import time
import gymnasium as gym
import torch
from gymnasium.experimental.wrappers import (
    AtariPreprocessingV0,
    FrameStackObservationV0,
)
from src.q_function import QFunction


def play_game(checkpoint_path, num_episodes=5, render_delay=0.03):
    """Load a trained model and watch it play"""
    
    # Setup environment with rendering
    env = gym.make("ALE/SpaceInvaders-v5", frameskip=1, render_mode="human")
    env = AtariPreprocessingV0(env, frame_skip=3)
    env = FrameStackObservationV0(env, 4)
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load model
    q_function = QFunction(4, env.action_space.n).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    q_function.load_state_dict(checkpoint['q_function_state_dict'])
    q_function.eval()
    
    print(f"Loaded model from episode {checkpoint['episode']}")
    print(f"Model epsilon: {checkpoint['epsilon']:.4f}")
    print(f"\nStarting playback of {num_episodes} episodes...")
    print("Close the game window or press Ctrl+C to stop\n")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
        
        total_reward = 0
        steps = 0
        
        while True:
            # Select action (greedy - no exploration)
            with torch.no_grad():
                action = torch.argmax(q_function(obs)).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
            
            total_reward += reward
            steps += 1
            
            time.sleep(render_delay)
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Reward = {total_reward:.0f}, Steps = {steps}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print(f"Average reward over {num_episodes} episodes: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Best episode: {max(episode_rewards):.0f}")
    print(f"Worst episode: {min(episode_rewards):.0f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Watch trained DQN agent play Space Invaders')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/latest.pt',
                        help='Path to checkpoint file (default: checkpoints/latest.pt)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play (default: 5)')
    parser.add_argument('--delay', type=float, default=0.03,
                        help='Delay between frames in seconds (default: 0.03)')
    
    args = parser.parse_args()
    
    try:
        play_game(args.checkpoint, args.episodes, args.delay)
    except KeyboardInterrupt:
        print("\n\nPlayback stopped by user")
    except FileNotFoundError:
        print(f"\nError: Checkpoint file not found: {args.checkpoint}")
        print("Train a model first or specify a valid checkpoint path")


if __name__ == "__main__":
    main()
