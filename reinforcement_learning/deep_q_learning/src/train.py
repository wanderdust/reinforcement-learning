from collections import deque
from itertools import count
import os
import json
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from gymnasium.experimental.wrappers import (
    AtariPreprocessingV0,
    ClipRewardV0,
    FrameStackObservationV0,
)
from src.q_function import QFunction
from src.replay_memory import ReplayMemory
from torch import optim
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DQN:
    def __init__(self, checkpoint_dir="checkpoints", log_dir="tensorboard_logs"):
        super().__init__()

        # Environment
        self.env = gym.make(
            "ALE/SpaceInvaders-v5",
            frameskip=1,
            # render_mode="human",
        )
        self.env = AtariPreprocessingV0(self.env, frame_skip=3)
        self.env = FrameStackObservationV0(self.env, 4)
        self.env = ClipRewardV0(self.env, -1, 1)

        # Initialising Q-Function(s)
        self.q_function = QFunction(4, self.env.action_space.n).to(self.device)
        self.q_function_target = QFunction(4, self.env.action_space.n).to(self.device)
        self.q_function_target.load_state_dict(self.q_function.state_dict())
        self.optimizer = optim.RMSprop(self.q_function.parameters(), lr=0.00025)
        self.criterion = MSELoss()

        # Utils
        self.memory_size = 10_000
        self.memory = ReplayMemory(self.memory_size)

        # Training params
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.discount = 0.9
        self.update_target_q_every = 10_000
        self.batch_size = 32
        self.min_replay_memory_size = 500

        # Tensorboard with timestamped runs
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir_with_run = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(log_dir=log_dir_with_run)
        
        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Metrics tracking
        self.episode_count = 0
        self.total_steps = 0
        self.metrics_file = os.path.join(checkpoint_dir, "metrics.jsonl")

    @property
    def device(self):
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def policy(self, x):
        if self.epsilon >= np.random.uniform():
            return self.env.action_space.sample()

        return torch.argmax(self.q_function(x)).item()

    def update_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay

    def train(self, max_episodes=None, checkpoint_every=100):
        for i in count(0):
            if max_episodes and i >= max_episodes:
                break
                
            rewards, actions, actual_score = self.episode()
            self.update_epsilon()
            self.episode_count = i

            total_reward = sum(rewards)
            avg_reward_per_step = total_reward / len(rewards)
            
            # Tensorboard logging
            self.writer.add_scalar("Training/Clipped_Total_Reward", total_reward, global_step=i)
            self.writer.add_scalar("Training/Avg_Reward_Per_Step", avg_reward_per_step, global_step=i)
            self.writer.add_scalar("Performance/Actual_Game_Score", actual_score, global_step=i)
            self.writer.add_scalar("Episode_Length", len(rewards), global_step=i)
            self.writer.add_scalar("Epsilon", self.epsilon, global_step=i)
            self.writer.add_histogram("actions", torch.tensor(actions), global_step=i)
            
            # Simple console logging
            action_meanings = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
            action_dist = ', '.join([f"{action_meanings[a]}:{actions.count(a)}" for a in range(6)])
            print(f"Episode {i}: Score={actual_score:.0f}, Clipped Reward={total_reward:.1f}, Steps={len(rewards)}, Epsilon={self.epsilon:.4f}")
            print(f"  Actions: {action_dist}")
            
            # Save metrics to file
            self._save_metrics(i, total_reward, avg_reward_per_step, len(rewards), actual_score)
            
            # Checkpoint saving
            if (i + 1) % checkpoint_every == 0:
                self.save_checkpoint(f"checkpoint_episode_{i+1}.pt")
                print(f"✓ Checkpoint saved at episode {i+1}")

    def episode(self):
        rewards = []
        actions = []

        obs, info = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        episode_raw_reward = 0.0

        for step_i in count(0):
            action = self.policy(obs)
            actions.append(action)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Track actual game reward before clipping
            if 'episode' in info:
                episode_raw_reward = info['episode']['r']
            
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

            self.memory.add(obs, action, reward, next_obs, terminated)
            obs = next_obs

            if self.memory.size() > self.min_replay_memory_size:
                self.learn(self.total_steps)

            rewards.append(reward.item())
            self.total_steps += 1

            if terminated:
                # If episode info not available, estimate from lives or use clipped sum
                if episode_raw_reward == 0.0:
                    episode_raw_reward = sum(rewards)
                return rewards, actions, episode_raw_reward

    def learn(self, step_i):
        observations, actions, rewards, next_observations, dones = self.memory.sample(
            size=self.batch_size, device=self.device
        )

        with torch.no_grad():
            max_values = torch.max(self.q_function_target(next_observations), 1)[0]
            target = rewards + self.discount * max_values * (1 - dones)

        predicted = self.q_function(observations).gather(1, actions.unsqueeze(dim=1)).squeeze()

        self.optimizer.zero_grad()
        loss = self.criterion(predicted, target)
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar("Loss", loss.item(), global_step=step_i)

        if step_i % self.update_target_q_every == 0:
            self.q_function_target.load_state_dict(self.q_function.state_dict())

        return loss
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'q_function_state_dict': self.q_function.state_dict(),
            'q_function_target_state_dict': self.q_function_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, checkpoint_path)
        
        # Also save as "latest" for easy loading
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save({
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'q_function_state_dict': self.q_function.state_dict(),
            'q_function_target_state_dict': self.q_function_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, latest_path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.q_function.load_state_dict(checkpoint['q_function_state_dict'])
        self.q_function_target.load_state_dict(checkpoint['q_function_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode']
        self.total_steps = checkpoint['total_steps']
        
        print(f"Loaded checkpoint from episode {self.episode_count}")
    
    def _save_metrics(self, episode, total_reward, avg_reward_per_step, steps, actual_score):
        """Save metrics to JSONL file for easy monitoring"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'clipped_total_reward': total_reward,
            'avg_reward_per_step': avg_reward_per_step,
            'actual_game_score': actual_score,
            'steps': steps,
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
