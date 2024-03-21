from collections import deque
from itertools import count

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
    def __init__(self):
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
        self.optimizer = optim.RMSprop(self.q_function.parameters(), lr=0.01)
        self.criterion = MSELoss()

        # Utils
        self.memory_size = 500
        self.memory = ReplayMemory(self.memory_size)

        # Training params
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.discount = 0.9
        self.update_target_q_every = 10_000
        self.batch_size = 32
        self.min_replay_memory_size = 500

        # Tensorboard
        self.writer = SummaryWriter(log_dir="./tensorboard_logs")

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

    def train(self):
        for i in count(0):
            rewards, actions = self.episode()
            self.update_epsilon()

            avg_reward = sum(rewards) / len(rewards)
            self.writer.add_scalar("Avg Episode Reward", avg_reward, global_step=i)
            self.writer.add_scalar("Epsilon", self.epsilon, global_step=i)
            self.writer.add_histogram("actions", torch.tensor(actions), global_step=i)

    def episode(self):
        rewards = []
        actions = []

        obs, info = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)

        for step_i in count(0):
            action = self.policy(obs)
            actions.append(action)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

            self.memory.add(obs, action, reward, next_obs, terminated)
            obs = next_obs

            if self.memory.size() > self.min_replay_memory_size:
                self.learn(step_i)

            rewards.append(reward.item())

            if terminated:
                return rewards, actions

    def learn(self, step_i):
        observations, actions, rewards, next_observations, dones = self.memory.sample(
            size=self.batch_size, device=self.device
        )

        max_values = torch.max(self.q_function_target(next_observations), 1)[0]
        target = rewards + self.discount * max_values * (1 - dones)

        predicted = self.q_function(observations).gather(1, actions.unsqueeze(dim=1))

        self.optimizer.zero_grad()
        loss = self.criterion(predicted, target)
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar("Loss", loss.item(), global_step=step_i)

        if step_i % self.update_target_q_every == 0:
            self.q_function_target.load_state_dict(self.q_function.state_dict())
