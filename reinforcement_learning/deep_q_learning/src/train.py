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
        self.env = AtariPreprocessingV0(self.env, frame_skip=4)
        self.env = FrameStackObservationV0(self.env, 4)
        self.env = ClipRewardV0(self.env, -1, 1)

        # Initialising Q-Function(s)
        self.q_function = QFunction(4, self.env.action_space.n, self.device).to(
            self.device
        )
        self.q_function_target = QFunction(4, self.env.action_space.n, self.device).to(
            self.device
        )
        self.optimizer = optim.RMSprop(self.q_function.parameters(), lr=0.01)
        self.criterion = MSELoss()

        # Utils
        self.memory_size = 500
        self.memory = ReplayMemory(self.memory_size)

        # Training params
        self.epsilon = 0.1
        self.discount = 0.9
        self.update_target_q_every = 10000

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

        return torch.argmax(self.q_function(x))

    def train(self):
        obs, info = self.env.reset()

        for step_i in count(0):
            action = self.policy(obs)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

            self.memory.add(obs, action, reward, next_obs, terminated)
            obs = next_obs

            self.writer.add_scalar("Reward", reward, global_step=step_i)
            self.writer.add_scalar("Epsilon", self.epsilon, global_step=step_i)

            if step_i > self.memory_size:
                self.learn(step_i)

            if terminated:
                obs, info = self.env.reset()

    def learn(self, step_i):
        # 4. Sample random batch of transitions
        batch = self.memory.sample(size=64)

        # 5. Set target `y = r + discount * max_a Q(next_state)` or `y=r` if final state
        for obs, action, reward, next_obs, terminated in batch:
            if terminated:
                y = reward
            else:
                y = reward + self.discount * torch.max(
                    self.q_function_target(next_obs)
                ).to(self.device)

            q_s = self.q_function_target(obs).squeeze()[action].to(self.device)

            self.optimizer.zero_grad()
            loss = self.criterion(q_s, y)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar("Loss", loss.item(), global_step=step_i)

            if step_i % self.update_target_q_every == 0:
                self.q_function_target.load_state_dict(self.q_function.state_dict())
