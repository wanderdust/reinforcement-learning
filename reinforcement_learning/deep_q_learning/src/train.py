from itertools import count

import gymnasium as gym
import numpy as np
from gymnasium.experimental.wrappers import (
    AtariPreprocessingV0,
    ClipRewardV0,
    FrameStackObservationV0,
)
from src.q_function import QFunction
from src.replay_memory import ReplayMemory
from torch import argmax, optim
from torch.nn import MSELoss


class DQN:
    def __init__(self):
        super().__init__()

        # Environment
        self.env = gym.make(
            "ALE/SpaceInvaders-v5",
            render_mode="human",
            frameskip=1,
        )
        self.env = AtariPreprocessingV0(self.env, frame_skip=4)
        self.env = FrameStackObservationV0(self.env, 4)
        self.env = ClipRewardV0(self.env, -1, 1)

        # Initialising Q-Function
        self.q_function = QFunction(4, self.env.action_space.n)
        self.q_function_target = QFunction(4, self.env.action_space.n)
        self.optimizer = optim.RMSprop(self.q_function.parameters(), lr=0.01)
        self.criterion = MSELoss()

        # Utils
        self.memory = ReplayMemory()

        # Training params
        self.epsilon = 0.1
        self.discount = 0.9
        self.update_target_q_every = 10000

    def policy(self, x):
        if self.epsilon >= np.random.uniform():
            return self.env.action_space.sample()

        return argmax(self.q_function(x))

    def train(self):
        obs, info = self.env.reset()

        for i in count(0):
            self.training_step(i, obs)

    def training_step(self, i, obs):
        # 1. Take action based on policy and observe r + x_t+1
        # if prob <= 0.1 random action, else use Q-function
        action = self.policy(obs)

        # 2. Take action
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # 3. Store preprocessed state in memory (x, a, r, x_t+1)
        self.memory.add(obs, action, reward, next_obs, terminated)

        # 4. Sample random batch of transitions
        batch = self.memory.sample(size=64)

        # 5. Set target `y = r + discount * max_a Q(next_state)` or `y=r` if final state
        for obs, action, reward, next_obs, terminated in batch:
            if terminated:
                y = reward
            else:
                y = reward + self.discount * np.max(self.q_function_target(next_obs))

            loss = self.criterion(self.q_function_target(obs)[action], y)
            loss.backward()
            self.optimizer.step()

            if i % self.update_target_q_every == 0:
                self.q_function_target.load_state_dict(self.q_function.state_dict())
