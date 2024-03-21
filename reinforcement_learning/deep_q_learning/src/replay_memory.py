import random
from collections import deque, namedtuple

import numpy as np
import torch


class ReplayMemory:
    def __init__(self, maxlen: int = 500):
        self.memory = deque(maxlen=maxlen)
        self.experience = namedtuple(
            "Experience", ["obs", "action", "reward", "next_obs", "terminated"]
        )

    def add(
        self,
        obs: np.array,
        action: int,
        reward: float,
        next_obs: np.array,
        terminated: bool,
    ):
        self.memory.append(self.experience(obs, action, reward, next_obs, terminated))

    def sample(self, size: int, device: str):
        sample = random.sample(self.memory, size)

        observations = torch.zeros((size, 4, 84, 84), dtype=torch.float32).to(device)
        actions = torch.zeros(size, dtype=torch.int64).to(device)
        rewards = torch.zeros(size).to(device)
        next_observations = torch.zeros((size, 4, 84, 84), dtype=torch.float32).to(device)
        dones = torch.zeros(size).to(device)

        for i, (observation, action, reward, next_observation, done) in enumerate(sample):
            observations[i] = observation
            actions[i] = action
            rewards[i] = reward
            next_observations[i] = next_observation
            dones[i] = done

        return observations, actions, rewards, next_observations, dones
