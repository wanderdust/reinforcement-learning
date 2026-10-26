import random
from collections import deque, namedtuple

import numpy as np
import torch


class ReplayMemory:
    def __init__(self, maxlen: int = 10_000):
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
        self.memory.append(
            self.experience(obs.to("cpu", torch.uint8), action, reward, next_obs.to("cpu", torch.uint8), terminated)
        )

    def sample(self, size: int, device: str):
        sample = random.sample(self.memory, size)

        observations = torch.zeros((size, 4, 84, 84), dtype=torch.uint8)
        actions = torch.zeros(size, dtype=torch.int64).to(device)
        rewards = torch.zeros(size).to(device)
        next_observations = torch.zeros((size, 4, 84, 84), dtype=torch.uint8)
        dones = torch.zeros(size).to(device)

        for i, (observation, action, reward, next_observation, done) in enumerate(sample):
            observations[i] = observation
            actions[i] = action
            rewards[i] = reward
            next_observations[i] = next_observation
            dones[i] = done

        return (
            observations.to(device, torch.float32),
            actions,
            rewards,
            next_observations.to(device, torch.float32),
            dones,
        )

    def size(self):
        return len(self.memory)
