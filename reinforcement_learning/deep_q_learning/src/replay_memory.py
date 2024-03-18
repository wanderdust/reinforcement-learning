import random
from collections import deque, namedtuple

import numpy as np


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

    def sample(self, size: int):
        size = size if len(self.memory) > size else len(self.memory)
        return random.sample(self.memory, size)
