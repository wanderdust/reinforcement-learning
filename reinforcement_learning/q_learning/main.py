import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm import tqdm


class QFunction:
    def __init__(self, lr: int = 0.1, discount_factor: int = 0.9):
        self.lr = lr
        self.discount_factor = discount_factor
        self.q_values = None

    def init_q_values(self, num_actions, num_states) -> dict:
        q_values = defaultdict(dict)

        for state in range(num_states):
            for action in range(num_actions):
                q_values[state][action] = 0

        self.q_values = q_values

    def update(self, state, action, next_state, reward):
        next_state_max = max(self.q_values[next_state].values())

        update = (
            self.lr * (reward + self.discount_factor * next_state_max)
            - self.q_values[state][action]
        )

        self.q_values[state][action] += update

    def save_q_values(self, name: str = "q_values", out_path: Path = Path("outputs")):
        if not out_path.exists():
            out_path.mkdir(exist_ok=True)

        with open(out_path / f"{name}.json", "w") as f:
            json.dump(self.q_values, f)

    def load_q_values(self, name: str = "q_values", load_path: Path = Path("outputs")):
        with open(load_path / f"{name}.json", "r") as f:
            q_values_str = json.load(f)

        self.q_values = {
            int(state): {int(action): float(value) for action, value in actions.items()}
            for state, actions in q_values_str.items()
        }


class QLearning:
    def __init__(
        self, env, q_function: QFunction, epsilon: int = 0.99, epsilon_decay=1
    ):
        self.env = env
        self.q_function = q_function

        self.q_function.init_q_values(
            self.env.action_space.n, self.env.observation_space.n
        )
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon

    def policy(self, state):
        self.epsilon *= self.epsilon_decay

        if np.random.uniform() <= self.epsilon:
            return int(self.env.action_space.sample())

        q_values = self.q_function.q_values[state]

        # If 2 or more actions have the same q-value, choose at random
        max_value = max(q_values.values())
        max_actions = [
            action for action, value in q_values.items() if value == max_value
        ]
        return int(np.random.choice(max_actions))

    def train(self, num_episodes: int):
        state, info = self.env.reset()

        for i in tqdm(range(num_episodes)):
            # 1. Select action based on q_value + policy
            action = self.policy(state)

            # 2. Take action
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # 3. Update Q-Value
            self.q_function.update(state, action, next_state, reward)

            # 4. Set the next_state as the current state for next iteration
            state = next_state

            if terminated or truncated:
                state, info = self.env.reset()

        self.q_function.save_q_values()
        self.env.close()

    def play(self, max_steps: int = 100):
        self.q_function.load_q_values()

        state, info = self.env.reset()

        for _ in range(max_steps):
            # 1. Select action based on q_value + policy
            action = self.policy(state)

            # 2. Take action
            state, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                return

        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.9)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)

    args = parser.parse_args()

    env_params = {
        "id": "FrozenLake-v1",
        "desc": None,
        "map_name": "8x8",
        "is_slippery": False,
    }

    q_function = QFunction(lr=args.lr, discount_factor=args.discount_factor)

    if args.train:
        q_learning = QLearning(
            gym.make(**env_params),
            q_function=q_function,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
        )
        q_learning.train(args.episodes)

    if args.play:
        q_learning = QLearning(
            gym.make(**env_params, render_mode="human"),
            q_function=q_function,
            epsilon=0,
            epsilon_decay=1,
        )
        q_learning.play()
