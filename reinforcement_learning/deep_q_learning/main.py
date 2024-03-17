import gymnasium as gym
from gymnasium.experimental.wrappers import (
    AtariPreprocessingV0,
    FrameStackObservationV0,
)

env = gym.make(
    "ALE/SpaceInvaders-v5",
    render_mode="human",
    frameskip=1,
)

env = AtariPreprocessingV0(env)
env = FrameStackObservationV0(env, 4)

observation, info = env.reset()


if __name__ == "__main__":
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
