# Q-Learning with Gymnasium

This Python project implements a Q-Learning algorithm for reinforcement learning, using the Gymnasium library. It provides functionality for both training and playing environments, specifically designed for the "FrozenLake-v1" game.

## Requirements
- Python 3.6+
- Gymnasium
- Numpy
- tqdm

## Installation
To install the required libraries, run:

```
pip install gymnasium numpy tqdm
```


## Structure
- `QFunction`: Handles the Q-learning logic, including Q-value initialization, updating, saving, and loading.
- `Policy`: Defines the action selection policy based on the current Q-values.
- `Train`: Manages the training process, including episode handling and Q-value updating.
- `Game`: Facilitates playing the game with the learned Q-values.

## Usage
To run the script, use the following command:
```
python main.py [--train] [--play] [--episodes EPISODES] [--lr LEARNING_RATE] [--discount-factor DISCOUNT_FACTOR] [--max-steps MAX_STEPS]
```

- `--train`: Triggers the training process.
- `--play`: Plays the game using the learned Q-values.
- `--episodes`: Sets the number of episodes for training (default: 500).
- `--lr`: Sets the learning rate (default: 0.1).
- `--discount-factor`: Sets the discount factor for future rewards (default: 0.9).
- `--max-steps`: Sets the maximum steps per episode (default: 30).

## Output
- Saves Q-values to `outputs/q_values.json` after training.
- During play, the agent uses the learned Q-values to navigate the environment.
