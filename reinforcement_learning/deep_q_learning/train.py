#!/usr/bin/env python3
"""Training script for DQN agent"""
import argparse
from datetime import datetime
from src.train import DQN


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on Space Invaders')
    parser.add_argument('--max-episodes', type=int, default=None, 
                        help='Maximum number of episodes to train (default: infinite)')
    parser.add_argument('--checkpoint-every', type=int, default=100,
                        help='Save checkpoint every N episodes (default: 100)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--log-dir', type=str, default='tensorboard_logs',
                        help='Directory for tensorboard logs (default: tensorboard_logs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file (e.g., latest.pt)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DQN Training - Space Invaders")
    print("=" * 60)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Log directory: {args.log_dir}")
    print(f"Checkpoint interval: every {args.checkpoint_every} episodes")
    if args.max_episodes:
        print(f"Max episodes: {args.max_episodes}")
    else:
        print("Max episodes: unlimited (Ctrl+C to stop)")
    print("=" * 60)
    
    dqn = DQN(checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir)
    
    # Print the run name for TensorBoard
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"TensorBoard run: {run_name}")
    print(f"View with: make tensorboard")
    print("=" * 60)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        dqn.load_checkpoint(args.resume)
    
    try:
        dqn.train(max_episodes=args.max_episodes, checkpoint_every=args.checkpoint_every)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving final checkpoint...")
        dqn.save_checkpoint("interrupted.pt")
        print("✓ Checkpoint saved as 'interrupted.pt'")


if __name__ == "__main__":
    main()
