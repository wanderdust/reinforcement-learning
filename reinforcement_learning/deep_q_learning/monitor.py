#!/usr/bin/env python3
"""Simple monitoring script to track training progress"""
import argparse
import json
import os
import time
from collections import deque


def monitor_training(metrics_file="checkpoints/metrics.jsonl", window=100, refresh=10):
    """Monitor training progress from metrics file"""
    
    print("=" * 80)
    print("DQN Training Monitor")
    print("=" * 80)
    print(f"Metrics file: {metrics_file}")
    print(f"Moving average window: {window} episodes")
    print(f"Refresh interval: {refresh} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_size = 0
    
    try:
        while True:
            if not os.path.exists(metrics_file):
                print(f"Waiting for training to start... ({metrics_file} not found)")
                time.sleep(refresh)
                continue
            
            # Check if file has new data
            current_size = os.path.getsize(metrics_file)
            if current_size == last_size:
                time.sleep(refresh)
                continue
            
            last_size = current_size
            
            # Read all metrics
            metrics = []
            with open(metrics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
            
            if not metrics:
                time.sleep(refresh)
                continue
            
            # Calculate statistics
            recent_metrics = metrics[-window:] if len(metrics) > window else metrics
            
            latest = metrics[-1]
            avg_reward = sum(m['total_reward'] for m in recent_metrics) / len(recent_metrics)
            avg_steps = sum(m['steps'] for m in recent_metrics) / len(recent_metrics)
            max_reward = max(m['total_reward'] for m in recent_metrics)
            min_reward = min(m['total_reward'] for m in recent_metrics)
            
            # Clear screen (works on Unix-like systems)
            os.system('clear' if os.name != 'nt' else 'cls')
            
            # Display stats
            print("=" * 80)
            print("DQN Training Monitor - Live Stats")
            print("=" * 80)
            print(f"\nLatest Episode: {latest['episode']}")
            print(f"Total Steps Trained: {latest['total_steps']:,}")
            print(f"Current Epsilon: {latest['epsilon']:.4f}")
            print(f"\nLast Episode Performance:")
            print(f"  Total Reward: {latest['total_reward']:.2f}")
            print(f"  Steps: {latest['steps']}")
            
            print(f"\nMoving Average (last {len(recent_metrics)} episodes):")
            print(f"  Avg Total Reward: {avg_reward:.2f}")
            print(f"  Avg Steps: {avg_steps:.1f}")
            print(f"  Best Reward: {max_reward:.2f}")
            print(f"  Worst Reward: {min_reward:.2f}")
            
            # Simple progress indicator
            print(f"\nRecent Episodes (last 20):")
            recent_20 = metrics[-20:]
            for m in recent_20:
                bar_length = int(max(0, min(50, m['total_reward'] / 10)))
                bar = '█' * bar_length
                print(f"  Ep {m['episode']:4d}: {bar} {m['total_reward']:.1f}")
            
            print("\n" + "=" * 80)
            print(f"Last updated: {latest['timestamp']}")
            print(f"Refreshing every {refresh} seconds... (Ctrl+C to stop)")
            
            time.sleep(refresh)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")


def main():
    parser = argparse.ArgumentParser(description='Monitor DQN training progress')
    parser.add_argument('--metrics-file', type=str, default='checkpoints/metrics.jsonl',
                        help='Path to metrics file (default: checkpoints/metrics.jsonl)')
    parser.add_argument('--window', type=int, default=100,
                        help='Moving average window size (default: 100)')
    parser.add_argument('--refresh', type=int, default=10,
                        help='Refresh interval in seconds (default: 10)')
    
    args = parser.parse_args()
    
    monitor_training(args.metrics_file, args.window, args.refresh)


if __name__ == "__main__":
    main()
