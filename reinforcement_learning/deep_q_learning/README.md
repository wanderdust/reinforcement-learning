# Deep Q-Learning (DQN) - Space Invaders

Train a DQN agent to play Atari Space Invaders.

## Quick Start

### Installation
```bash
make install
```

### Training
```bash
# Start training (runs indefinitely, Ctrl+C to stop)
make train

# Train for a specific number of episodes
python train.py --max-episodes 1000

# Resume from checkpoint
make resume
```

### Monitoring Training

**Option 1: Simple Console Monitor (Recommended for SSH/Remote)**
```bash
make monitor
```
This displays live training stats updated every 10 seconds including:
- Current episode and total steps
- Epsilon value
- Recent rewards and moving averages
- Visual progress bar

**Option 2: TensorBoard (Detailed Metrics)**
```bash
make tensorboard
# Open http://localhost:6006 in your browser
```

Each training run creates a timestamped subdirectory (e.g., `20251106_123045`) so you can compare different experiments.

**Option 3: Check Metrics File**
```bash
tail -f checkpoints/metrics.jsonl
```

### Watch Trained Agent Play
```bash
# Play 5 episodes with the trained model
make play

# Play 10 episodes
make play-10

# Use specific checkpoint
python play.py --checkpoint checkpoints/checkpoint_episode_500.pt --episodes 3
```

## Training Configuration

The training saves:
- **Checkpoints**: `checkpoints/` directory
  - `latest.pt` - most recent checkpoint
  - `checkpoint_episode_N.pt` - saved every 100 episodes
  - `interrupted.pt` - saved when you Ctrl+C
- **Metrics**: `checkpoints/metrics.jsonl` - simple line-by-line JSON for easy monitoring
- **TensorBoard logs**: `tensorboard_logs/` - detailed training metrics
  - Each run in timestamped subdirectory: `tensorboard_logs/YYYYMMDD_HHMMSS/`

## Makefile Commands

```bash
make help          # Show all commands
make train         # Start training
make train-short   # Quick test (1000 episodes)
make resume        # Resume from latest checkpoint
make play          # Watch agent play
make monitor       # Live training monitor
make tensorboard   # Launch TensorBoard
make clean         # Remove all checkpoints and logs
make clean-logs    # Remove only logs (keep checkpoints)
```

## Monitoring Progress

### Is My Agent Learning? Quick Checklist

**🔍 Key Metrics to Watch:**

1. **Actual Game Score** ⭐ (Most Important) - The real game points
   - Random: ~100-300 points per episode
   - Learning: Gradually increases to 500+
   - Well-trained: 1000-2000+
   - This is the unclipped score you'd see playing the game

2. **Clipped Total Reward** - Used for training (rewards clipped to -1/0/+1)
   - Random: ~5-15 per episode
   - Learning: Gradually increases to 20-50+
   - Note: This is NOT the game score, just the sum of clipped rewards

3. **Epsilon** - Should decay from 1.0 → 0.1
   - Episode 0: 1.0 (pure exploration)
   - Episode 1000: ~0.95
   - Episode 5000: ~0.78
   - Episode 10000: ~0.60

4. **Episode Length** - May increase as agent survives longer
   - Random: ~200-500 steps
   - Learning: May increase to 1000+ steps

5. **Loss** (in TensorBoard) - Should stabilize after initial volatility
   - Early: Very noisy, jumping around
   - Learning: Gradually stabilizes
   - Good sign: Decreasing trend overall

6. **Actions** (console output) - Distribution should shift
   - Random: All actions roughly equal (~16% each)
   - Learning: More FIRE, RIGHTFIRE, LEFTFIRE (useful actions)
   - Bad sign: Stuck on NOOP or only one action

**⚠️ Warning Signs:**
- **Actual Game Score** stays flat after 5000+ episodes → Check hyperparameters
- Loss exploding (going to infinity) → Learning rate too high
- Agent only uses 1-2 actions → May be stuck in local optimum

**📝 Note on Rewards:**
- DQN uses reward clipping (all rewards → -1, 0, or +1) for stable training
- "Clipped Total Reward" is what the agent optimizes, but not the actual score
- Watch "Actual Game Score" to see real performance!

### Random vs Learning
- **Random agent**: ~200-300 reward on average, epsilon = 1.0
- **Learning agent**: Epsilon decreases over time, rewards should increase
- Look for: Increasing average rewards and decreasing epsilon

### What to Watch
1. **Total Reward**: Should trend upward over time
2. **Epsilon**: Should decay from 1.0 to 0.1
3. **Episode Length**: May increase as agent learns to survive longer

## Running on a Server

### Start Training in Background
```bash
# Using nohup
nohup make train > training.log 2>&1 &

# Or using screen
screen -S dqn_training
make train
# Press Ctrl+A then D to detach
```

### Monitor from Another Terminal
```bash
# Live monitor
make monitor

# Or tail the metrics
tail -f checkpoints/metrics.jsonl

# Check training log
tail -f training.log
```

### For TensorBoard on Remote Server
```bash
# On server
tensorboard --logdir=tensorboard_logs --host=0.0.0.0 --port=6006

# On local machine (SSH tunnel)
ssh -L 6006:localhost:6006 user@server
# Then open http://localhost:6006
```

## File Structure
```
.
├── train.py              # Training script
├── play.py               # Play/test trained agent
├── monitor.py            # Simple console monitoring
├── Makefile              # Easy commands
├── main.py               # (original entry point)
├── requirements.txt      # Dependencies
└── src/
    ├── train.py          # DQN training logic
    ├── q_function.py     # Q-network architecture
    └── replay_memory.py  # Experience replay
```

## Training Parameters

Key hyperparameters in `src/train.py`:
- `epsilon_decay`: 0.9999 (exploration decay rate)
- `discount`: 0.9 (future reward discount)
- `batch_size`: 32
- `memory_size`: 500
- `update_target_q_every`: 10,000 steps
- Checkpoint interval: 100 episodes (configurable via `--checkpoint-every`)

## Tips

- Training will take **hours to days** to see meaningful progress
- Start with `make train-short` to verify everything works
- Use `make monitor` for simple monitoring without GUI
- Checkpoints are saved every 100 episodes automatically
- The `latest.pt` checkpoint is always overwritten with the most recent state
- Press Ctrl+C during training to save an `interrupted.pt` checkpoint
