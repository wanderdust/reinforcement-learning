.PHONY: help train play monitor tensorboard clean install

# Auto-detect if using uv
PYTHON := $(shell command -v uv > /dev/null 2>&1 && echo "uv run python" || echo "python")

help:
	@echo "DQN Training Makefile Commands"
	@echo "================================"
	@echo "make install          - Install dependencies"
	@echo "make train            - Start training (infinite episodes)"
	@echo "make train-short      - Train for 1000 episodes (quick test)"
	@echo "make resume           - Resume training from latest checkpoint"
	@echo "make play             - Watch trained agent play (5 episodes)"
	@echo "make monitor          - Monitor training progress in real-time"
	@echo "make tensorboard      - Launch TensorBoard for detailed metrics"
	@echo "make clean            - Remove checkpoints and logs"
	@echo "make clean-logs       - Remove only logs (keep checkpoints)"

install:
	@if command -v uv > /dev/null 2>&1; then \
		echo "Using uv..."; \
		uv pip install -r requirements.txt; \
	else \
		pip install -r requirements.txt; \
	fi

train:
	$(PYTHON) train.py

train-short:
	$(PYTHON) train.py --max-episodes 1000

resume:
	$(PYTHON) train.py --resume latest.pt

play:
	$(PYTHON) play.py

play-10:
	$(PYTHON) play.py --episodes 10

monitor:
	$(PYTHON) monitor.py

tensorboard:
	@echo "Starting TensorBoard..."
	@echo "Open http://localhost:6006 in your browser"
	@if command -v uv > /dev/null 2>&1; then \
		uv run python -m tensorboard.main --logdir=tensorboard_logs; \
	else \
		python -m tensorboard.main --logdir=tensorboard_logs; \
	fi

clean:
	rm -rf checkpoints/ tensorboard_logs/
	@echo "Cleaned checkpoints and logs"

clean-logs:
	rm -rf tensorboard_logs/
	@echo "Cleaned logs (checkpoints preserved)"
