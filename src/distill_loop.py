#!/usr/bin/env python3
"""
Automated distillation loop:
  1. Generate Stockfish self-play data
  2. Deduplicate and split into train/test/validate
  3. Train the neural network
  4. Repeat

Usage: python3 distill_loop.py <num_cycles> <generate_seconds> <train_seconds>
Example: python3 distill_loop.py 10 1800 600
  = 10 cycles, 30 min generating per cycle, 10 min training per cycle
"""

import subprocess
import sys
import os
import shutil
import time
import torch
from alpha_net import ChessNet
import config

rootDir = config.rootDir
model_dir = f"{rootDir}/data/model_data"

def ensure_initial_model():
    """Create random.gz and latest.gz if they don't exist."""
    random_path = f"{model_dir}/random.gz"
    latest_path = f"{model_dir}/latest.gz"

    if not os.path.exists(random_path):
        print("Creating initial random network...")
        net = ChessNet()
        torch.save({'state_dict': net.state_dict()}, random_path)
        print(f"Saved {random_path}")

    if not os.path.exists(latest_path):
        shutil.copy(random_path, latest_path)
        print(f"Copied random.gz -> latest.gz")

def run_step(description, cmd):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"WARNING: {description} exited with code {result.returncode}")
    return result.returncode

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 distill_loop.py <num_cycles> <generate_seconds> <train_seconds>")
        print("Example: python3 distill_loop.py 10 1800 600")
        sys.exit(1)

    num_cycles = int(sys.argv[1])
    gen_seconds = int(sys.argv[2])
    train_seconds = int(sys.argv[3])

    ensure_initial_model()

    for cycle in range(1, num_cycles + 1):
        print(f"\n{'#'*60}")
        print(f"  CYCLE {cycle}/{num_cycles}")
        print(f"{'#'*60}")

        # Step 1: Generate data (unique run ID using timestamp)
        run_id = f"{int(time.time())}"
        run_step(
            f"Cycle {cycle}: Generating Stockfish self-play data ({gen_seconds}s)",
            [sys.executable, "generate.py", run_id, str(gen_seconds)]
        )

        # Step 2: Deduplicate
        run_step(
            f"Cycle {cycle}: Deduplicating positions",
            [sys.executable, "game_de_dup.py"]
        )

        # Step 3: Train
        run_step(
            f"Cycle {cycle}: Training network ({train_seconds}s)",
            [sys.executable, "train_one.py", str(cycle), "train", str(train_seconds)]
        )

        print(f"\nCycle {cycle} complete.")

    print(f"\nAll {num_cycles} cycles complete.")
    print(f"Trained model at: {model_dir}/latest.gz")
