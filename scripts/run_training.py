#!/usr/bin/env python3
"""
Convenience script for running training with different configurations.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run shell command and print output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result


def main():
    """Run example training configurations."""
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    
    print("Running example training configurations...")
    print("=" * 50)
    
    # Example 1: Basic training with default parameters
    print("\n1. Basic training with default parameters:")
    cmd = [
        sys.executable, "main.py",
        "--checkpoint_dir", "models/basic",
        "--lr", "3e-5",
        "--weight_decay", "0.1",
        "--warmup_ratio", "0.2"
    ]
    run_command(cmd)
    
    # Example 2: Training with config file
    print("\n2. Training with config file:")
    cmd = [
        sys.executable, "main.py",
        "--checkpoint_dir", "models/config_based",
        "--config", "config/default_config.json"
    ]
    run_command(cmd)
    
    # Example 3: Training with different hyperparameters
    print("\n3. Training with different hyperparameters:")
    cmd = [
        sys.executable, "main.py",
        "--checkpoint_dir", "models/custom",
        "--lr", "2e-5",
        "--weight_decay", "0.05",
        "--warmup_ratio", "0.15",
        "--batch_size", "32"
    ]
    run_command(cmd)
    
    print("\n" + "=" * 50)
    print("All training runs completed successfully!")
    print("Check the models/ directory for outputs.")


if __name__ == "__main__":
    main()