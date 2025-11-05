#!/usr/bin/env python3
"""
Convenience script for running hyperparameter sweeps.
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
    """Run example sweep configurations."""
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    
    print("Running example sweep configurations...")
    print("=" * 50)
    
    # Example 1: Bayesian optimization sweep
    print("\n1. Bayesian optimization sweep (12 runs):")
    cmd = [
        sys.executable, "main.py",
        "--sweep",
        "--method", "bayes",
        "--count", "12",
        "--checkpoint_dir", "models/sweep_bayes"
    ]
    run_command(cmd)
    
    # Example 2: Grid search sweep
    print("\n2. Grid search sweep (limited runs):")
    cmd = [
        sys.executable, "main.py",
        "--sweep",
        "--method", "grid",
        "--count", "8",
        "--checkpoint_dir", "models/sweep_grid"
    ]
    run_command(cmd)
    
    # Example 3: Random search sweep
    print("\n3. Random search sweep (10 runs):")
    cmd = [
        sys.executable, "main.py",
        "--sweep",
        "--method", "random",
        "--count", "10",
        "--checkpoint_dir", "models/sweep_random"
    ]
    run_command(cmd)
    
    print("\n" + "=" * 50)
    print("All sweeps completed successfully!")
    print("Check W&B dashboard for results.")


if __name__ == "__main__":
    main()