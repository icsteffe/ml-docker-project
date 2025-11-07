#!/usr/bin/env python3
"""
Main training script for DistilBERT hyperparameter tuning.

Usage:
    python main.py --checkpoint_dir models --lr 1e-3
    python main.py --checkpoint_dir models --lr 3e-5 --weight_decay 0.1 --warmup_ratio 0.2
    python main.py --sweep --method bayes --count 12
"""
import argparse
import json
import os
from pathlib import Path

import wandb

from src import train_model, create_sweep_config, set_seed
from src.trainer import create_run_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DistilBERT on GLUE tasks with hyperparameter tuning"
    )
    
    # Training arguments
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="models",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--lr", "--learning_rate",
        type=float, 
        default=3e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.2,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Model and task arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mrpc",
        help="GLUE task name"
    )
    
    # W&B arguments
    parser.add_argument(
        "--project_name",
        type=str,
        default="MLOPS_p2_distilbert",
        help="W&B project name"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    
    # Sweep arguments
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run hyperparameter sweep instead of single training"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["bayes", "grid", "random"],
        default="bayes",
        help="Sweep method"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of sweep runs"
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: dict, output_path: str):
    """Save configuration to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def run_single_training(args):
    """Run a single training run."""
    # Create configuration from args
    config = {
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.batch_size,
    }

    # Load from config file if provided
    if args.config:
        file_config = load_config(args.config)
        config.update(file_config)

    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create unique run name based on hyperparameters
    run_name = create_run_name(config)
    print(f"\nRun name: {run_name}")

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration with hyperparameter-based filename
    config_filename = f"{run_name}_config.json"
    save_config(config, Path(args.checkpoint_dir) / config_filename)

    # Train model
    results = train_model(
        config=config,
        project_name=args.project_name,
        model_name=args.model_name,
        task_name=args.task_name,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.max_epochs,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )

    print("\nTraining completed!")
    print("Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    # Save results with hyperparameter-based filename
    results_filename = f"{run_name}_results.json"
    save_config(results, Path(args.checkpoint_dir) / results_filename)

    return results


def run_sweep(args):
    """Run hyperparameter sweep."""
    if args.no_wandb:
        raise ValueError("Sweeps require W&B. Remove --no_wandb flag.")
    
    # Create sweep configuration
    sweep_config = create_sweep_config(method=args.method)
    
    print(f"Creating {args.method} sweep with {args.count} runs...")
    print("Sweep configuration:")
    print(json.dumps(sweep_config, indent=2))
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project_name)
    
    print(f"Sweep ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/{wandb.api.default_entity}/{args.project_name}/sweeps/{sweep_id}")
    
    # Define sweep function
    def sweep_train():
        with wandb.init() as run:
            config = dict(wandb.config)
            
            # No fixed parameters needed - all provided by sweep config
            
            results = train_model(
                config=config,
                project_name=args.project_name,
                model_name=args.model_name,
                task_name=args.task_name,
                checkpoint_dir=args.checkpoint_dir,
                max_epochs=args.max_epochs,
                seed=args.seed,
                use_wandb=True,
            )
            
            # Log final metrics
            if results:
                wandb.log({
                    "final_val_accuracy": results.get("val/accuracy", 0),
                    "final_val_f1": results.get("val/f1", 0),
                    "final_val_loss": results.get("val/loss", 0),
                })
    
    # Run sweep
    wandb.agent(sweep_id, function=sweep_train, count=args.count)
    
    print(f"\nSweep completed! View results at:")
    print(f"https://wandb.ai/{wandb.api.default_entity}/{args.project_name}/sweeps/{sweep_id}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    if args.sweep:
        run_sweep(args)
    else:
        run_single_training(args)


if __name__ == "__main__":
    main()