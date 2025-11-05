"""
Training utilities and helper functions.
"""
import os
import random
from pathlib import Path
from typing import Dict, Any

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from .data_module import GLUEDataModule
from .model import GLUETransformer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_name(config: Dict[str, Any]) -> str:
    """Create descriptive run name from hyperparameters."""
    lr = config.get('learning_rate', config.get('lr', 'unknown'))
    wd = config.get('weight_decay', 'unknown')
    wr = config.get('warmup_ratio', 'unknown')
    return f"lr{lr}_wd{wd}_wr{wr}"


def train_model(
    config: Dict[str, Any],
    project_name: str = "MLOPS_p2_distilbert",
    model_name: str = "distilbert-base-uncased",
    task_name: str = "mrpc",
    checkpoint_dir: str = "models",
    max_epochs: int = 3,
    seed: int = 42,
    use_wandb: bool = True,
    **kwargs
):
    """
    Train a GLUE model with given hyperparameters.
    
    Args:
        config: Dictionary containing hyperparameters
        project_name: W&B project name
        model_name: Pretrained model name
        task_name: GLUE task name
        checkpoint_dir: Directory to save model checkpoints
        max_epochs: Number of training epochs
        seed: Random seed
        use_wandb: Whether to use W&B logging
        **kwargs: Additional arguments
    
    Returns:
        Dict with final evaluation results
    """
    set_seed(seed)
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    # Setup W&B logging
    logger = None
    if use_wandb:
        run_name = create_run_name(config)
        logger = WandbLogger(
            project=project_name,
            name=run_name,
            config=config
        )
    
    # Data module
    dm = GLUEDataModule(
        model_name_or_path=model_name,
        task_name=task_name,
        max_seq_length=config.get('max_seq_length', 128),
        train_batch_size=config.get('per_device_train_batch_size', 16),
        eval_batch_size=config.get('per_device_eval_batch_size', 16),
    )
    dm.setup("fit")
    
    # Model
    model = GLUETransformer(
        model_name_or_path=model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=task_name,
        learning_rate=config.get('learning_rate', 2e-5),
        weight_decay=config.get('weight_decay', 0.0),
        warmup_steps=int(config.get('warmup_ratio', 0.0) * 
                        (len(dm.train_dataloader()) * max_epochs)),
        train_batch_size=config.get('per_device_train_batch_size', 16),
        eval_batch_size=config.get('per_device_eval_batch_size', 16),
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        default_root_dir=checkpoint_dir,
        enable_checkpointing=True,
        deterministic=True,
    )
    
    # Train
    trainer.fit(model, datamodule=dm)
    
    # Final evaluation
    results = trainer.validate(model, datamodule=dm)
    if results:
        results = results[0]  # Take first validation result
    
    return results


def create_sweep_config(
    learning_rate_range: tuple = (2.5e-5, 4e-5),
    weight_decay_range: tuple = (0.08, 0.12),
    warmup_ratio_range: tuple = (0.15, 0.25),
    method: str = "bayes"
) -> Dict[str, Any]:
    """
    Create W&B sweep configuration.
    
    Args:
        learning_rate_range: (min, max) learning rate
        weight_decay_range: (min, max) weight decay
        warmup_ratio_range: (min, max) warmup ratio
        method: Sweep method ('bayes', 'grid', 'random')
    
    Returns:
        Sweep configuration dictionary
    """
    sweep_config = {
        "name": f"hyperparameter_sweep_{method}",
        "method": method,
        "metric": {
            "name": "val/accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "learning_rate": {
                "distribution": "uniform",
                "min": learning_rate_range[0],
                "max": learning_rate_range[1]
            },
            "weight_decay": {
                "distribution": "uniform",
                "min": weight_decay_range[0],
                "max": weight_decay_range[1]
            },
            "warmup_ratio": {
                "distribution": "uniform",
                "min": warmup_ratio_range[0],
                "max": warmup_ratio_range[1]
            },
            # Fixed parameters
            "per_device_train_batch_size": {"value": 16},
            "per_device_eval_batch_size": {"value": 16},
            "max_seq_length": {"value": 128},
        }
    }
    
    if method == "grid":
        # Convert to discrete values for grid search
        sweep_config["parameters"]["learning_rate"] = {
            "values": [2.5e-5, 3e-5, 3.5e-5, 4e-5]
        }
        sweep_config["parameters"]["weight_decay"] = {
            "values": [0.08, 0.09, 0.1, 0.11, 0.12]
        }
        sweep_config["parameters"]["warmup_ratio"] = {
            "values": [0.15, 0.18, 0.2, 0.22, 0.25]
        }
    
    return sweep_config