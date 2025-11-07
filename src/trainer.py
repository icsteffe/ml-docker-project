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
from lightning.pytorch.callbacks import ModelCheckpoint

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
    """Create descriptive run name from hyperparameters (matches Project 1 notebook)."""
    lr = config.get('learning_rate', config.get('lr', 'unknown'))
    wd = config.get('weight_decay', 'unknown')
    wr = config.get('warmup_ratio', 'unknown')

    # Use raw values to match Project 1 notebook naming
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
        train_batch_size=config.get('per_device_train_batch_size', 16),
        seed=seed,
    )
    dm.setup("fit")
    
    # Calculate warmup steps
    warmup_ratio = config.get('warmup_ratio', 0.0)
    # Estimate total training steps: (dataset_size / batch_size) * epochs
    # We'll let Lightning calculate this properly during training
    warmup_steps = 0  # Will be calculated properly in the model
    
    # Model
    model = GLUETransformer(
        model_name_or_path=model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=task_name,
        learning_rate=config.get('learning_rate', 2e-5),
        weight_decay=config.get('weight_decay', 0.0),
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,  # Pass ratio to model for proper calculation
        train_batch_size=config.get('per_device_train_batch_size', 16),
        adam_beta1=config.get('adam_beta1', 0.9),
        adam_beta2=config.get('adam_beta2', 0.999),
        optimizer_type=config.get('optimizer_type', 'adamw_torch'),
        lr_scheduler_type=config.get('lr_scheduler_type', 'linear'),
        classifier_dropout=config.get('classifier_dropout', None),
    )
    
    # Model checkpoint callback - save best model by accuracy (matches Project 1)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best-checkpoint-{epoch:02d}-{accuracy:.4f}',
        monitor='accuracy',
        mode='max',
        save_top_k=1,
        save_last=False,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        default_root_dir=checkpoint_dir,
        callbacks=[checkpoint_callback],
        deterministic=True,
        accumulate_grad_batches=config.get('gradient_accumulation_steps', 1),
    )

    # Train
    trainer.fit(model, datamodule=dm)

    # Load best model checkpoint (matches Project 1's load_best_model_at_end=True)
    if checkpoint_callback.best_model_path:
        print(f"\nLoading best model from: {checkpoint_callback.best_model_path}")
        best_model = GLUETransformer.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            model_name_or_path=model_name,
            num_labels=dm.num_labels,
            eval_splits=dm.eval_splits,
            task_name=task_name,
        )
        # Run final validation with best model
        final_results = trainer.validate(best_model, datamodule=dm)
        results = final_results[0] if final_results else {}
    else:
        # Fallback to last epoch metrics
        results = trainer.callback_metrics

    return dict(results)


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