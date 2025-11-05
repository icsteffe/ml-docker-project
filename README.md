# MLOps Project 2: DistilBERT Hyperparameter Tuning

This project adapts the Jupyter notebook from Project 1 into a structured MLOps pipeline for training DistilBERT on GLUE tasks with comprehensive hyperparameter tuning and experiment tracking.

## Project Structure

```
mlops-project2/
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── data_module.py      # Lightning DataModule for GLUE tasks
│   ├── model.py           # DistilBERT Lightning module
│   └── trainer.py         # Training utilities and sweep configs
├── config/                # Configuration files
│   └── default_config.json
├── scripts/               # Convenience scripts
│   ├── run_training.py
│   └── run_sweep.py
├── models/                # Model checkpoints and outputs
├── data/                  # Dataset cache (auto-created)
├── main.py               # Main training script
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/Mac:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Login to Weights & Biases:**
   ```bash
   wandb login
   ```

## Usage

### Single Training Run

Train with default hyperparameters:
```bash
python main.py --checkpoint_dir models --lr 3e-5
```

Train with custom hyperparameters:
```bash
python main.py --checkpoint_dir models --lr 1e-3 --weight_decay 0.1 --warmup_ratio 0.2
```

Train with configuration file:
```bash
python main.py --checkpoint_dir models --config config/default_config.json
```

### Hyperparameter Sweeps

Run Bayesian optimization sweep:
```bash
python main.py --sweep --method bayes --count 12 --checkpoint_dir models/sweep
```

Run grid search sweep:
```bash
python main.py --sweep --method grid --count 20 --checkpoint_dir models/grid
```

Run random search sweep:
```bash
python main.py --sweep --method random --count 15 --checkpoint_dir models/random
```

### Command Line Arguments

#### Training Arguments
- `--checkpoint_dir`: Directory to save model checkpoints (default: "models")
- `--lr, --learning_rate`: Learning rate (default: 3e-5)
- `--weight_decay`: Weight decay (default: 0.1)
- `--warmup_ratio`: Warmup ratio (default: 0.2)
- `--batch_size`: Training batch size (default: 16)
- `--max_epochs`: Maximum training epochs (default: 3)
- `--seed`: Random seed (default: 42)

#### Model and Task Arguments
- `--model_name`: Pretrained model name (default: "distilbert-base-uncased")
- `--task_name`: GLUE task name (default: "mrpc")

#### W&B Arguments
- `--project_name`: W&B project name (default: "MLOPS_p2_distilbert")
- `--no_wandb`: Disable W&B logging

#### Sweep Arguments
- `--sweep`: Run hyperparameter sweep
- `--method`: Sweep method (bayes/grid/random, default: "bayes")
- `--count`: Number of sweep runs (default: 12)

#### Configuration
- `--config`: Path to JSON configuration file

## Features

### Modular Architecture
- **Data Module**: Handles GLUE dataset loading and preprocessing
- **Model Module**: DistilBERT Lightning module with configurable hyperparameters
- **Trainer Module**: Training utilities, sweep configurations, and helper functions

### Experiment Tracking
- Full Weights & Biases integration
- Automatic run naming based on hyperparameters
- Comprehensive metric logging
- Sweep support for automated hyperparameter optimization

### Hyperparameter Optimization
- Support for Bayesian, Grid, and Random search
- Configurable search spaces
- Built-in optimal ranges based on Project 1 results

### Reproducibility
- Fixed random seeds
- Deterministic training
- Configuration saving and loading
- Checkpoint management

## Example Commands

```bash
# Basic training
python main.py --checkpoint_dir models --lr 3e-5

# Training with optimal config from Project 1
python main.py --checkpoint_dir models --lr 3e-5 --weight_decay 0.12 --warmup_ratio 0.24

# Bayesian optimization around optimal region
python main.py --sweep --method bayes --count 12

# Grid search with limited runs
python main.py --sweep --method grid --count 20

# Training without W&B
python main.py --checkpoint_dir models --lr 2e-5 --no_wandb
```

## Configuration Files

The `config/default_config.json` contains the optimal hyperparameters found in Project 1:

```json
{
  "learning_rate": 3e-5,
  "weight_decay": 0.1,
  "warmup_ratio": 0.2,
  "per_device_train_batch_size": 16,
  "per_device_eval_batch_size": 16,
  "max_seq_length": 128,
  "optimizer_type": "adamw_torch",
  "lr_scheduler_type": "linear",
  "classifier_dropout": 0.1
}
```

## Results from Project 1

- **Manual Tuning (Week 2)**: Best accuracy 85.78%
- **Bayesian Optimization (Week 3)**: Best accuracy 86.03%
- **Optimal Configuration**: LR=3e-5, WD=0.12, WR=0.24

## Development

To extend the project:

1. **Add new models**: Modify `src/model.py`
2. **Add new tasks**: Extend `src/data_module.py`
3. **Customize training**: Modify `src/trainer.py`
4. **Add new sweep strategies**: Update sweep configs in `src/trainer.py`

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- PyTorch Lightning
- Transformers
- Datasets
- Weights & Biases
- Evaluate