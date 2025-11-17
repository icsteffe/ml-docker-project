# MLOps Project 2: DistilBERT Hyperparameter Tuning

## Overview

A production-ready MLOps pipeline for fine-tuning DistilBERT on GLUE tasks (MRPC) with automated hyperparameter optimization and experiment tracking. The project implements Bayesian hyperparameter search using Weights & Biases, containerized training workflows, and modular architecture for reproducible NLP model development.

## Architecture

**Key Components:**
- **Model**: DistilBERT-base-uncased with PyTorch Lightning wrapper
- **Data**: GLUE MRPC task (Microsoft Research Paraphrase Corpus)
- **Training**: PyTorch Lightning with mixed precision and gradient accumulation
- **Optimization**: W&B Sweeps (Bayesian, Grid, Random search)
- **Tracking**: Weights & Biases experiment logging
- **Deployment**: Docker containerization with environment isolation

**Project Structure:**
```
mlops-project2/
├── src/                    # Modular source code
│   ├── data_module.py      # GLUE data loading
│   ├── model.py           # DistilBERT Lightning module
│   ├── trainer.py         # Training utilities
│   └── __init__.py
├── config/                # Configuration files
│   ├── default_config.json     # Standard hyperparameters
│   └── optimal_config.json     # Best parameters from tuning
├── models/                # Output directory for trained models
├── main.py               # Main training script
├── Dockerfile            # Container definition
└── requirements.txt      # Python dependencies
```

## Key Results

Best hyperparameters found through Bayesian optimization, values can be found in optimal_config.json:

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 86.03% |
| **F1 Score** | 90.45% |
| **Validation Loss** | 0.3552 |

**Optimal Hyperparameters:**
- Learning Rate: 2.98e-5
- Weight Decay: 0.0875
- Warmup Ratio: 0.181
- Batch Size: 16

## Configuration

### Available Hyperparameters : default values
```json
{
  "learning_rate": 1e-5,              // AdamW learning rate
  "weight_decay": 0.1,                // L2 regularization
  "warmup_ratio": 0.2,                // LR warmup fraction
  "per_device_train_batch_size": 16,  // Batch size per GPU
  "gradient_accumulation_steps": 1,   // Gradient accumulation
  "optimizer_type": "adamw_torch",    // Optimizer
  "lr_scheduler_type": "linear",      // LR scheduler
  "classifier_dropout": 0.1           // Dropout rate
}
```

### Command-Line Arguments
```bash
# Training
--lr, --learning_rate    Learning rate (default: 1e-5)
--weight_decay          Weight decay (default: 0.1)
--warmup_ratio          Warmup ratio (default: 0.2)
--batch_size            Batch size (default: 16)
--max_epochs            Training epochs (default: 3)
--seed                  Random seed (default: 42)

# Model & Task
--model_name            Pretrained model (default: distilbert-base-uncased)
--task_name             GLUE task (default: mrpc)

# Paths
--checkpoint_dir        Model save directory (default: models)
--config                Path to JSON config file

# W&B
--project_name          W&B project name (default: MLOPS_p2_distilbert)
--no_wandb              Disable W&B logging

# Sweeps
--sweep                 Enable hyperparameter sweep
--method                Sweep method: bayes|grid|random (default: bayes)
--count                 Number of sweep runs (default: 12)
```

## Prerequisites

### 1. Weights & Biases Account
- Create account at [wandb.ai](https://wandb.ai)
- Get API key from [wandb.ai/authorize](https://wandb.ai/authorize)
- Note your username/entity name

### 2. Docker Desktop
- Required for containerized training
- Install from [docker.com](https://www.docker.com/products/docker-desktop)
- Ensure Docker daemon is running

## Running Training

### Local Training (Codebase)

**Setup:**
```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Login to W&B
wandb login
```

**Training with optimal config:**
```bash
python main.py --config config/optimal_config.json --max_epochs 3
```

**Training with custom hyperparameters:**
```bash
python main.py --lr 2e-5 --weight_decay 0.1 --warmup_ratio 0.15 --max_epochs 5
```

### Docker Training (Local/Cloud)

**Setup:**
```bash
# 1. Create .env file with W&B credentials
cp .env.example .env

# 2. Edit .env with your credentials:
#    WANDB_API_KEY=your_api_key_here
#    WANDB_ENTITY=your_username
#    WANDB_PROJECT=MLOPS_p2_distilbert_docker
```

**Run with Docker:**
```bash
# Build image
docker build -t mlops-distilbert .

# Run training with optimal config
docker run --rm --env-file .env -v ${PWD}/models:/app/models mlops-distilbert

# Run with custom config
docker run --rm --env-file .env -v ${PWD}/models:/app/models mlops-distilbert \
  python main.py --lr 3e-5 --weight_decay 0.12 --max_epochs 5
```

**Run with Docker Compose:**
```bash
# Single training run
docker-compose up training

# Custom hyperparameters
docker-compose --profile custom up training-custom

# Hyperparameter sweep
docker-compose --profile sweep up sweep
```

### Cloud Deployment
Deploy the same Docker container to any cloud platform:
- **AWS**: ECS, SageMaker, EC2
- **Google Cloud**: Cloud Run, AI Platform, Compute Engine
- **Azure**: Container Instances, ML

The container is self-contained with all dependencies and will run identically across environments.

## Hyperparameter Sweeps

W&B Sweeps automatically explore hyperparameter spaces to find optimal configurations.

**Run sweep:**
```bash
# Bayesian optimization (recommended)
python main.py --sweep --method bayes --count 12

# Grid search (exhaustive)
python main.py --sweep --method grid --count 20

# Random search
python main.py --sweep --method random --count 15
```

**Sweep search ranges:**
- Learning Rate: 1e-5 to 1e-4 (explored on log/exponential scale)
- Weight Decay: 0.08 to 0.12 (linear scale)
- Warmup Ratio: 0.15 to 0.25 (linear scale)

The learning rate is sampled on a logarithmic scale to effectively cover different orders of magnitude, which is critical for finding optimal learning rates in deep learning.

View real-time results in W&B dashboard. The sweep will automatically track best runs and hyperparameters.

## Monitoring

- **W&B Dashboard**: Real-time metrics at `wandb.ai/{entity}/{project}`
- **Model Checkpoints**: Saved to `./models/` directory
- **Config Files**: Saved as `{run_name}_config.json`
- **Results**: Saved as `{run_name}_results.json`

## Quick Start

```bash
# Fastest way to reproduce optimal results
git clone <repo-url>
cd mlops-project2
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
wandb login
python main.py --config config/optimal_config.json --max_epochs 3
```