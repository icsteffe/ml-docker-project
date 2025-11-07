# MLOps Project 2: DistilBERT Hyperparameter Tuning

A complete MLOps pipeline for training DistilBERT on GLUE tasks with automated hyperparameter tuning and experiment tracking. This project adapts a Jupyter notebook into a production-ready containerized training system.

## Quick Setup

### Local Development
```bash
# 1. Clone and navigate to project
cd mlops-project2

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training with optimal hyperparameters
python main.py --checkpoint_dir models --lr 3e-05 --weight_decay 0.08754104905198969 --warmup_ratio 0.1814720791654623
```

### Docker (Production)
```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your W&B credentials

# 2. Build and run container
docker build -t mlops-distilbert .
docker run --rm --env-file .env -v ${PWD}/models:/app/models mlops-distilbert

```

## Project Structure

```
mlops-project2/
├── src/                    # Modular source code
│   ├── data_module.py      # GLUE data loading
│   ├── model.py           # DistilBERT Lightning module  
│   ├── trainer.py         # Training utilities
│   └── __init__.py
├── config/                # Configuration files
│   ├── default_config.json     # Standard hyperparameters
│   └── optimal_config.json     # Best parameters from Project 1
├── scripts/               # Utility scripts
├── models/                # Output directory for trained models
├── main.py               # Main training script
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-service deployment
├── requirements.txt      # Python dependencies
└── .env.example         # Environment template
```

## Key Features

- ** Optimal Hyperparameters**: Pre-configured with best parameters from Project 1 (86.03% accuracy)
- ** Experiment Tracking**: Full W&B integration with automatic run naming
- ** Containerized**: Docker support for consistent deployments
- ** Reproducible**: Fixed seeds and deterministic training
- ** Multiple Search Methods**: Bayesian, Grid, and Random hyperparameter optimization

## Usage Examples

### Basic Training
```bash
# Use optimal hyperparameters (recommended)
python main.py --config config/optimal_config.json

# Custom hyperparameters  
python main.py --lr 2e-5 --weight_decay 0.1 --warmup_ratio 0.15
```

### Hyperparameter Sweeps
```bash
# Bayesian optimization (12 runs)
python main.py --sweep --method bayes --count 12

# Grid search
python main.py --sweep --method grid --count 20
```

### Docker Deployment
```bash
# Single training run
docker-compose up training

# Custom hyperparameters
docker-compose --profile custom up training-custom

# Hyperparameter sweep
docker-compose --profile sweep up sweep
```

## Optimal Results

Based on extensive hyperparameter optimization in Project 1:

| Parameter | Value | Performance |
|-----------|-------|-------------|
| Learning Rate | 3e-5 | **86.03% Accuracy** |
| Weight Decay | 0.0875 | **90.45% F1 Score** |
| Warmup Ratio | 0.1815 | **0.3552 Loss** |

The model automatically names runs as: `lr3e-05_wd0.088_wr0.181`

## Environment Setup

Create `.env` file with your W&B credentials:

```bash
# Weights & Biases Configuration
WANDB_API_KEY=your_api_key_here
WANDB_ENTITY=your_username_or_team
WANDB_PROJECT=MLOPS_p2_distilbert_docker
```

## Monitoring & Logs

- **W&B Dashboard**: Automatic experiment tracking and metrics
- **Model Outputs**: Saved to `./models/` directory
- **Docker Logs**: `docker-compose logs training`

## Advanced Usage

### Custom Configuration example
```json
// config/custom_config.json
{
  "learning_rate": 2e-5,
  "weight_decay": 0.1,
  "warmup_ratio": 0.15,
  "per_device_train_batch_size": 32,
  "max_epochs": 5
}
```

### Multiple Environments
```bash
# Development
python main.py --no_wandb --max_epochs 1

# Production
docker run --env-file .env mlops-distilbert

# Experimentation  
python main.py --sweep --method random --count 50
```

## Development

### Adding New Models
1. Extend `src/model.py` with new architecture
2. Update `src/data_module.py` for new datasets
3. Modify `src/trainer.py` for custom training loops

### Testing
```bash
# Quick test run
python main.py --max_epochs 1 --no_wandb

# Docker test
docker run --rm mlops-distilbert python main.py --max_epochs 1
```

## Dependencies

**Core ML Stack:**
- PyTorch Lightning (training framework)
- Transformers (DistilBERT model)
- Datasets (GLUE data loading)
- Evaluate (metrics computation)

**MLOps Tools:**
- Weights & Biases (experiment tracking)
- Docker (containerization)
- NumPy, Pandas (data processing)

## Project Evolution

**From Project 1:** Jupyter notebook exploration → **To Project 2:** Production MLOps pipeline

- Converted notebook to modular Python scripts
- Added Docker containerization
- Integrated experiment tracking
- Implemented automated hyperparameter optimization
- Created reproducible training pipeline

---

* Ready to train? Start with:** `python main.py --config config/optimal_config.json`