# MLOps Project 2: DistilBERT Hyperparameter Tuning

A complete MLOps pipeline for training DistilBERT on GLUE tasks with automated hyperparameter tuning and experiment tracking.

## Quick Setup

```bash
# 1. Clone and navigate to project
cd mlops-project2

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

### Test Locally
```bash
python main.py --checkpoint_dir models --config config/optimal_config.json --max_epochs 3
```

### Docker (Production)
```bash
# 1. Setup environment
cp .env.example .env
```
#### .env example
```bash
# Edit .env with your W&B credentials
# Environment variables for Docker containers
# Copy this file to .env and fill in your values

# Weights & Biases API Key
# Get from: https://wandb.ai/authorize
WANDB_API_KEY=your_wandb_api_key_here

# W&B entity (username/team name)
WANDB_ENTITY=your_username_or_team

# W&B project name
WANDB_PROJECT=MLOPS_p2_distilbert_docker
```

# 2. Build and run container
```bash
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