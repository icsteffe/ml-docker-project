# Docker Deployment Guide

This guide shows how to build and run the containerized DistilBERT training pipeline.

## Quick Start

### 1. Set up environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your W&B API key
# WANDB_API_KEY=your_actual_api_key_here
```

### 2. Build the Docker image
```bash
docker build -t mlops-distilbert .
```

### 3. Run training with optimal hyperparameters
```bash
# Using Docker Compose (recommended)
docker-compose up training

# Or using Docker directly
docker run --rm \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  mlops-distilbert
```

## Deployment Options

### Option 1: Docker Compose (Recommended)

**Run with optimal hyperparameters:**
```bash
docker-compose up training
```

**Run with custom hyperparameters:**
```bash
docker-compose --profile custom up training-custom
```

**Run hyperparameter sweep:**
```bash
docker-compose --profile sweep up sweep
```

### Option 2: Docker Commands

**Basic training run:**
```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config:ro \
  mlops-distilbert \
  python main.py --config /app/config/optimal_config.json --checkpoint_dir /app/models
```

**Custom hyperparameters:**
```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  mlops-distilbert \
  python main.py --lr 2e-5 --weight_decay 0.1 --warmup_ratio 0.15 --checkpoint_dir /app/models
```

**Hyperparameter sweep:**
```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  mlops-distilbert \
  python main.py --sweep --method bayes --count 8 --checkpoint_dir /app/models
```

**Run without W&B logging:**
```bash
docker run --rm \
  -e WANDB_MODE=disabled \
  -v $(pwd)/models:/app/models \
  mlops-distilbert
```

## Optimal Hyperparameters

The container runs with these optimal hyperparameters by default:
- **Learning Rate**: 3e-5
- **Weight Decay**: 0.120  
- **Warmup Ratio**: 0.240
- **Batch Size**: 16
- **Max Epochs**: 3

These were found through Bayesian optimization and achieved:
- **Accuracy**: 86.03%
- **F1 Score**: 90.36%
- **Loss**: 0.3912

## Volume Mounts

- `./models:/app/models` - Persist trained models and checkpoints
- `./config:/app/config:ro` - Mount configuration files (read-only)
- Cache volumes for HuggingFace models and W&B data

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WANDB_API_KEY` | W&B API key for experiment tracking | No* |
| `WANDB_ENTITY` | W&B username/team name | No |
| `WANDB_PROJECT` | W&B project name | No |
| `WANDB_MODE` | Set to "disabled" to run without W&B | No |

*Required for W&B logging

## Troubleshooting

**Build fails:**
```bash
# Clean build with no cache
docker build --no-cache -t mlops-distilbert .
```

**Permission errors:**
```bash
# Fix file permissions on Linux/Mac
chmod +x scripts/docker_entrypoint.sh
```

**Out of memory:**
```bash
# Reduce batch size
docker run --rm --env-file .env -v $(pwd)/models:/app/models \
  mlops-distilbert python main.py --batch_size 8 --checkpoint_dir /app/models
```

**Check logs:**
```bash
# View container logs
docker-compose logs training

# Run interactively for debugging
docker run -it --env-file .env mlops-distilbert /bin/bash
```

## Performance Tips

1. **Use GPU**: Add `--gpus all` to docker run commands if you have NVIDIA GPU support
2. **Persistent cache**: The compose file creates volumes for HuggingFace and W&B caches
3. **Batch size**: Adjust based on available memory (16 is optimal for most systems)
4. **Multi-run**: Use the sweep functionality for systematic hyperparameter exploration