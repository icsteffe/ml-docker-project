#!/bin/bash
set -e

echo "Starting MLOps DistilBERT Training Container"
echo "=============================================="

# Print environment info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Working directory: $(pwd)"
echo "Available disk space: $(df -h . | tail -1 | awk '{print $4}')"

# Check if W&B API key is set
if [ -n "$WANDB_API_KEY" ]; then
    echo "W&B API key found - logging enabled"
    wandb login --relogin
else
    echo "Warning: No W&B API key found - training will run without logging"
    export WANDB_MODE=disabled
fi

# Create output directories
mkdir -p /app/models /app/outputs

# Print configuration
echo ""
echo "Training Configuration:"
echo "======================"
if [ -f "/app/config/optimal_config.json" ]; then
    echo "Using optimal config:"
    cat /app/config/optimal_config.json | python -m json.tool
else
    echo "Using command line arguments: $@"
fi

echo ""
echo "Starting training..."
echo "===================="

# Execute the main command
exec "$@"