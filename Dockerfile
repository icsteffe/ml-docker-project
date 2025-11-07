# Lightweight Dockerfile using slim Python base
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir \
    transformers==4.46.2 \
    datasets>=3.0.1 \
    evaluate>=0.4.3 \
    lightning \
    wandb \
    accelerate>=0.34.0 \
    numpy \
    pandas \
    scipy \
    scikit-learn

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY main.py .

# Create output directories with proper permissions
RUN mkdir -p /app/models /app/.cache && chmod 777 /app/.cache

# Set environment variables for cache
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface

# Default command with optimal parameters
CMD ["bash", "-c", "wandb login $WANDB_API_KEY && python main.py --checkpoint_dir /app/models --lr 3e-05 --weight_decay 0.08754104905198969 --warmup_ratio 0.1814720791654623 --batch_size 16 --max_epochs 3"]