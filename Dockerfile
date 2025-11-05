# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY main.py .

# Create directories for outputs
RUN mkdir -p /app/models /app/outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV WANDB_CACHE_DIR=/app/.cache/wandb
ENV HF_HOME=/app/.cache/huggingface

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command runs training with optimal hyperparameters
CMD ["python", "main.py", \
     "--checkpoint_dir", "/app/models", \
     "--lr", "3e-5", \
     "--weight_decay", "0.120", \
     "--warmup_ratio", "0.240", \
     "--batch_size", "16", \
     "--max_epochs", "3", \
     "--project_name", "MLOPS_p2_distilbert_docker"]