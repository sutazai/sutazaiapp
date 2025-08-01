FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install FSDP and distributed training dependencies
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    transformers \
    accelerate \
    deepspeed \
    fairscale \
    wandb \
    tensorboard \
    scipy \
    scikit-learn \
    datasets \
    tokenizers

# Copy FSDP service files
COPY fsdp_service.py requirements.txt* ./

# Install additional requirements if they exist
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Create model storage directory
RUN mkdir -p /models /data

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_DISTRIBUTED_DEBUG=DETAIL

# Expose port for distributed training coordination
EXPOSE 29500

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('FSDP service healthy')" || exit 1

# Default command
CMD ["python", "fsdp_service.py"] 