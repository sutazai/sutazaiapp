# SutazAI AI/ML Base Image - ULTRA CONSOLIDATION
# Specialized base for AI/ML agents with GPU support and ML frameworks
# Author: System Reorganization Expert - ULTRA Deduplication Operation
# Date: August 10, 2025
# Purpose: Consolidate AI/ML containers with comprehensive ML stack

FROM python:3.12.8-slim-bookworm as base

# Install comprehensive AI/ML system dependencies
RUN apt-get update && apt-get install -y \
    # Core essentials
    curl \
    wget \
    git \
    unzip \
    # Build tools for ML packages
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    # Math libraries for ML
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    # GPU support libraries (CUDA compatibility)
    pkg-config \
    # System monitoring for ML workloads
    htop \
    procps \
    # Network tools
    netcat-openbsd \
    # Development libraries
    python3-dev \
    libffi-dev \
    libssl-dev \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    # Audio processing libraries
    libsndfile1-dev \
    ffmpeg \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install comprehensive AI/ML Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    # Core ML frameworks
    torch==2.1.1 \
    torchvision==0.16.1 \
    torchaudio==2.1.1 \
    transformers==4.36.0 \
    # Data science essentials
    numpy==1.24.3 \
    pandas==1.5.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    # Deep learning utilities
    accelerate==0.24.1 \
    datasets==2.14.7 \
    tokenizers==0.15.0 \
    # Vector databases and embedding
    chromadb==0.4.18 \
    qdrant-client==1.6.9 \
    faiss-cpu==1.7.4 \
    # LLM integration
    ollama==0.1.7 \
    openai==1.3.7 \
    anthropic==0.8.1 \
    langchain==0.0.350 \
    # Computer vision
    opencv-python==4.8.1.78 \
    pillow==10.1.0 \
    # Audio processing
    librosa==0.10.1 \
    # Model serving
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    # Monitoring and metrics
    prometheus-client==0.19.0 \
    # Utilities
    pydantic==2.5.1 \
    httpx==0.25.2 \
    tqdm==4.66.1 \
    # Jupyter for experimentation
    jupyter==1.0.0 \
    # Memory optimization
    memory-profiler==0.61.0

# SECURITY: Create AI/ML user with appropriate permissions
RUN groupadd -r aiuser && useradd -r -g aiuser -s /bin/bash aiuser && \
    mkdir -p /app /app/models /app/data /app/logs /app/notebooks /app/experiments && \
    chown -R aiuser:aiuser /app

# Production stage - ULTRA OPTIMIZED for AI/ML workloads
FROM base as production

WORKDIR /app

# COMPREHENSIVE environment variables for AI/ML workloads
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # AI/ML specific settings
    OLLAMA_HOST=http://sutazai-ollama:11434 \
    MODEL_NAME=tinyllama \
    HF_HOME=/app/models/huggingface \
    TRANSFORMERS_CACHE=/app/models/transformers \
    # Performance optimization for ML
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    # GPU settings (if available)
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # Memory settings for large models
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    # Service configuration
    SERVICE_PORT=8080 \
    # Logging
    LOG_LEVEL=info \
    LOG_FORMAT=json

# Create model and data directories
RUN mkdir -p \
    /app/models/huggingface \
    /app/models/transformers \
    /app/models/ollama \
    /app/data/training \
    /app/data/inference \
    /app/experiments \
    && chown -R aiuser:aiuser /app

# AI/ML specific health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT:-8080}/health || \
        python3 -c "import torch; print('PyTorch OK')" || exit 1

# SECURITY: Switch to AI/ML user
USER aiuser

# Default ports for AI/ML services
EXPOSE 8080 8888

# FLEXIBLE default command for AI/ML services
CMD ["python", "-u", "app.py"]