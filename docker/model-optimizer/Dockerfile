FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for model optimization
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    torch \
    torchvision \
    torchaudio \
    transformers \
    optimum \
    accelerate \
    bitsandbytes \
    deepspeed \
    fairscale \
    onnx \
    onnxruntime \
    tensorrt \
    intel-extension-for-pytorch \
    py-cpuinfo \
    psutil \
    gputil \
    nvidia-ml-py3 \
    requests \
    aiohttp \
    asyncio \
    numpy \
    scipy \
    structlog \
    prometheus-client \
    python-dotenv

# Copy optimizer service files
COPY model_optimizer.py requirements.txt* ./
COPY optimizer_config.yaml ./

# Install additional requirements if they exist
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Create directories
RUN mkdir -p /app/data /app/logs /app/models /app/optimized_models /app/cache

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OPTIMIZATION_MODE=performance
ENV AUTO_OPTIMIZE=true
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start model optimizer
CMD ["python", "model_optimizer.py"] 