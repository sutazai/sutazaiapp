# Enhanced Model Manager for SutazAI
FROM python:3.11-slim

LABEL maintainer="SutazAI Team"
LABEL description="Enhanced model management with DeepSeek-Coder and Llama 2"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    aiohttp \
    transformers \
    torch \
    accelerate \
    bitsandbytes \
    huggingface-hub \
    sentencepiece \
    tokenizers \
    pydantic \
    python-multipart

# Copy enhanced model manager
COPY enhanced_model_service.py /app/
COPY model_manager.py /app/
COPY deepseek_integration.py /app/
COPY requirements.txt /app/

# Create necessary directories
RUN mkdir -p /data/models /data/cache /logs

# Set permissions
RUN chmod +x /app/enhanced_model_service.py

# Expose port
EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Start enhanced model manager
CMD ["python", "enhanced_model_service.py"]