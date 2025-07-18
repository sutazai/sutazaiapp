# FAISS (Fast Similarity Search) Docker Container
FROM python:3.11-slim

LABEL maintainer="SutazAI Team"
LABEL description="FAISS vector similarity search service for SutazAI"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install FAISS and dependencies
RUN pip install --no-cache-dir \
    faiss-cpu \
    fastapi \
    uvicorn \
    numpy \
    pydantic \
    httpx \
    python-multipart \
    aiofiles

# Copy FAISS service code
COPY faiss_service.py /app/
COPY requirements.txt /app/

# Create data directories
RUN mkdir -p /data/faiss_indexes /logs

# Health check script
COPY health_check.py /app/

# Set permissions
RUN chmod +x /app/faiss_service.py

# Expose port
EXPOSE 8088

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python health_check.py || exit 1

# Start FAISS service
CMD ["python", "faiss_service.py"]