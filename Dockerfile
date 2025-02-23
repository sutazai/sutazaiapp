FROM nvidia/cuda:12.1.1-base-ubuntu22.04 as builder

# System optimizations
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    NCCL_VERSION=2.18.1-1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    build-essential \
    cmake \
    ninja-build \
    git \
    libopenblas-dev \
    libomp-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 8000 8501

# Set environment variables
ENV OMP_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN useradd -m -s /bin/bash sutazai && \
    chown -R sutazai:sutazai /app

# Switch to non-root user
USER sutazai

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/bin/bash", "./deploy_all.sh"]