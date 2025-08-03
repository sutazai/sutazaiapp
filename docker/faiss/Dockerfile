# Multi-stage build for FAISS service optimization
FROM python:3.12.8-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Dependencies stage
FROM base AS dependencies

# Upgrade pip with timeout
RUN python -m pip install --upgrade pip --timeout=120 --retries=5

# Install core dependencies with optimized order
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    numpy>=2.1.3 \
    scipy>=1.14.1

# Install FAISS CPU (heavy package)
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    faiss-cpu>=1.9.0

# Install web framework
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    flask>=3.1.0 \
    gunicorn>=21.2.0

# Application stage
FROM dependencies AS application

# Copy service files
COPY faiss_service.py .
COPY health_check.py .

# Create non-root user
RUN groupadd -r faiss && useradd -r -g faiss faiss \
    && chown -R faiss:faiss /app

USER faiss

# Production stage
FROM application AS production

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD python health_check.py || exit 1

EXPOSE 8000

# Run with Gunicorn for better performance
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "faiss_service:app"]

# Development stage
FROM application AS development

CMD ["python", "faiss_service.py"]
