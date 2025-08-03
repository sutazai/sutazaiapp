# SECURITY HARDENED - Updated 2025-08-03
# Multi-stage build for optimal caching and faster builds
FROM python:3.12.8-slim AS base

WORKDIR /app

# Security: Create non-root user
RUN groupadd -r sutazai && useradd -r -g sutazai sutazai

# Install system dependencies with security patches
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Dependencies stage for better caching
FROM base AS dependencies

# Upgrade pip first with extended timeout
RUN python -m pip install --upgrade pip>=25.1.1 --timeout=120 --retries=5

# Copy requirements first for better caching
ARG REQUIREMENTS_FILE=requirements.txt
COPY ${REQUIREMENTS_FILE} requirements.txt
COPY requirements-minimal.txt .

# Install core dependencies first (lightweight packages)
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    fastapi>=0.115.6 \
    uvicorn[standard]>=0.32.1 \
    pydantic>=2.10.4 \
    pydantic-settings>=2.8.1 \
    sqlalchemy>=2.0.36 \
    alembic>=1.14.0 \
    psycopg2-binary>=2.9.10 \
    redis>=5.2.1

# Install ML dependencies in separate layer (heavy packages)
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    torch>=2.5.1 \
    transformers>=4.48.0 \
    sentence-transformers>=3.3.1 \
    scikit-learn>=1.6.0 \
    pandas>=2.2.3 \
    numpy>=2.1.3

# Install remaining dependencies (conditional based on build arg)
RUN if [ "$REQUIREMENTS_FILE" = "requirements-minimal.txt" ]; then \
        pip install --no-cache-dir --timeout=120 --retries=5 -r requirements-minimal.txt; \
    else \
        pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt || \
        pip install --no-cache-dir --timeout=300 --retries=3 -r requirements.txt; \
    fi

# Application stage
FROM dependencies AS application

# Copy application code
COPY . .

# Create necessary directories and set ownership
RUN mkdir -p /app/logs /app/uploads /app/data \
    && chown -R sutazai:sutazai /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER sutazai

# Production stage
FROM application AS production

# Expose port
EXPOSE 8000

# Health check with improved timing
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with optimized settings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]

# Development stage
FROM application AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest==8.3.4 \
    pytest-asyncio==0.23.5 \
    pytest-cov==4.1.0 \
    httpx==0.28.1 \
    black==24.10.0

# Enable reload for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Test stage
FROM base AS test

# Copy test requirements
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt || echo "No test requirements file"

# Install additional test tools
RUN pip install --no-cache-dir \
    pytest==8.0.0 \
    pytest-asyncio==0.23.0 \
    pytest-cov==4.1.0 \
    pytest-mock==3.12.0 \
    httpx==0.26.0

# Set test environment
ENV TESTING=1
ENV LOG_LEVEL=DEBUG

# Create test directories
RUN mkdir -p /app/test-results /app/htmlcov

# Default test command
CMD ["pytest", "-v", "--cov=app", "--cov-report=term-missing", "--cov-report=html:/app/htmlcov", "--cov-report=xml:/app/test-results/coverage.xml"]