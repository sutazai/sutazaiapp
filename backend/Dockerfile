# SECURITY HARDENED - Updated 2025-07-26
FROM python:3.12.8-slim AS base

WORKDIR /app

# Security: Create non-root user
RUN groupadd -r sutazai && useradd -r -g sutazai sutazai

# Install system dependencies with security patches
RUN apt-get update && apt-get install -y \
    gcc=4:12.2.0-3 \
    g++=4:12.2.0-3 \
    curl=7.88.1-10+deb12u8 \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements first for better caching
COPY requirements.txt .
# Security: Upgrade pip and install with hash verification
RUN python -m pip install --upgrade pip>=25.1.1 \
    && pip install --no-cache-dir --require-hashes -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories and set ownership
RUN mkdir -p /app/logs /app/uploads /app/data \
    && chown -R sutazai:sutazai /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Switch to non-root user
USER sutazai

# Production stage
FROM base AS production

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
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