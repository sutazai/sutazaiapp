FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

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