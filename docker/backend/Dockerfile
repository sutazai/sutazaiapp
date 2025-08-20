# SutazAI Backend Service - Consolidated Configuration
# Multi-stage build for optimized size and security
# Last Updated: 2025-08-20

FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    SERVICE_NAME=backend \
    SERVICE_PORT=10010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:10010/health || exit 1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 10010

# Run the application (production mode without --reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10010", "--workers", "2"]
