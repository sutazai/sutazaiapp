FROM node:18-slim as frontend-builder

# Install frontend dependencies
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci

# Build frontend
COPY web/ ./
RUN npm run build

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Clone Dify repository
RUN git clone https://github.com/langgenius/dify.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend build
COPY --from=frontend-builder /app/web/dist ./web/dist

# Create necessary directories
RUN mkdir -p /data /logs /storage

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV EDITION=COMMUNITY
ENV DEPLOY_ENV=PRODUCTION

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run Dify
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "200", "--preload", "app:app"]