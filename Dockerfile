FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    build-essential \
    libpq-dev \
    redis-tools \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements_frozen.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_frozen.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs cache models/ollama temp run backup

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 3000 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "scripts/deploy.py"]
