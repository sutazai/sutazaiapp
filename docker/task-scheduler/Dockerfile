FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    cron \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    sqlalchemy \
    psycopg2-binary \
    redis \
    aioredis \
    celery \
    schedule \
    asyncio \
    requests \
    aiohttp \
    python-multipart \
    python-dotenv \
    structlog \
    prometheus-client

# Copy scheduler service files
COPY task_scheduler.py requirements.txt* ./
COPY scheduler_config.yaml ./

# Install additional requirements if they exist
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Create directories
RUN mkdir -p /app/data /app/logs /app/tasks

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SCHEDULER_MODE=autonomous
ENV MAX_CONCURRENT_TASKS=10

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Start task scheduler

# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["python", "task_scheduler.py"] 