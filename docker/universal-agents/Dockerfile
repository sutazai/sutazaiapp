FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for the universal agent system
RUN pip install --no-cache-dir \
    aioredis \
    httpx \
    pydantic \
    asyncio-mqtt \
    python-multipart \
    fastapi \
    uvicorn

# Copy application code
COPY backend/ /app/backend/
COPY scripts/ /app/scripts/
COPY config/ /app/config/

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check script
COPY docker/universal-agents/healthcheck.py /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# Expose ports
EXPOSE 8910 9100 9101

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

# Default command
CMD ["python", "/app/scripts/start_universal_agent_system.py"]