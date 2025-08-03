FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    aiohttp \
    aioredis \
    pydantic \
    pydantic-settings \
    python-multipart \
    python-jose[cryptography] \
    passlib[bcrypt] \
    httpx \
    asyncio \
    prometheus-client \
    structlog \
    python-dotenv \
    limits \
    slowapi \
    starlette-prometheus

# Copy gateway service files
COPY api_gateway.py requirements.txt* ./
COPY gateway_config.yaml nginx.conf ./

# Install additional requirements if they exist
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Create directories
RUN mkdir -p /app/logs /app/cache /app/config

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8080
ENV REDIS_URL=redis://redis:6379/0

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start API Gateway
CMD ["python", "api_gateway.py"] 