FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install common Python dependencies for agents
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    httpx==0.25.2 \
    redis==5.0.1 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.23 \
    prometheus-client==0.19.0 \
    python-consul==1.1.0 \
    pika==1.3.2 \
    asyncio \
    aiofiles \
    python-multipart

# Create directories
RUN mkdir -p /app/data /app/logs /workspaces

# Add health check script
COPY health_check.py /app/health_check.py

# Default environment
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Default command

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]