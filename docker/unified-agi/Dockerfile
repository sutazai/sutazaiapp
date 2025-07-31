FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for unified AGI
RUN pip install --no-cache-dir \
    langchain==0.2.16 \
    langchain-community==0.2.17 \
    langchain-openai==0.1.25 \
    chromadb==0.5.3 \
    qdrant-client==1.11.3 \
    sentence-transformers==3.1.1 \
    numpy==1.26.4 \
    scikit-learn==1.5.2 \
    fastapi==0.115.6 \
    uvicorn==0.32.1 \
    websockets==13.1 \
    asyncio-mqtt==0.16.2 \
    aioredis==2.0.1 \
    psycopg2-binary==2.9.10

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/memory /app/knowledge

# Copy application code
COPY backend/ /app/backend/
COPY brain/ /app/brain/
COPY config/ /app/config/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create health check script
RUN echo '#!/usr/bin/env python3\nimport requests\nimport sys\ntry:\n    response = requests.get("http://localhost:9200/health", timeout=5)\n    sys.exit(0 if response.status_code == 200 else 1)\nexcept:\n    sys.exit(1)' > /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# Expose ports
EXPOSE 8900 8901 8902 9200

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

# Default command
CMD ["python", "/app/brain/unified_agi_system.py"]