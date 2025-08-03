FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - lightweight set
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn==0.32.1 \
    psycopg2-binary==2.9.10 \
    redis==5.2.0 \
    requests==2.32.3 \
    pydantic==2.10.3 \
    numpy==1.26.4

# Create directories
RUN mkdir -p /app/logs /app/memory

# Copy application code
COPY backend/ /app/backend/
COPY brain/ /app/brain/
COPY config/ /app/config/

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check script
RUN echo 'import requests; import sys; sys.exit(0 if requests.get("http://localhost:9200/health", timeout=5).status_code == 200 else 1)' > /app/healthcheck.py

# Expose ports
EXPOSE 8900 9200

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python /app/healthcheck.py

# Default command

# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["python", "/app/brain/simple_agi_brain.py"]