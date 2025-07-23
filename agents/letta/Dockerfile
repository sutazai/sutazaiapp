FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Letta (MemGPT)
RUN pip install letta[local] uvicorn fastapi

# Copy application files
COPY . /app/

# Create required directories
RUN mkdir -p /workspace /data

# Set environment variables
ENV WORKSPACE_PATH=/workspace
ENV DATA_PATH=/data
ENV PYTHONPATH=/app

# Health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8080/health || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Expose port
EXPOSE 8080

# Start the service
CMD ["python", "web_interface.py"]