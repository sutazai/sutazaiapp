FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    docker.io \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy health check script
COPY health_check.py .

# Install Python dependencies
RUN pip install requests docker

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run health check

# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["python", "health_check.py"]