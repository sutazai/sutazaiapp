FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    docker.io \
    systemd \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Fix Docker socket permissions issue by running as root
# This is necessary for containers that need Docker API access
RUN chmod +x /app/app.py

# Health check endpoint
EXPOSE 8522

# Run as root to access Docker socket
USER root


# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["python", "app.py"]