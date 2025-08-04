# Hardware Resource Optimizer Agent Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent files
COPY app.py .
COPY startup.sh .
COPY shared/ ./shared/

# Make startup script executable
RUN chmod +x startup.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080
ENV AGENT_TYPE=hardware-resource-optimizer
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start the agent
CMD ["python", "app.py"]