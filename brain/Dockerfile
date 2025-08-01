FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/brain

# Copy minimal requirements first for better caching
COPY requirements_minimal.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Copy brain code
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/brain/checkpoints \
    /workspace/brain/logs \
    /workspace/brain/models \
    /workspace/brain/data \
    /workspace/brain/experiences

# Set environment variables
ENV PYTHONPATH=/workspace/brain:$PYTHONPATH
ENV BRAIN_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Expose brain API port
EXPOSE 8888

# Run brain API service
CMD ["python", "-u", "brain_api.py"]