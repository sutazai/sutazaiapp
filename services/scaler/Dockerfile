FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    docker \
    python-consul \
    redis \
    prometheus-client \
    requests \
    asyncio

# Copy application code
COPY service_scaler.py /app/

# Make executable
RUN chmod +x /app/service_scaler.py

# Run the scaler
CMD ["python", "/app/service_scaler.py"]