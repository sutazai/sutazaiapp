FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Letta (formerly MemGPT)
RUN pip install --no-cache-dir \
    letta \
    fastapi \
    uvicorn \
    psycopg2-binary \
    requests

# Copy service files
COPY letta_service.py .
COPY requirements.txt .

# Install additional requirements if any
RUN pip install --no-cache-dir -r requirements.txt || true

# Create workspace
RUN mkdir -p /app/workspace

EXPOSE 8080

CMD ["uvicorn", "letta_service:app", "--host", "0.0.0.0", "--port", "8080"]