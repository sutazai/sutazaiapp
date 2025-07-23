FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Aider from PyPI
RUN pip install --no-cache-dir aider-chat

# Copy service files
COPY aider_service.py .
COPY requirements.txt .

# Install service dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create workspace
RUN mkdir -p /app/workspace

EXPOSE 8080

CMD ["uvicorn", "aider_service:app", "--host", "0.0.0.0", "--port", "8080"]
