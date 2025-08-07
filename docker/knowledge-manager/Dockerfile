FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000


# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["uvicorn", "knowledge_manager:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]