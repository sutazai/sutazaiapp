FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install AutoGen with all features
RUN pip install --no-cache-dir \
    'ag2[all]' \
    fastapi \
    uvicorn \
    httpx \
    redis

# Copy service implementation
COPY autogen_service.py /app/

EXPOSE 8080

CMD ["uvicorn", "autogen_service:app", "--host", "0.0.0.0", "--port", "8080"]