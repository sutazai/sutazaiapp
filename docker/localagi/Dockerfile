FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Clone LocalAGI
RUN git clone https://github.com/mudler/LocalAGI.git . || mkdir -p /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    websockets \
    pydantic \
    chromadb \
    langchain \
    langchain-community

# Create LocalAGI service
COPY # localagi removed_service.py /app/main.py

EXPOSE 8080


# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]