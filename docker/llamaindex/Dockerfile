FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install LlamaIndex and dependencies
RUN pip install --no-cache-dir \
    llama-index \
    llama-index-llms-ollama \
    llama-index-embeddings-ollama \
    fastapi \
    uvicorn \
    requests

# Create LlamaIndex service
COPY llamaindex_service.py .

EXPOSE 8080

CMD ["uvicorn", "llamaindex_service:app", "--host", "0.0.0.0", "--port", "8080"]
EOF < /dev/null
