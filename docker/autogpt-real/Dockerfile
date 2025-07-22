FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    aiohttp \
    openai \
    python-multipart

# Create a simple AutoGPT-like agent
COPY agent.py /app/agent.py

# Health check endpoint
EXPOSE 8080

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8080"]