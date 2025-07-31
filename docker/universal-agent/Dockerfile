FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent runtime
COPY agent_runtime.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Environment variables (can be overridden)
ENV AGENT_NAME=""
ENV AGENT_CAPABILITIES=""
ENV MODEL_PROVIDER="ollama"
ENV MODEL_NAME="llama2:latest"
ENV MODEL_TEMPERATURE="0.7"
ENV MODEL_MAX_TOKENS="4096"
ENV SYSTEM_PROMPT=""
ENV OLLAMA_HOST="http://ollama:11434"
ENV LITELLM_HOST="http://litellm:4000"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run the agent
ENTRYPOINT ["./entrypoint.sh"]