# Awesome-Code-AI Integration Container
FROM python:3.11-slim

LABEL maintainer="SutazAI Team"
LABEL description="Awesome Code AI collection integration for SutazAI"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone Awesome Code AI repository
RUN git clone https://github.com/sourcegraph/awesome-code-ai.git /app/awesome-code-ai

# Install Python dependencies (GPU/CPU adaptive)
RUN if [ "${PYTORCH_CPU_ONLY:-false}" = "true" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch; \
    fi && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    aiohttp \
    pydantic \
    python-multipart \
    transformers \
    openai \
    langchain \
    GitPython

# Copy integration service
COPY awesome_code_service.py /app/
COPY code_ai_manager.py /app/
COPY requirements.txt /app/

# Create necessary directories
RUN mkdir -p /data/code_analysis /data/models /logs

# Set permissions
RUN chmod +x /app/awesome_code_service.py

# Expose port
EXPOSE 8089

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8089/health || exit 1

# Start service
CMD ["python", "awesome_code_service.py"]