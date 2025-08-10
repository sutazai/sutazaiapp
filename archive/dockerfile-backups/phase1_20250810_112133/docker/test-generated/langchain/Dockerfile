# SutazAI Python Agent Service Template
# This template generates service-specific Dockerfiles
# Author: DevOps Manager - Deduplication Operation
# Date: August 10, 2025
#
# Usage: Replace langchain, 8503, LangChain agent framework service
# Example: sed 's/langchain/hardware-optimizer/g' template > service/Dockerfile

FROM sutazai-python-agent-master:latest

# Service-specific metadata
LABEL service.name="langchain" \
      service.port="8503" \
      service.description="LangChain agent framework service" \
      service.type="python-agent" \
      build.date="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
      build.version="v1.0.0"

# Copy service-specific requirements if they exist
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy service source code
COPY . /app/

# Set service-specific environment variables
ENV SERVICE_NAME=langchain \
    SERVICE_PORT=8503 \
    OLLAMA_BASE_URL=http://ollama:11434

# Expose service port
EXPOSE 8503

# Override health check for service-specific endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8503/health || exit 1

# Service-specific startup command
CMD ["python", "-m", "langchain"]