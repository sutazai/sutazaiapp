# SutazAI Python Agent Service Template
# This template generates service-specific Dockerfiles
# Author: DevOps Manager - Deduplication Operation
# Date: August 10, 2025
#
# Usage: Replace autogpt, 8501, AutoGPT autonomous agent service
# Example: sed 's/autogpt/hardware-optimizer/g' template > service/Dockerfile

FROM sutazai-python-agent-master:latest

# Service-specific metadata
LABEL service.name="autogpt" \
      service.port="8501" \
      service.description="AutoGPT autonomous agent service" \
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
ENV SERVICE_NAME=autogpt \
    SERVICE_PORT=8501 \
    OLLAMA_BASE_URL=http://ollama:11434

# Expose service port
EXPOSE 8501

# Override health check for service-specific endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Service-specific startup command
CMD ["python", "-m", "autogpt"]