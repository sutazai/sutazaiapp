# SutazAI Python Agent Service Template
# This template generates service-specific Dockerfiles
# Author: DevOps Manager - Deduplication Operation
# Date: August 10, 2025
#
# Usage: Replace crewai, 8502, CrewAI multi-agent collaboration system
# Example: sed 's/crewai/hardware-optimizer/g' template > service/Dockerfile

FROM sutazai-python-agent-master:latest

# Service-specific metadata
LABEL service.name="crewai" \
      service.port="8502" \
      service.description="CrewAI multi-agent collaboration system" \
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
ENV SERVICE_NAME=crewai \
    SERVICE_PORT=8502 \
    OLLAMA_BASE_URL=http://ollama:11434

# Expose service port
EXPOSE 8502

# Override health check for service-specific endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8502/health || exit 1

# Service-specific startup command
CMD ["python", "-m", "crewai"]