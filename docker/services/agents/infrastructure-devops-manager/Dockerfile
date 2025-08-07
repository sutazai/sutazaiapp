# Infrastructure DevOps Manager Agent - extends agent base
FROM sutazai/agent-base:latest

# Switch to root for package installation
USER root

# Install infrastructure and monitoring tools
RUN apt-get update && apt-get install -y \
    docker.io \
    git \
    curl \
    wget \
    jq \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Install infrastructure-specific Python packages
RUN pip install --no-cache-dir \
    docker>=7.0.0 \
    prometheus-client>=0.20.0 \
    grafana-api>=1.0.3 \
    psutil>=5.9.8 \
    boto3>=1.34.0 \
    azure-mgmt-resource>=23.0.0 \
    google-cloud-monitoring>=2.21.0

# Create infrastructure directories
RUN mkdir -p /app/monitoring /app/terraform /app/configs && \
    chown -R agent:agent /app

# Copy agent-specific files
COPY --chown=agent:agent config.json /app/config/
COPY --chown=agent:agent requirements.txt /app/
COPY --chown=agent:agent *.py /app/

# Install any additional requirements
RUN pip install --no-cache-dir -r /app/requirements.txt 2>/dev/null || true

# Switch back to non-root user
USER agent

# Agent-specific environment variables
ENV AGENT_NAME=infrastructure-devops-manager \
    AGENT_TYPE=infrastructure-devops-manager \
    AGENT_CAPABILITIES="infrastructure_monitoring,resource_optimization,scaling_automation,performance_analysis" \
    OLLAMA_URL=http://sutazai-ollama:11434 \
    BACKEND_URL=http://sutazai-backend:8000 \
    DOCKER_HOST=unix:///var/run/docker.sock

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the agent

# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["python", "app.py"]