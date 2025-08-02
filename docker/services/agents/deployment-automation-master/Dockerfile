# Deployment Automation Master Agent - extends agent base
FROM sutazai/agent-base:latest

# Switch to root for package installation
USER root

# Install Docker CLI and deployment tools
RUN apt-get update && apt-get install -y \
    docker.io \
    git \
    ssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install deployment-specific Python packages
RUN pip install --no-cache-dir \
    docker>=7.0.0 \
    kubernetes>=29.0.0 \
    ansible>=8.0.0 \
    terraform-external-data>=2.0.0 \
    paramiko>=3.4.0 \
    fabric>=3.2.0

# Create deployment directories
RUN mkdir -p /app/playbooks /app/scripts /app/deployments && \
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
ENV AGENT_NAME=deployment-automation-master \
    AGENT_TYPE=deployment-automation-master \
    AGENT_CAPABILITIES="docker_management,k8s_deployment,ci_cd_automation,infrastructure_provisioning" \
    OLLAMA_URL=http://sutazai-ollama:11434 \
    BACKEND_URL=http://sutazai-backend:8000 \
    DOCKER_HOST=unix:///var/run/docker.sock

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the agent
CMD ["python", "app.py"]