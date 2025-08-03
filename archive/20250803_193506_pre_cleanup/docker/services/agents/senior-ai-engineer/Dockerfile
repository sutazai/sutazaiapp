# Senior AI Engineer Agent - extends agent base
FROM sutazai/agent-base:latest

# Switch to root for package installation
USER root

# Install additional requirements for AI engineering
RUN pip install --no-cache-dir \
    jupyter>=1.0.0 \
    notebook>=7.0.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    plotly>=5.17.0 \
    pandas>=2.1.0 \
    scipy>=1.11.0

# Create specialized directories
RUN mkdir -p /app/notebooks /app/models /app/experiments && \
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
ENV AGENT_NAME=senior-ai-engineer \
    AGENT_TYPE=senior-ai-engineer \
    AGENT_CAPABILITIES="code_generation,architecture_design,ml_engineering,system_optimization" \
    OLLAMA_URL=http://sutazai-ollama:11434 \
    BACKEND_URL=http://sutazai-backend:8000

# Health check with agent-specific endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the agent
CMD ["python", "app.py"]