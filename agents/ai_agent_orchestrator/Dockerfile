FROM sutazai-python-agent-master:v1.0.0

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy core messaging module
COPY messaging.py ./core/
COPY app.py .
COPY metrics.py .

# Create agent-specific directories (base directories already exist and are owned by appuser)
RUN mkdir -p /app/logs && chmod -R 777 /app/logs

# Override base environment variables for this specific service
ENV SERVICE_PORT=8589
ENV AGENT_ID=ai-agent-orchestrator
ENV AGENT_NAME="AI Agent Orchestrator"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8589/health || exit 1

# Expose port
EXPOSE 8589

# Run the application
CMD ["python", "app.py"]