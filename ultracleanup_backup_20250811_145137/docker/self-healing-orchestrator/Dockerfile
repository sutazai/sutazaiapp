FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8577
ENV AGENT_ID=self-healing-orchestrator
ENV AGENT_NAME="Self Healing Orchestrator"

# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8577/health || exit 1

# Run the application

# Run as non-root user

# Expose port
EXPOSE 8577

CMD ["python", "app.py"]
