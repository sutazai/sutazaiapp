FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8120
ENV AGENT_ID=jarvis-ai
ENV AGENT_NAME="Jarvis Ai"

# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8120/health || exit 1

# Start JARVIS service

# Run as non-root user

# Expose port
EXPOSE 8120

CMD ["python", "jarvis_service.py"]
