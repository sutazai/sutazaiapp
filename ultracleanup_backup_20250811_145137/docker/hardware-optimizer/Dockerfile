FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8523
ENV AGENT_ID=hardware-optimizer
ENV AGENT_NAME="Hardware Optimizer"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8523/health || curl -f http://localhost:8523/ || exit 1

# Expose port
EXPOSE 8523

CMD ["python", "app.py"]
