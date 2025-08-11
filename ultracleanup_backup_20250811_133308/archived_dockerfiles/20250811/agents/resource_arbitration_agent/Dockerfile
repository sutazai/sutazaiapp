FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8588
ENV AGENT_ID=resource-arbitration-agent
ENV AGENT_NAME="Resource Arbitration Agent"

# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8588/health || exit 1

# Expose port

# Expose port
EXPOSE 8588

CMD ["python", "app.py"]
