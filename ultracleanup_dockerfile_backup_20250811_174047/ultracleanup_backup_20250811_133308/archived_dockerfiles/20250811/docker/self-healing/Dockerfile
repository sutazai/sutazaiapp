FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8010
ENV AGENT_ID=self-healing
ENV AGENT_NAME="Self Healing"

# HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

# Expose API port

# Expose port
EXPOSE 8010

CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8010"]
