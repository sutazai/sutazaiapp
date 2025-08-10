FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8576
ENV AGENT_ID=gradient-compression-specialist
ENV AGENT_NAME="Gradient Compression Specialist"

# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8576/health || exit 1

# Run the application

# Run as non-root user

# Expose port
EXPOSE 8576

CMD ["python", "app.py"]
