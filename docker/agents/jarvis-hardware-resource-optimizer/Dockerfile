FROM sutazai-python-agent-master:v1.0.0

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8080
ENV AGENT_ID=jarvis-hardware-resource-optimizer
ENV AGENT_NAME="Jarvis Hardware Resource Optimizer"

# HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1


# Security: Create non-root user

# Expose port
EXPOSE 8080


# Run as non-root user
USER appuser
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
