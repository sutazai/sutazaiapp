# Hardware Resource Optimizer Agent - On-demand optimization tool
FROM sutazai-python-agent-master:v1.0.0

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy agent files
COPY app.py .
COPY shared/ ./shared/

# Override base environment variables for this specific service
ENV SERVICE_PORT=8080
ENV AGENT_ID=hardware-resource-optimizer
ENV AGENT_NAME="Hardware Resource Optimizer"
ENV AGENT_TYPE=hardware-resource-optimizer

# Override health check for this specific service pattern
HEALTHCHECK --interval=60s --timeout=15s --start-period=10s --retries=2 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Switch to non-root user for security
USER appuser

CMD ["python", "app.py"]