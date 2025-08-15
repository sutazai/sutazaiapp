FROM sutazai-python-agent-master:v1.0.0

# Copy agent-specific requirements and install additional dependencies
# Note: Base image already has appuser:appgroup configured
COPY --chown=appuser:appgroup requirements.txt /tmp/agent-requirements.txt

# Switch to non-root user for package installation
USER appuser
RUN pip install --user --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code with proper ownership
COPY --chown=appuser:appgroup . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8080
ENV AGENT_ID=ai-agent-orchestrator
ENV AGENT_NAME="AI Agent Orchestrator"
ENV DEBUG=false

# Comprehensive health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health').read()" || exit 1

# Expose port
EXPOSE 8080

# Ensure we're running as non-root user (redundant but explicit)
USER appuser

# Run the application
CMD ["python", "app.py"]
