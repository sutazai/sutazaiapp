FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8080
ENV AGENT_ID=service-account-manager
ENV AGENT_NAME="Service Account Manager"

# HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1


# Security: Create non-root user

# Expose port
EXPOSE 8080

CMD ["python", "-u", "app.py"]
