FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8570
ENV AGENT_ID=knowledge-distillation-expert
ENV AGENT_NAME="Knowledge Distillation Expert"

# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8570/health || exit 1

# Run the application

# Run as non-root user

# Expose port
EXPOSE 8570

CMD ["python", "app.py"]
