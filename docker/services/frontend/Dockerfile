FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8501
ENV AGENT_ID=frontend
ENV AGENT_NAME="Frontend"

# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application

# Run as non-root user

# Expose port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
