FROM sutazai-python-agent-master:v1.0.0

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8000
ENV AGENT_ID=faiss
ENV AGENT_NAME="Faiss"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:8000/ || exit 1

# Expose port
EXPOSE 8000


# Run as non-root user
USER appuser
CMD ["python", "faiss_service.py"]
