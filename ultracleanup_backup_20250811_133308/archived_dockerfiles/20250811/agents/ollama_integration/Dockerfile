FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements/base.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Create agent-specific directories (base directories already exist)
RUN mkdir -p /app/schemas /app/agents/core

# Copy application code
COPY app.py .
COPY ollama_schemas.py ./schemas/
COPY metrics.py ./agents/core/

# Override base environment variables for this specific service
ENV SERVICE_PORT=8090
ENV AGENT_ID=ollama-integration
ENV AGENT_NAME="Ollama Integration Service"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8090/health || exit 1

# Expose port
EXPOSE 8090

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
