FROM sutazai-python-agent-master:latest

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=9200
ENV AGENT_ID=ai-metrics-exporter
ENV AGENT_NAME="Ai Metrics Exporter"

# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:9200/health || exit 1


# Expose port
EXPOSE 9200

CMD ["python", "ai_metrics_exporter.py"]
