FROM sutazai-python-agent-master:v1.0.0

# Copy agent-specific requirements and install additional dependencies
COPY requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/agent-requirements.txt && rm /tmp/agent-requirements.txt

# Copy application code
COPY . .

# Override base environment variables for this specific service
ENV SERVICE_PORT=8090
ENV AGENT_ID=jarvis-voice-interface
ENV AGENT_NAME="Jarvis Voice Interface"

# HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1


# Expose port
EXPOSE 8090

CMD ["python", "app.py"]
