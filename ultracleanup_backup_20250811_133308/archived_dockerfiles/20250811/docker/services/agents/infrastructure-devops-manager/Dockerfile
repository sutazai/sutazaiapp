# Migrated to Master Base Image
# Original: sutazai/agent-base:latest
# Migration Date: 2025-08-10T19:58:30.712235
FROM sutazai-python-agent-master:latest

COPY COPY --chown=agent:agent requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Service-specific setup
RUN pip install --no-cache-dir \
RUN mkdir -p /app/monitoring /app/terraform /app/configs && \
RUN pip install --no-cache-dir -r /app/requirements.txt 2>/dev/null || true

# Copy application code
COPY --chown=agent:agent config.json /app/config/
COPY --chown=agent:agent *.py /app/
COPY . .

# Environment configuration
ENV AGENT_NAME=infrastructure-devops-manager \
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \

# Security: Run as non-root user
USER appuser

CMD ["python", "app.py"]