# SutazAI Senior AI Engineer Service - ULTRAFIX Dockerfile Consolidation  
# Migrated to use sutazai-python-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
FROM sutazai-python-agent-master:latest

# Copy application files
COPY . .

# Install service-specific requirements if they exist
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi || echo "No requirements.txt found, using base image dependencies"

# Make startup script executable if it exists
RUN chmod +x startup.sh 2>/dev/null || echo "No startup.sh found"

# Override base environment variables for this service
ENV SERVICE_PORT=8080
ENV AGENT_ID=sutazai-senior-ai-engineer
ENV AGENT_NAME="SutazAI Senior AI Engineer Service"

# Override health check for this service
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose service port
EXPOSE 8080

# Switch to non-root user (inherited from base)
USER appuser

# Use service-specific entry point
CMD ["python", "agent.py"]