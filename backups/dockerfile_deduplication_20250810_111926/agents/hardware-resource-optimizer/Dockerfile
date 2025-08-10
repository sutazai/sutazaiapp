# Hardware Resource Optimizer Agent - On-demand optimization tool
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for hardware optimization
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    net-tools \
    findutils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent files
COPY app.py .
COPY shared/ ./shared/

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080
ENV AGENT_TYPE=hardware-resource-optimizer
ENV LOG_LEVEL=INFO

# Health check for on-demand service
HEALTHCHECK --interval=60s --timeout=15s --start-period=10s --retries=2 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start the on-demand optimization agent

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["python", "app.py"]