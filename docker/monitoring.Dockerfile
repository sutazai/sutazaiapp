FROM python:3.12-alpine

# Install system dependencies
RUN apk add --no-cache curl docker-cli procps

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY docker/monitoring-requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy monitoring scripts
COPY monitoring/ ./monitoring/
COPY scripts/system_monitor.py ./
COPY scripts/health-monitor.py ./

# Create non-root user
RUN adduser -D -s /bin/sh monitor
USER monitor

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python health-monitor.py --check

# Default command
CMD ["python", "system_monitor.py"]