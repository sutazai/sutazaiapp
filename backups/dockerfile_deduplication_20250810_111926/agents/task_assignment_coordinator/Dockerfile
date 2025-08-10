FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY task_assignment_coordinator/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY task_assignment_coordinator /app

# Copy core module
COPY core /app/core

# Create directories and set up Python path
RUN mkdir -p /app/logs /app/data
ENV PYTHONPATH=/app:/app/agents:/opt/sutazaiapp/agents

# Create non-root user with proper permissions
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8551/health || exit 1

# Expose port
EXPOSE 8551

# Start application
CMD ["python", "app.py"]