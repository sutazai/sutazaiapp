FROM python:3.11-slim

# Build arguments for user creation
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

# Install system dependencies and create non-root user early
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g ${GROUP_ID} appuser \
    && useradd -u ${USER_ID} -g appuser -m -s /bin/bash appuser

# Install Python dependencies as root (for efficiency)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy core messaging module
RUN mkdir -p /app/core
COPY messaging.py ./core/

# Copy application code
COPY app.py .
COPY metrics.py .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app \
    && chmod -R 777 /app/logs

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8589/health || exit 1

# Expose port
EXPOSE 8589

# Run the application
CMD ["python", "app.py"]