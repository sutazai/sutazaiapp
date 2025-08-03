FROM python:3.11-slim

LABEL maintainer="Sutazai Infrastructure Team"
LABEL description="Hygiene Monitoring Backend API"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r hygiene && useradd -r -g hygiene hygiene

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY docker/hygiene-backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY monitoring/enhanced-hygiene-backend.py ./app.py
COPY docker/hygiene-backend/entrypoint.sh ./entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R hygiene:hygiene /app

# Make entrypoint executable
RUN chmod +x ./entrypoint.sh

# Switch to non-root user
USER hygiene

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/hygiene/status || exit 1

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]