# Dockerfile for MCP Monitoring Server

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy monitoring code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 monitoring && \
    chown -R monitoring:monitoring /app

# Switch to non-root user
USER monitoring

# Expose port
EXPOSE 10204

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10204/health || exit 1

# Run monitoring server
CMD ["python", "monitoring_server.py"]