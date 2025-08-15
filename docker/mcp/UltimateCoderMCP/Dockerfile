# Base image: lightweight but with apt tools for ripgrep
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies (ripgrep for fast code search)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ripgrep && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Expose MCP server over STDIO
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health').read()" || exit 1
# Health check for service availability


# Security: Create non-root user
RUN addgroup -g 1001 appgroup && \
    adduser -D -u 1001 -G appgroup -s /bin/sh appuser

# Change ownership of application files
RUN chown -R appuser:appgroup /app || true

# Switch to non-root user
USER appuser

CMD ["python", "main.py"]
