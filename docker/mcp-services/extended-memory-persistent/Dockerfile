# Extended Memory MCP Service with SQLite Persistence
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /var/lib/mcp /var/log/mcp /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.5 \
    uvicorn[standard]==0.32.1 \
    python-multipart==0.0.17 \
    aiosqlite==0.20.0 \
    pydantic==2.10.3

# Copy server code
COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Create non-root user for security
RUN useradd -m -u 1000 mcp && \
    chown -R mcp:mcp /app /var/lib/mcp /var/log/mcp

# Switch to non-root user
USER mcp

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVICE_PORT=3009 \
    SQLITE_PATH=/var/lib/mcp/extended_memory.db \
    ENABLE_CACHE=true

# Expose port
EXPOSE 3009

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3009/health || exit 1

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]