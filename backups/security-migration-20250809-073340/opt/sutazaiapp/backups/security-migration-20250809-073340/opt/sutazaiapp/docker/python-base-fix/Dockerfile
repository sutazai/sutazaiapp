# Base Python image with health check and common dependencies
FROM python:3.11-slim

# Install health check tools and common dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    netcat-openbsd \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Add health check endpoint
RUN pip install --no-cache-dir flask aiohttp

# Add simple health check server
RUN echo 'from flask import Flask\n\
app = Flask(__name__)\n\
@app.route("/health")\n\
def health():\n\
    return {"status": "healthy"}, 200\n\
if __name__ == "__main__":\n\
    app.run(host="0.0.0.0", port=8080)' > /tmp/health_server.py

# Default health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

USER appuser
