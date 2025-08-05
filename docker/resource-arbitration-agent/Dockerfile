FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Set environment variables
ENV AGENT_ID=resource-arbitration-agent
ENV AGENT_NAME="Resource Arbitration Agent"
ENV PORT=8588

# Expose port
EXPOSE 8588

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8588/health || exit 1

# Run as non-root user
USER nobody

CMD ["python", "app.py"]