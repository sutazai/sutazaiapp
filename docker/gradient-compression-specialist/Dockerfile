FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Set environment variables
ENV AGENT_ID=gradient-compression-specialist
ENV AGENT_NAME="Gradient Compression Specialist"
ENV PORT=8576

# Expose port
EXPOSE 8576

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8576/health || exit 1

# Run the application

# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["python", "app.py"]
