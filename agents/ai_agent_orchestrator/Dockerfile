FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy core messaging module first
RUN mkdir -p /app/core
COPY messaging.py ./core/

# Copy application code
COPY app.py .
COPY metrics.py .

# Create logs directory
RUN mkdir -p /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8589/health || exit 1

# Expose port
EXPOSE 8589

# Run the application
CMD ["python", "app.py"]