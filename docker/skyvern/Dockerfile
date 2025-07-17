FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Clone and install Skyvern
RUN git clone https://github.com/Skyvern-AI/skyvern.git . \
    && pip install -e .

# Create necessary directories
RUN mkdir -p /workspace /logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "skyvern.forge.app:app", "--host", "0.0.0.0", "--port", "8080"]