FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Clone and install GPT Engineer
RUN git clone https://github.com/AntonOsika/gpt-engineer.git . \
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
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "gpt_engineer.web.main:app", "--host", "0.0.0.0", "--port", "8080"]