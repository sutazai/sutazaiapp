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
ENV AGENT_ID=transformers-migration-specialist
ENV AGENT_NAME="Transformers Migration Specialist"
ENV PORT=8558

# Expose port
EXPOSE 8558

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8558/health || exit 1

# Run the application
CMD ["python", "app.py"]
