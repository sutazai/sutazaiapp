# Multi-stage build for optimal frontend deployment
FROM python:3.12.8-slim AS base

WORKDIR /app

# Security: Create non-root user
RUN groupadd -r streamlit && useradd -r -g streamlit streamlit

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Dependencies stage
FROM base AS dependencies

# Upgrade pip with extended timeout
RUN python -m pip install --upgrade pip --timeout=120 --retries=5

# Install core packages first
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    streamlit>=1.40.2 \
    httpx>=0.28.1 \
    requests>=2.32.3 \
    pandas>=2.2.3 \
    plotly>=5.24.1

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    -r requirements.txt || pip install --no-cache-dir --timeout=300 --retries=3 -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories and set ownership
RUN mkdir -p /app/data /app/.streamlit \
    && chown -R streamlit:streamlit /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER streamlit

# Application stage
FROM dependencies AS application

# Production stage
FROM application AS production

# Expose port
EXPOSE 8501

# Health check with improved timing
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Run the application with optimized settings
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=false", "--browser.gatherUsageStats=false", "--server.maxUploadSize=200"]

# Development stage  
FROM application AS development

# Enable auto-reload for development
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true", "--browser.gatherUsageStats=false"]