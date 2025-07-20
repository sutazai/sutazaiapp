# SutazAI Frontend Dockerfile
# ===========================
# Optimized Streamlit application

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r sutazai && useradd -r -g sutazai sutazai

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY core/frontend/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Copy application code
COPY core/frontend/ /app/

# Create necessary directories
RUN mkdir -p /app/uploads /app/cache && \
    chown -R sutazai:sutazai /app

# Switch to non-root user
USER sutazai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Start application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]