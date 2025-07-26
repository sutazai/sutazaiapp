# SECURITY HARDENED - Updated 2025-07-26
FROM python:3.13.5-slim

WORKDIR /app

# Security: Create non-root user
RUN groupadd -r sutazai && useradd -r -g sutazai sutazai

# Install system dependencies with security patches
RUN apt-get update && apt-get install -y \
    gcc=4:12.2.0-3 \
    g++=4:12.2.0-3 \
    curl=7.88.1-10+deb12u8 \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements
COPY requirements.txt .
# Security: Upgrade pip and install with verification
RUN python -m pip install --upgrade pip>=25.1.1 \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/.streamlit

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=false", "--browser.gatherUsageStats=false"]