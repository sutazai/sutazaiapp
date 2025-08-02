FROM python:3.11-slim

WORKDIR /app

# Security: Create non-root user
RUN groupadd -r streamlit && useradd -r -g streamlit streamlit

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install httpx first
RUN pip install httpx==0.28.1

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy application code
COPY . .

# Create necessary directories and set ownership
RUN mkdir -p /app/data /app/.streamlit \
    && chown -R streamlit:streamlit /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Switch to non-root user
USER streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=false", "--browser.gatherUsageStats=false"]