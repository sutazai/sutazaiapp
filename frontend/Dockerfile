# JARVIS Frontend Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=11000 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_THEME_BASE=dark \
    STREAMLIT_THEME_PRIMARY_COLOR="#00D4FF"

# Use a simpler approach - install only essential packages
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        portaudio19-dev \
        python3-pyaudio \
        gcc \
        g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (including audio libraries)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/temp /app/logs /app/data

# Environment variables are already set above

# Expose Streamlit port
EXPOSE 11000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:11000/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=11000", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.serverAddress=localhost", \
     "--browser.gatherUsageStats=false"]