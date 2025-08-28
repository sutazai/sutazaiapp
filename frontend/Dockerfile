# JARVIS Frontend Dockerfile
FROM python:3.11-slim

# Install system dependencies for audio processing and health checks
RUN apt-get update && apt-get install -y \
    curl \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    espeak \
    libespeak-dev \
    alsa-utils \
    pulseaudio \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional audio dependencies
RUN pip install --no-cache-dir pyaudio

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/temp /app/logs /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=11000
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_THEME_BASE=dark
ENV STREAMLIT_THEME_PRIMARY_COLOR="#00D4FF"

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