# JARVIS Unified Voice Interface
FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    # Audio libraries
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    pulseaudio-utils \
    alsa-utils \
    # Speech recognition dependencies
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    # System tools
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/data \
    /app/logs \
    /app/config \
    /app/plugins \
    /app/static \
    /app/templates

# Copy static files and templates
COPY static/ /app/static/
COPY templates/ /app/templates/

# Set audio permissions
RUN usermod -a -G audio root || true

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Run the application
CMD ["python", "main.py"]