# Perfect Jarvis Voice Interface Docker Container
FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    portaudio19-dev \
    python3-pyaudio \
    alsa-utils \
    pulseaudio \
    espeak \
    espeak-data \
    libespeak-dev \
    flac \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/temp

# Set permissions
RUN chmod +x /app/app.py

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=5 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(('localhost',8080))==0 else 1)"

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run the application
CMD ["python", "app.py"]