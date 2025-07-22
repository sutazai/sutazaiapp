FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone RealtimeSTT
RUN git clone https://github.com/KoljaB/RealtimeSTT.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask requests whisper

VOLUME ["/app/data"]
EXPOSE 8080

# Create web interface
COPY web_interface.py .

CMD ["python", "web_interface.py"]
