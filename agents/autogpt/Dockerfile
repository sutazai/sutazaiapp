FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone AutoGPT
RUN git clone https://github.com/Significant-Gravitas/AutoGPT.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create configuration
COPY config.yaml .env

# Set up workspace
RUN mkdir -p /app/workspace
VOLUME ["/app/workspace", "/app/data"]

# Expose port
EXPOSE 8080

# Create simple web interface
COPY web_interface.py .

CMD ["python", "web_interface.py"]
