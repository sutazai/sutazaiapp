FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install ShellGPT
RUN pip install shell-gpt flask requests

VOLUME ["/app/data"]

# Create web interface
COPY web_interface.py .

CMD ["python", "web_interface.py"]
