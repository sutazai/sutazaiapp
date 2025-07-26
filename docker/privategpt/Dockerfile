FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone PrivateGPT
RUN git clone https://github.com/imartinez/privateGPT.git .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration
COPY settings.yaml settings.yaml

EXPOSE 8001

CMD ["python", "-m", "scripts.setup", "--profile", "local"]
