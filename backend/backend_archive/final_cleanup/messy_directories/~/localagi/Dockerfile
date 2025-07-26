FROM python:3.10-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create simple LocalAGI server
COPY . .

# Create startup script
RUN echo '#\!/bin/bash\ncd /app && python -m uvicorn main:app --host 0.0.0.0 --port 8080' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
EOF < /dev/null
