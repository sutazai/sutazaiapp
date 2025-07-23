FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone AutoGPT and install from the classic directory
RUN git clone https://github.com/Significant-Gravitas/AutoGPT.git autogpt_repo && \
    cd autogpt_repo/classic/original_autogpt && \
    pip install --no-cache-dir -e .

# Copy service files
COPY autogpt_service.py .
COPY requirements.txt .

# Install service dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create workspace
RUN mkdir -p /app/workspace

EXPOSE 8080

CMD ["uvicorn", "autogpt_service:app", "--host", "0.0.0.0", "--port", "8080"]
