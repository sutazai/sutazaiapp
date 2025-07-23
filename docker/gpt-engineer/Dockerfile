FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install GPT-Engineer from PyPI
RUN pip install --no-cache-dir gpt-engineer

# Copy service files
COPY gpt_engineer_service.py .
COPY requirements.txt .

# Install service dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create workspace
RUN mkdir -p /app/workspace

EXPOSE 8080

CMD ["uvicorn", "gpt_engineer_service:app", "--host", "0.0.0.0", "--port", "8080"]
