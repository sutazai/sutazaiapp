FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    crewai \
    crewai-tools \
    fastapi \
    uvicorn \
    requests \
    openai \
    pydantic \
    python-multipart

# Copy service files
COPY crewai_service.py .
COPY requirements.txt .

# Install additional requirements if any
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "crewai_service:app", "--host", "0.0.0.0", "--port", "8080"]