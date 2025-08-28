FROM python:3.11-slim

WORKDIR /app

# Configure APT for better reliability with retries and timeouts
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/99retries \
    && echo 'Acquire::http::Timeout "120";' > /etc/apt/apt.conf.d/99timeout \
    && echo 'Acquire::https::Timeout "120";' >> /etc/apt/apt.conf.d/99timeout

# Install system dependencies with proper error handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create indices directory
RUN mkdir -p /app/indices

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]