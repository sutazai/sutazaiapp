FROM python:3.12.8-slim-bookworm

WORKDIR /app

# Install minimal system dependencies for security
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements_minimal.txt .
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Copy application code
COPY . .
# Use minimal main.py for now
RUN cp app/main_minimal.py app/main.py

ENV PYTHONUNBUFFERED=1
ENV DEBUG=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
