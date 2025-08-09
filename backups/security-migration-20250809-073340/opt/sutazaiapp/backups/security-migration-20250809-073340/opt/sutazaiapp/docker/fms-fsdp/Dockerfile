FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fsdp_service.py .
COPY health_check.py .

EXPOSE 8000


# Run as non-root user
USER nobody

# Run as non-root user
USER nobody

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "fsdp_service:app", "--host", "0.0.0.0", "--port", "8000"]