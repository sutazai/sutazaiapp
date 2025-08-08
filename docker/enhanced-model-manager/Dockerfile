FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi uvicorn requests torch transformers sentence-transformers

COPY model_manager.py .

EXPOSE 8000

# Run as non-root user
USER nobody

# Run as non-root user
USER nobody

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "model_manager:app", "--host", "0.0.0.0", "--port", "8000"]
