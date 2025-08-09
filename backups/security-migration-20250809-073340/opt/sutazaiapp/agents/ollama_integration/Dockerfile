FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/schemas /app/agents/core

# Copy application code
COPY app.py .
COPY ollama_schemas.py ./schemas/
COPY metrics.py ./agents/core/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Run the service
EXPOSE 8090

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]