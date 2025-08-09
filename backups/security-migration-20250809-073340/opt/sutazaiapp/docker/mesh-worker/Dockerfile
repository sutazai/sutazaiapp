FROM python:3.11-slim

WORKDIR /app

# Install only what is needed for the worker
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy minimal code needed for the worker
COPY scripts/mesh/ /app/scripts/mesh/
COPY agents/agent_with_health.py /app/agents/agent_with_health.py
COPY backend/app/mesh/ /app/backend/app/mesh/

ENV PYTHONPATH=/app
ENV REDIS_URL=redis://redis:6379/0
ENV LOG_LEVEL=INFO

# Defaults (can be overridden via environment)
ENV MESH_TOPIC=nlp
ENV MESH_GROUP=nlp
ENV MESH_CONSUMER=worker-1
ENV MESH_RESULTS=nlp


# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["python", "scripts/mesh/agent_stream_consumer.py", "--topic", "${MESH_TOPIC}", "--group", "${MESH_GROUP}", "--consumer", "${MESH_CONSUMER}", "--results", "${MESH_RESULTS}"]

