FROM python:3.11-slim

WORKDIR /app

# Update pip first to avoid issues
RUN pip install --upgrade pip

# Install minimal dependencies
RUN pip install --no-cache-dir \
    httpx \
    pydantic

# Copy agent code
COPY agent.py .

# Run the agent

# Run as non-root user
USER nobody

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["python", "agent.py"]