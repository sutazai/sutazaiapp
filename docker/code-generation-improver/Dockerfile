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
CMD ["python", "agent.py"]