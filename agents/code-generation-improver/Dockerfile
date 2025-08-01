FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir \
    httpx \
    asyncio \
    pydantic

# Copy agent code
COPY agent.py .

# Run the agent
CMD ["python", "agent.py"]