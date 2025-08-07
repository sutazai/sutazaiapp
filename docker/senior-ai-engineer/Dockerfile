FROM python:3.12-slim

WORKDIR /agent

# Copy requirements and install dependencies
COPY requirements.txt* ./
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || pip install fastapi uvicorn requests httpx asyncio

# Copy agent files
COPY . .

# Make startup script executable
RUN chmod +x startup.sh 2>/dev/null || echo "No startup.sh found"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the agent

# Run as non-root user
USER nobody
CMD ["python", "agent.py"]