FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Docker CLI
RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    cron \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for code improvement
RUN pip install --no-cache-dir \
    aider-chat \
    semgrep \
    black \
    isort \
    pylint \
    mypy \
    gitpython \
    schedule \
    fastapi \
    uvicorn \
    httpx \
    pydantic

# Copy service implementation
COPY code_improver.py /app/
COPY improve_cron.sh /app/

# Make scripts executable
RUN chmod +x /app/improve_cron.sh

# Set up cron job for autonomous improvement
RUN echo "0 */6 * * * /app/improve_cron.sh" | crontab -

EXPOSE 8080

# Start both cron and the web service
CMD service cron start && uvicorn code_improver:app --host 0.0.0.0 --port 8080