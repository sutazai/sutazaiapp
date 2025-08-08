FROM python:3.11-slim

LABEL maintainer="Sutazai Rule Control Team"
LABEL description="Intelligent Rule Control API"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create app user  
RUN groupadd -r rulecontrol && useradd -r -g rulecontrol rulecontrol

WORKDIR /app

# Copy requirements
COPY docker/rule-control/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY scripts/agents/rule-control-manager.py ./app.py
COPY docker/rule-control/entrypoint.sh ./entrypoint.sh

# Create directories
RUN mkdir -p /app/config /app/logs && \
    chown -R rulecontrol:rulecontrol /app

RUN chmod +x ./entrypoint.sh

USER rulecontrol

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8100/api/health/live || exit 1

EXPOSE 8100


# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["./entrypoint.sh"]