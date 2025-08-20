# SutazAI Frontend Service - Rule 11 Compliant
# Secure, non-root, pinned versions
# Last Updated: 2025-01-15
FROM python:3.11.9-alpine3.19

# Security: Create non-root user
RUN addgroup -g 1000 -S appgroup && \
    adduser -u 1000 -S appuser -G appgroup -h /home/appuser -s /bin/sh

# Install system dependencies
RUN apk update && \
    apk upgrade && \
    apk add --no-cache \
        curl \
        ca-certificates && \
    rm -rf /var/cache/apk/*

WORKDIR /app

# Copy and install requirements as root
COPY requirements_optimized.txt /tmp/frontend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/frontend-requirements.txt && \
    rm /tmp/frontend-requirements.txt

# Switch to non-root user
USER appuser

# Copy application and components with proper ownership
COPY --chown=appuser:appgroup app.py .
COPY --chown=appuser:appgroup components/ ./components/
COPY --chown=appuser:appgroup pages/ ./pages/
COPY --chown=appuser:appgroup utils/ ./utils/

# Override base environment variables for frontend service
ENV SERVICE_PORT=8501
ENV AGENT_ID=sutazai-frontend
ENV AGENT_NAME="SutazAI Frontend Service"
ENV PORT=8501

# Override health check for frontend service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Expose frontend port
EXPOSE 8501

# Ensure we're running as non-root user (already set above, but explicit for clarity)
USER appuser

# Start frontend service
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
