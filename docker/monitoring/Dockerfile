# SutazAI Monitoring Stack - Consolidated Configuration
# Unified monitoring utilities and exporters
# Last Updated: 2025-08-20

FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    git

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install monitoring dependencies
WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    prometheus-client \
    psutil \
    flask \
    requests \
    redis \
    psycopg2-binary \
    neo4j

# Production stage
FROM python:3.11-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    bash \
    postgresql-client

# Create non-root user
RUN addgroup -g 1000 monitor && \
    adduser -D -u 1000 -G monitor monitor

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create necessary directories
RUN mkdir -p /app /config /var/log/monitoring && \
    chown -R monitor:monitor /app /config /var/log/monitoring

# Set working directory
WORKDIR /app

# Copy monitoring scripts
COPY --chown=monitor:monitor exporters/ /app/exporters/
COPY --chown=monitor:monitor collectors/ /app/collectors/
COPY --chown=monitor:monitor config/ /config/

# Environment variables
ENV MONITORING_PORT=9090 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER monitor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${MONITORING_PORT}/health || exit 1

# Expose metrics port
EXPOSE 9090

# Run monitoring exporter
CMD ["python", "-m", "exporters.unified_exporter"]