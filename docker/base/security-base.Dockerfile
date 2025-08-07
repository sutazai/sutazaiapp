# Multi-stage security-hardened base image
# Stage 1: Build environment
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install security dependencies
COPY requirements-security.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-security.txt

# Stage 2: Hardened runtime environment
FROM python:3.11-slim

# Security: Create restricted user for security tools
RUN groupadd -r security && useradd -r -g security -s /bin/false security

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Security hardening environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SECURITY_MODE=strict \
    LOG_LEVEL=INFO \
    HEALTH_PORT=8080

# Create secure directories
RUN mkdir -p /app /app/data /app/logs /app/reports && \
    chown -R security:security /app && \
    chmod 750 /app

# Security: Remove unnecessary packages and clean
RUN apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/cache/apt/* /tmp/* /var/tmp/*

WORKDIR /app

# Switch to non-root user
USER security

# Health check for security services
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${HEALTH_PORT:-8080}/health || exit 1

# Default command
CMD ["python", "-c", "print('Security base image - override CMD in specific service')"]