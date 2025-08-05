# Base Alpine image with health check tools
FROM alpine:3.18

# Install health check dependencies
RUN apk add --no-cache \
    curl \
    wget \
    netcat-openbsd \
    python3 \
    py3-pip \
    bash

# Add health check script
COPY health-check.sh /usr/local/bin/health-check
RUN chmod +x /usr/local/bin/health-check

# Set default health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/health-check || exit 1
