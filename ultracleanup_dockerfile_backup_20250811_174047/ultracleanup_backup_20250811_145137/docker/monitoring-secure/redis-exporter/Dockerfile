# Secure Redis Exporter Dockerfile
# SECURITY: Runs as non-root user (redis-exporter:redis-exporter)
FROM oliver006/redis_exporter:v1.55.0-alpine AS base

# Create non-root user and group
USER root
RUN addgroup -g 10003 -S redis-exporter && \
    adduser -u 10003 -S redis-exporter -G redis-exporter && \
    mkdir -p /data && \
    chown -R redis-exporter:redis-exporter /data

# Install required tools for healthcheck
RUN apk add --no-cache wget

# Switch to non-root user
USER redis-exporter:redis-exporter

# Set working directory
WORKDIR /

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:9121/health || exit 1

# Expose metrics port
EXPOSE 9121

# Run redis exporter
ENTRYPOINT ["/redis_exporter"]
CMD []