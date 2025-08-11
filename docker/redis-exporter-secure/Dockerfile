FROM oliver006/redis_exporter:latest

# Redis exporter runs as nobody by default in alpine images
# We ensure proper permissions and security settings

# Create necessary directories with proper ownership
USER root
RUN addgroup -g 1001 redis-exporter && \
    adduser -u 1001 -G redis-exporter -s /bin/sh -D redis-exporter && \
    mkdir -p /var/lib/redis-exporter && \
    chown -R redis-exporter:redis-exporter /var/lib/redis-exporter && \
    chmod 755 /var/lib/redis-exporter

# Switch to non-root user
USER redis-exporter

# Redis exporter port
EXPOSE 9121

# Run redis exporter
ENTRYPOINT ["/redis_exporter"]