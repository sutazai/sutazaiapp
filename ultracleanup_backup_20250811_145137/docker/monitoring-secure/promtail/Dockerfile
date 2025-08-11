# Secure Promtail Dockerfile
# SECURITY: Runs as non-root user (promtail:promtail)
FROM grafana/promtail:2.9.0 AS base

# Create non-root user and group
USER root
RUN groupadd -g 10001 promtail && \
    useradd -u 10001 -g promtail -s /bin/false -M promtail && \
    mkdir -p /var/log/positions /tmp/promtail && \
    chown -R promtail:promtail /var/log/positions /tmp/promtail

# Copy config with proper permissions
COPY --chown=promtail:promtail config.yml /etc/promtail/config.yml

# Switch to non-root user
USER promtail:promtail

# Set working directory
WORKDIR /usr/bin

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3100/ready || exit 1

# Run promtail with config
ENTRYPOINT ["/usr/bin/promtail"]
CMD ["-config.file=/etc/promtail/config.yml", "-config.expand-env=true"]