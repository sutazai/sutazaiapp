# Secure Blackbox Exporter Dockerfile
# SECURITY: Runs as non-root user (nobody)
FROM prom/blackbox-exporter:v0.24.0 AS base

# The base image is minimal and already has nobody user
# We'll use nobody instead of creating a new user
USER root
RUN mkdir -p /etc/blackbox_exporter /data && \
    chown -R nobody:nobody /etc/blackbox_exporter /data

# Copy config with proper permissions
COPY --chown=nobody:nobody config.yml /etc/blackbox_exporter/config.yml

# Switch to non-root user
USER nobody

# Set working directory
WORKDIR /bin

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:9115/health || exit 1

# Expose metrics port
EXPOSE 9115

# Run blackbox exporter
ENTRYPOINT ["/bin/blackbox_exporter"]
CMD ["--config.file=/etc/blackbox_exporter/config.yml", \
     "--web.listen-address=:9115", \
     "--log.level=info"]