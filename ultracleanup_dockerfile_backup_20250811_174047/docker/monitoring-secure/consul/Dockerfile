# Secure Consul Dockerfile
# SECURITY: Runs as non-root user (consul:consul)
FROM hashicorp/consul:1.17 AS base

# Consul image already has a consul user (UID 100, GID 1000)
# We just need to ensure proper permissions
USER root

# Create necessary directories with proper ownership
RUN mkdir -p /consul/data /consul/config && \
    chown -R consul:consul /consul && \
    chmod 755 /consul/data /consul/config

# Copy config with proper permissions
COPY --chown=consul:consul consul.hcl /consul/config/consul.hcl

# Switch to non-root consul user (already exists in base image)
USER consul:consul

# Set working directory
WORKDIR /consul

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD consul members || exit 1

# Expose ports
EXPOSE 8500 8600 8301 8302

# Run consul agent
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["agent", "-server", "-bootstrap-expect=1", "-ui", "-client=0.0.0.0", "-config-dir=/consul/config"]