FROM hashicorp/consul:1.17

# Consul already has a consul user (UID 100, GID 1000)
# We need to ensure proper permissions for data and config directories

# Create necessary directories and fix permissions as root
USER root
RUN mkdir -p /consul/data /consul/config && \
    chown -R consul:consul /consul && \
    chmod -R 755 /consul && \
    # Ensure consul user can write to data directory
    chmod 777 /consul/data

# Copy configuration with proper ownership
COPY --chown=consul:consul ./config/consul/consul.hcl /consul/config/consul.hcl

# Switch to consul user (non-root)
USER consul

# Consul ports
EXPOSE 8300 8301 8301/udp 8302 8302/udp 8500 8600 8600/udp

# Run consul with proper configuration
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["agent", "-server", "-bootstrap-expect=1", "-ui", "-client=0.0.0.0", "-config-dir=/consul/config"]