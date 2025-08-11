FROM hashicorp/consul:1.17

# ULTRA-FIX: Fix setgroups permission issue by using proper user configuration
# Create consul user with specific UID/GID to avoid conflicts

USER root

# Remove existing consul user and recreate with proper permissions
RUN deluser consul 2>/dev/null || true && \
    delgroup consul 2>/dev/null || true && \
    addgroup -g 1000 consul && \
    adduser -D -u 100 -G consul -h /consul -s /bin/sh consul

# Create necessary directories and fix permissions
RUN mkdir -p /consul/data /consul/config /consul/logs && \
    chown -R consul:consul /consul && \
    chmod -R 755 /consul && \
    # Ensure consul user can write to data directory
    chmod 775 /consul/data /consul/logs

# Create default consul config
RUN echo "# Default consul config\n\
datacenter = \"dc1\"\n\
data_dir = \"/consul/data\"\n\
log_level = \"INFO\"\n\
server = true\n\
ui_config {\n\
  enabled = true\n\
}\n\
connect {\n\
  enabled = true\n\
}\n\
client_addr = \"0.0.0.0\"" > /consul/config/consul.hcl && \
    chown consul:consul /consul/config/consul.hcl

# ULTRA-FIX: Use direct USER switch instead of su-exec to avoid setgroups issue
USER consul

# Consul ports
EXPOSE 8300 8301 8301/udp 8302 8302/udp 8500 8600 8600/udp

# Working directory
WORKDIR /consul

# ULTRA-FIX: Use direct consul binary instead of docker-entrypoint.sh to avoid su-exec
ENTRYPOINT ["/bin/consul"]
CMD ["agent", "-server", "-bootstrap-expect=1", "-ui", "-client=0.0.0.0", "-bind=0.0.0.0", "-data-dir=/consul/data"]