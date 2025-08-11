FROM prom/blackbox-exporter:latest

# Blackbox exporter already runs as nobody user by default
# We just need to ensure proper permissions and security settings

# Create necessary directories with proper ownership
USER root
RUN mkdir -p /etc/blackbox_exporter /var/lib/blackbox_exporter && \
    chown -R nobody:nobody /etc/blackbox_exporter /var/lib/blackbox_exporter && \
    chmod 755 /etc/blackbox_exporter /var/lib/blackbox_exporter

# Copy configuration with proper ownership
COPY --chown=nobody:nobody ./monitoring/blackbox/config.yml /etc/blackbox_exporter/config.yml

# Switch to nobody user (non-root)
USER nobody

# Blackbox exporter port
EXPOSE 9115

# Run blackbox exporter with config
ENTRYPOINT ["/bin/blackbox_exporter"]
CMD ["--config.file=/etc/blackbox_exporter/config.yml"]