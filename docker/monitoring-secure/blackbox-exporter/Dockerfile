FROM prom/blackbox-exporter:v0.25.0

# Switch to nobody user for security
USER nobody

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:9115/ || exit 1

# Default command
CMD ["/bin/blackbox_exporter", "--config.file=/etc/blackbox_exporter/config.yml"]