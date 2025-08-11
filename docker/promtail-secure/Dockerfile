FROM grafana/promtail:2.9.0

# Create non-root user for promtail
RUN groupadd -g 1001 promtail && \
    useradd -u 1001 -g promtail -s /bin/sh -d /home/promtail -m promtail && \
    mkdir -p /etc/promtail /var/log/promtail /home/promtail && \
    chown -R promtail:promtail /etc/promtail /var/log/promtail /home/promtail && \
    chmod 755 /etc/promtail /var/log/promtail

# Config will be mounted via docker-compose volume

# Switch to non-root user
USER promtail

# Promtail default port
EXPOSE 9080

# Run promtail with config
ENTRYPOINT ["/usr/bin/promtail"]
CMD ["-config.file=/etc/promtail/config.yml"]