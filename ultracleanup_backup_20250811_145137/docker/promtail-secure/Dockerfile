FROM grafana/promtail:2.9.0

# Create non-root user for promtail
RUN addgroup -g 1001 promtail && \
    adduser -u 1001 -G promtail -s /bin/sh -D promtail && \
    mkdir -p /etc/promtail /var/log/promtail && \
    chown -R promtail:promtail /etc/promtail /var/log/promtail && \
    chmod 755 /etc/promtail /var/log/promtail

# Copy config with proper ownership
COPY --chown=promtail:promtail ./monitoring/promtail/config.yml /etc/promtail/config.yml

# Switch to non-root user
USER promtail

# Promtail default port
EXPOSE 9080

# Run promtail with config
ENTRYPOINT ["/usr/bin/promtail"]
CMD ["-config.file=/etc/promtail/config.yml"]