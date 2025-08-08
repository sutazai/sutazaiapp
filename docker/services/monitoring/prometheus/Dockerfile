# Multi-stage Prometheus with security hardening
# Stage 1: Build environment
FROM prom/prometheus:latest as base

# Stage 2: Hardened runtime
FROM alpine:3.19

# Security: Create prometheus user
RUN addgroup -g 65534 prometheus && \
    adduser -D -u 65534 -G prometheus -s /bin/false prometheus

# Install runtime dependencies
RUN apk --no-cache add \
    ca-certificates \
    curl \
    && rm -rf /var/cache/apk/*

# Copy Prometheus binary and files from base image
COPY --from=base /bin/prometheus /bin/prometheus
COPY --from=base /bin/promtool /bin/promtool
COPY --from=base /etc/prometheus /etc/prometheus
COPY --from=base /usr/share/prometheus /usr/share/prometheus

# Create directories with proper permissions
RUN mkdir -p /prometheus /etc/prometheus/rules.d && \
    chown -R prometheus:prometheus /prometheus /etc/prometheus && \
    chmod 755 /prometheus /etc/prometheus

# Copy configuration
COPY --chown=prometheus:prometheus prometheus.yml /etc/prometheus/
COPY --chown=prometheus:prometheus rules/ /etc/prometheus/rules.d/

# Switch to non-root user
USER prometheus

# Expose port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9090/-/healthy || exit 1

# Volume for data persistence
VOLUME ["/prometheus"]

# Run Prometheus

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["prometheus", \
    "--config.file=/etc/prometheus/prometheus.yml", \
    "--storage.tsdb.path=/prometheus", \
    "--web.console.libraries=/usr/share/prometheus/console_libraries", \
    "--web.console.templates=/usr/share/prometheus/consoles", \
    "--storage.tsdb.retention.time=15d", \
    "--web.enable-lifecycle", \
    "--web.enable-admin-api"]