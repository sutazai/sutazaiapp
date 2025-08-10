# Migrated to Master Base Image
# Original: alpine:3.18
# Migration Date: 2025-08-10T19:58:26.112835
FROM sutazai-python-agent-master:latest

# Service-specific setup
RUN chmod +x /usr/local/bin/health-check
RUN groupadd -r appuser && useradd -r -g appuser appuser \

# Copy application code
COPY health-check.sh /usr/local/bin/health-check
COPY . .

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \

# Security: Run as non-root user
USER appuser

CMD /usr/local/bin/health-check || exit 1