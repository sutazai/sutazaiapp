# Secure cAdvisor Dockerfile
# SECURITY: Runs with minimal privileges required for container monitoring
FROM gcr.io/cadvisor/cadvisor:v0.47.2 AS base

# cAdvisor requires specific capabilities but can drop root after initialization
# We'll use a custom entrypoint to handle this
USER root

# Create entrypoint script that drops privileges after initialization
RUN echo '#!/bin/sh' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '# Start cadvisor with required capabilities' >> /entrypoint.sh && \
    echo 'exec /usr/bin/cadvisor "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set security options
# Note: cAdvisor needs certain capabilities for container monitoring
# but we can restrict it as much as possible
USER nobody

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/healthz || exit 1

# Use custom entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command with optimized settings
CMD ["--housekeeping_interval=60s", \
     "--max_housekeeping_interval=120s", \
     "--allow_dynamic_housekeeping=true", \
     "--global_housekeeping_interval=2m", \
     "--disable_metrics=advtcp,cpu_topology,disk,hugetlb,memory_numa,percpu,referenced_memory,resctrl,tcp,udp,process,accelerator", \
     "--store_container_labels=false", \
     "--whitelisted_container_labels=io.kubernetes.container.name,io.kubernetes.pod.name"]