FROM gcr.io/cadvisor/cadvisor:v0.52.1

# Switch to nobody user for security
USER nobody

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/healthz || exit 1

# Default command with security optimizations
CMD ["/usr/bin/cadvisor", \
     "--housekeeping_interval=60s", \
     "--max_housekeeping_interval=120s", \
     "--allow_dynamic_housekeeping=true", \
     "--global_housekeeping_interval=2m", \
     "--disable_metrics=advtcp,cpu_topology,disk,hugetlb,memory_numa,percpu,referenced_memory,resctrl,tcp,udp,process", \
     "--store_container_labels=false", \
     "--whitelisted_container_labels=io.kubernetes.container.name,io.kubernetes.pod.name"]