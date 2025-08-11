FROM gcr.io/cadvisor/cadvisor:latest

# cAdvisor requires access to host system metrics, but we can still run with limited privileges
# Create a non-root user for cAdvisor
USER root
RUN addgroup -g 1001 cadvisor && \
    adduser -u 1001 -G cadvisor -s /bin/sh -D cadvisor && \
    # Create necessary directories
    mkdir -p /var/lib/cadvisor /tmp/cadvisor && \
    chown -R cadvisor:cadvisor /var/lib/cadvisor /tmp/cadvisor && \
    chmod 755 /var/lib/cadvisor /tmp/cadvisor

# Switch to non-root user
USER cadvisor

# cAdvisor port
EXPOSE 8080

# Use the default entrypoint from base image with security options
ENTRYPOINT ["/usr/bin/cadvisor"]
CMD ["--housekeeping_interval=60s", \
     "--max_housekeeping_interval=120s", \
     "--allow_dynamic_housekeeping=true", \
     "--global_housekeeping_interval=2m", \
     "--docker_only=true", \
     "--disable_root_cgroup_stats=true", \
     "--store_container_labels=false"]