#!/bin/bash
# Set resource limits for containers
docker update --memory="2g" --cpus="1" $(docker ps -q)
echo "Resource limits set successfully!"

# System Resource Limits Check Script

log_message "=== Starting System Resource Limits Check ==="

# Check system resource limits
log_message "System Resource Limits:"
ulimit -a | while read -r line; do
    log_message "$line"
done

log_message "=== System Resource Limits Check Completed ==="

# Automated resource limit enforcement
AUTO_LIMIT() {
    echo "Starting automated resource limit enforcement..."
    
    # Set CPU limits
    set_cpu_limits() {
        for user in $(cut -d: -f1 /etc/passwd); do
            cpulimit -l 80 -u $user
        done
    }
    
    # Set memory limits
    set_memory_limits() {
        for user in $(cut -d: -f1 /etc/passwd); do
            ulimit -m 1048576
        done
    }
    
    # Set process limits
    set_process_limits() {
        for user in $(cut -d: -f1 /etc/passwd); do
            ulimit -u 1024
        done
    }
    
    set_cpu_limits
    set_memory_limits
    set_process_limits
    echo "Resource limits enforced at $(date)" >> /var/log/resource_limits.log
}

echo "=== Resource Limits ==="
ulimit -a