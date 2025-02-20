#!/bin/bash
# Monitor critical services
SERVICES=("docker" "nginx" "postgresql")
for service in "${SERVICES[@]}"; do
    if ! systemctl is-active --quiet $service; then
        systemctl restart $service
        echo "Restarted $service"
    fi
done 