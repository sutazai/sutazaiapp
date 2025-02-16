#!/bin/bash

check_sync_status() {
    local services=("code-sync" "auto-detection-engine" "resource-monitor")
    for service in "${services[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            echo "❌ $service is not running"
            return 1
        fi
    done
    echo "✅ All sync services are running"
    return 0
}

case $1 in
    status)
        check_sync_status
        exit $?
        ;;
    *)
        echo "Usage: $0 {status}"
        exit 1
        ;;
esac 