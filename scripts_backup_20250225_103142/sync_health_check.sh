#!/bin/bash

check_sync_health() {
    # Check if service is running
    if ! systemctl is-active --quiet code-sync; then
        return 1
    fi
    
    # Check last sync time
    local last_sync=$(stat -c %Y /var/log/code-sync.log)
    local current_time=$(date +%s)
    if (( current_time - last_sync > SYNC_INTERVAL * 2 )); then
        return 2
    fi
    
    return 0
}

case $1 in
    health)
        check_sync_health
        exit $?
        ;;
    *)
        echo "Usage: $0 {health}"
        exit 1
        ;;
esac 