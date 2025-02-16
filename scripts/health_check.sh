#!/bin/bash

check_engine_health() {
    local status=$(systemctl is-active auto-detection-engine)
    if [[ "$status" != "active" ]]; then
        return 1
    fi
    
    # Check if the engine is processing tasks
    local last_activity=$(stat -c %Y /var/log/auto-detection-engine.log)
    local current_time=$(date +%s)
    if (( current_time - last_activity > 300 )); then
        return 2
    fi
    
    return 0
}

check_service_health() {
    local service=$1
    local endpoint=$2
    
    local status=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint")
    if [ "$status" -ne 200 ]; then
        handle_error "Service $service is unhealthy (status: $status)"
    fi
}

case $1 in
    health)
        check_engine_health
        exit $?
        ;;
    *)
        echo "Usage: $0 {health}"
        exit 1
        ;;
esac 