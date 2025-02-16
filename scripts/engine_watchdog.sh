#!/bin/bash

ENGINE_PID_FILE="/var/run/auto-detection-engine.pid"
ENGINE_LOG="/var/log/auto-detection-engine.log"
MAX_RESTARTS=5
RESTART_COUNT=0

monitor_engine() {
    while true; do
        if ! pgrep -F "$ENGINE_PID_FILE" >/dev/null; then
            if (( RESTART_COUNT >= MAX_RESTARTS )); then
                send_notification "Auto-Detection Engine failed to start after $MAX_RESTARTS attempts" "CRITICAL"
                exit 1
            fi
            
            start_engine
            ((RESTART_COUNT++))
        fi
        sleep 60
    done
}

start_engine() {
    /opt/sutazai/scripts/auto_detection_engine.sh &
    echo $! > "$ENGINE_PID_FILE"
    send_notification "Auto-Detection Engine started (Attempt: $((RESTART_COUNT + 1)))" "INFO"
}

monitor_engine 