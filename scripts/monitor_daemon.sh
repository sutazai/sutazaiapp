#!/usr/bin/env bash

# SutazAI Autonomous Monitor Daemon Management Script

MONITOR_SCRIPT="/opt/sutazaiapp/scripts/autonomous_monitor.py"
PID_FILE="/var/run/sutazai_monitor.pid"
LOG_FILE="/opt/sutazaiapp/logs/autonomous_monitor/daemon.log"

start_monitor() {
    if [ -f "$PID_FILE" ]; then
        echo "Autonomous Monitor is already running."
        exit 1
    fi

    nohup python3 "$MONITOR_SCRIPT" >> "$LOG_FILE" 2>&1 & 
    echo $! > "$PID_FILE"
    echo "Autonomous Monitor started in background."
}

stop_monitor() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Autonomous Monitor is not running."
        exit 1
    fi

    pid=$(cat "$PID_FILE")
    kill -9 "$pid"
    rm "$PID_FILE"
    echo "Autonomous Monitor stopped."
}

status_monitor() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null; then
            echo "Autonomous Monitor is running (PID: $pid)"
        else
            echo "Autonomous Monitor is not running (stale PID file)"
            rm "$PID_FILE"
        fi
    else
        echo "Autonomous Monitor is not running"
    fi
}

case "$1" in
    start)
        start_monitor
        ;;
    stop)
        stop_monitor
        ;;
    restart)
        stop_monitor
        start_monitor
        ;;
    status)
        status_monitor
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
esac

exit 0 