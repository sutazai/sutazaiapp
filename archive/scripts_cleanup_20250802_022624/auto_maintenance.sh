#!/bin/bash
# SutazAI Auto Maintenance Script
# This script runs maintenance tasks to keep the system running optimally.

# Set the path to the application
APP_ROOT="/opt/sutazaiapp"
SCRIPTS_DIR="$APP_ROOT/scripts"
LOGS_DIR="$APP_ROOT/logs"
PYTHON="python3"

# Create maintenance log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOGS_DIR/maintenance_$TIMESTAMP.log"

# Ensure log directory exists
mkdir -p "$LOGS_DIR"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

# Start maintenance
log "Starting SutazAI auto maintenance"

# Check for running processes
log "Checking for running processes"
ps aux | grep -i "sutazai" | grep -v grep | tee -a "$LOG_FILE"

# Clean up stale PID files
log "Cleaning up stale PID files"
for pid_file in "$APP_ROOT/pids"/*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        process_running=$(ps -p "$pid" > /dev/null 2>&1; echo $?)
        
        if [ $process_running -ne 0 ]; then
            log "Removing stale PID file: $pid_file (PID: $pid)"
            rm "$pid_file"
        else
            log "Valid PID file: $pid_file (PID: $pid)"
                fi
            fi
        done

# Rotate large log files
log "Rotating large log files"
find "$LOGS_DIR" -name "*.log" -type f -size +10M | while read -r log_file; do
    log "Rotating large log file: $log_file"
    mv "$log_file" "${log_file}.old"
                touch "$log_file"
done

# Clean up temporary files
log "Cleaning up temporary files"
find "$APP_ROOT/tmp" -type f -mtime +7 -delete 2>/dev/null
find "$APP_ROOT/tmp" -type d -empty -delete 2>/dev/null

# Run system optimizer
log "Running system optimizer"
cd "$APP_ROOT" && $PYTHON "$SCRIPTS_DIR/system_optimizer.py" | tee -a "$LOG_FILE"

# Optimize transformer models
log "Optimizing transformer models"
if [ -f "$SCRIPTS_DIR/optimize_transformers.py" ]; then
    cd "$APP_ROOT" && $PYTHON "$SCRIPTS_DIR/optimize_transformers.py" | tee -a "$LOG_FILE"
fi

# Run database maintenance
log "Running database maintenance"
if [ -f "$APP_ROOT/storage/sutazai.db" ]; then
    sqlite3 "$APP_ROOT/storage/sutazai.db" <<EOF
PRAGMA integrity_check;
PRAGMA optimize;
VACUUM;
EOF
    log "Database maintenance completed"
fi

# Ensure all services are running
log "Ensuring all services are running"

# Check and restart backend if needed
if ! pgrep -f "backend" > /dev/null; then
    log "Starting backend service"
    if [ -f "$APP_ROOT/bin/start_backend.sh" ]; then
        bash "$APP_ROOT/bin/start_backend.sh" >> "$LOG_FILE" 2>&1
    elif [ -f "$APP_ROOT/systemd/backend.service" ]; then
        systemctl --user restart backend >> "$LOG_FILE" 2>&1
    else
        cd "$APP_ROOT" && $PYTHON -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level info >> "$LOGS_DIR/backend.log" 2>&1 &
        echo $! > "$APP_ROOT/pids/backend.pid"
    fi
fi

# Check and restart web UI if needed
if ! pgrep -f "webui" > /dev/null; then
    log "Starting web UI service"
    if [ -f "$APP_ROOT/bin/start_webui.sh" ]; then
        bash "$APP_ROOT/bin/start_webui.sh" >> "$LOG_FILE" 2>&1
    elif [ -f "$APP_ROOT/systemd/webui.service" ]; then
        systemctl --user restart webui >> "$LOG_FILE" 2>&1
    else
        if [ -f "$APP_ROOT/web_ui/package.json" ]; then
            cd "$APP_ROOT/web_ui" && npm run start >> "$LOGS_DIR/webui.log" 2>&1 &
            echo $! > "$APP_ROOT/pids/webui.pid"
        fi
    fi
fi

# Create a summary of running services
log "Creating service status summary"
{
    echo "=== Service Status ==="
    echo "Backend: $(pgrep -f 'backend' > /dev/null && echo 'Running' || echo 'Stopped')"
    echo "Web UI: $(pgrep -f 'webui' > /dev/null && echo 'Running' || echo 'Stopped')"
    echo "Vector DB: $(pgrep -f 'vector-db' > /dev/null && echo 'Running' || echo 'Stopped')"
    echo "LocalAGI: $(pgrep -f 'localagi' > /dev/null && echo 'Running' || echo 'Stopped')"
    echo "======================"
} | tee -a "$LOG_FILE"

# Create a maintenance summary
REPORT_FILE="$LOGS_DIR/maintenance_report.txt"
{
    echo "SutazAI Maintenance Report"
    echo "=========================="
    echo "Timestamp: $(date)"
    echo ""
    echo "System Information:"
    echo "  - CPU Usage: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}')%"
    echo "  - Memory Usage: $(free -m | grep Mem | awk '{print $3 " MB used, " $2 " MB total"}')"
    echo "  - Disk Usage: $(df -h / | grep / | awk '{print $5 " used, " $4 " available"}')"
    echo ""
    echo "Services Status:"
    echo "  - Backend: $(pgrep -f 'backend' > /dev/null && echo 'Running' || echo 'Stopped')"
    echo "  - Web UI: $(pgrep -f 'webui' > /dev/null && echo 'Running' || echo 'Stopped')"
    echo "  - Vector DB: $(pgrep -f 'vector-db' > /dev/null && echo 'Running' || echo 'Stopped')"
    echo "  - LocalAGI: $(pgrep -f 'localagi' > /dev/null && echo 'Running' || echo 'Stopped')"
    echo ""
    echo "Maintenance Tasks:"
    echo "  - System optimization completed"
    echo "  - Database maintenance completed"
    echo "  - Log rotation completed"
    echo "  - Temporary file cleanup completed"
    echo ""
    echo "For detailed information, see: $LOG_FILE"
} > "$REPORT_FILE"

log "Auto maintenance completed successfully"
log "Maintenance summary written to: $REPORT_FILE"

exit 0 