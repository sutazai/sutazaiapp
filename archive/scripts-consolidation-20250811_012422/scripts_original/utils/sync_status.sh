#!/bin/bash

# Strict error handling
set -euo pipefail

# Advanced sync status script

# Source configuration

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

source /opt/sutazaiapp/scripts/config/sync_config.sh

# Output format (text, json)
FORMAT="text"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --json)
            FORMAT="json"
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--json] [--verbose|-v]"
            exit 1
            ;;
    esac
done

# Determine current server
CURRENT_IP=$(hostname -I | awk '{print $1}')
if [[ "$CURRENT_IP" == "$CODE_SERVER" ]]; then
    SERVER_TYPE="code"
    REMOTE_SERVER="$DEPLOY_SERVER"
elif [[ "$CURRENT_IP" == "$DEPLOY_SERVER" ]]; then
    SERVER_TYPE="deploy"
    REMOTE_SERVER="$CODE_SERVER"
else
    echo "Unknown server type. Not part of the sync configuration."
    exit 1
fi

# Check if remote server is reachable
REMOTE_REACHABLE=false
if ping -c 1 $REMOTE_SERVER &> /dev/null; then
    REMOTE_REACHABLE=true
fi

# Check sync service status
SERVICE_STATUS=$(systemctl is-active sutazai-sync-monitor.service 2>/dev/null || echo "inactive")

# Get last sync time
LAST_SYNC_LOG=$(find "$PROJECT_ROOT/logs/sync/" -name "sync_*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$LAST_SYNC_LOG" ]; then
    LAST_SYNC_TIME=$(stat -c %Y "$LAST_SYNC_LOG")
    CURRENT_TIME=$(date +%s)
    TIME_DIFF=$((CURRENT_TIME - LAST_SYNC_TIME))
    LAST_SYNC_HUMAN="$(date -d @$LAST_SYNC_TIME '+%Y-%m-%d %H:%M:%S') ($(($TIME_DIFF / 60)) minutes ago)"
    
    # Check for errors in last sync
    if grep -q "ERROR" "$LAST_SYNC_LOG"; then
        LAST_SYNC_STATUS="error"
        LAST_SYNC_ERROR=$(grep "ERROR" "$LAST_SYNC_LOG" | head -1)
    else
        LAST_SYNC_STATUS="success"
        LAST_SYNC_ERROR=""
    fi
else
    LAST_SYNC_HUMAN="Never"
    LAST_SYNC_STATUS="unknown"
    LAST_SYNC_ERROR=""
fi

# Get Git status if on code server
GIT_STATUS=""
if [[ "$SERVER_TYPE" == "code" ]] && [ -d "$PROJECT_ROOT/.git" ]; then
    cd $PROJECT_ROOT
    GIT_STATUS=$(git status --short)
fi

# System metrics
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
MEM_USAGE=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
DISK_USAGE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}')

# Output in specified format
if [ "$FORMAT" == "json" ]; then
    # JSON output
    cat << EOF
{
    "server": {
        "type": "$SERVER_TYPE",
        "remote_server": "$REMOTE_SERVER",
        "remote_reachable": $REMOTE_REACHABLE
    },
    "sync": {
        "service_status": "$SERVICE_STATUS",
        "last_sync": "$LAST_SYNC_HUMAN",
        "status": "$LAST_SYNC_STATUS"
    },
    "system": {
        "cpu_usage": "$CPU_USAGE%",
        "memory_usage": "$MEM_USAGE%",
        "disk_usage": "$DISK_USAGE"
    }
EOF

    # Add git status if available
    if [ -n "$GIT_STATUS" ]; then
        echo "    ,\"git\": {"
        echo "        \"has_changes\": $([ -n "$GIT_STATUS" ] && echo "true" || echo "false")"
        if [ "$VERBOSE" = true ]; then
            echo "        ,\"changes\": ["
            while IFS= read -r line; do
                echo "            \"$line\","
            done < <(echo "$GIT_STATUS" | sed 's/"/\\"/g' | sed '$!s/$/,/')
            echo "        ]"
        fi
        echo "    }"
    fi

    # Add error if available
    if [ -n "$LAST_SYNC_ERROR" ]; then
        echo "    ,\"error\": \"$LAST_SYNC_ERROR\""
    fi

    echo "}"
else
    # Text output
    echo "======== SutazAI Sync Status ========"
    echo "Server: $SERVER_TYPE (Remote: $REMOTE_SERVER)"
    echo "Remote reachable: $([ "$REMOTE_REACHABLE" = true ] && echo "Yes" || echo "No")"
    echo "Sync service: $SERVICE_STATUS"
    echo "Last sync: $LAST_SYNC_HUMAN"
    echo "Sync status: $LAST_SYNC_STATUS"
    
    if [ -n "$LAST_SYNC_ERROR" ]; then
        echo "Sync error: $LAST_SYNC_ERROR"
    fi
    
    echo ""
    echo "System metrics:"
    echo "  CPU usage: $CPU_USAGE%"
    echo "  Memory usage: $MEM_USAGE%"
    echo "  Disk usage: $DISK_USAGE"
    
    if [ -n "$GIT_STATUS" ]; then
        echo ""
        echo "Git status (code server):"
        if [ "$VERBOSE" = true ]; then
            echo "$GIT_STATUS"
        else
            echo "  $(echo "$GIT_STATUS" | wc -l) uncommitted changes"
            echo "  (Use --verbose to see details)"
        fi
    fi
fi 