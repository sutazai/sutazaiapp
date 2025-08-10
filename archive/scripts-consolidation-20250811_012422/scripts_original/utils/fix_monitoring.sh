#!/bin/bash

# Strict error handling
set -euo pipefail

#
# fix_monitoring.sh - Script to fix and cleanup stuck monitoring processes
#

# Define color codes

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

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting cleanup of monitoring processes...${NC}"

# 1. Stop systemd services first (if they exist)
echo -e "${YELLOW}Stopping systemd services...${NC}"
sudo systemctl stop sutazai-prometheus sutazai-node-exporter 2>/dev/null || true
sleep 2

# 2. Find and kill Prometheus processes
echo -e "${YELLOW}Finding and terminating Prometheus processes...${NC}"
PROM_PIDS=$(pgrep -f prometheus || true)
if [ -n "$PROM_PIDS" ]; then
    echo -e "Found Prometheus PIDs: $PROM_PIDS"
    for pid in $PROM_PIDS; do
        sudo kill -15 $pid 2>/dev/null || true
    done
    sleep 2
    # Force kill any remaining
    PROM_PIDS=$(pgrep -f prometheus || true)
    if [ -n "$PROM_PIDS" ]; then
        echo -e "${YELLOW}Force killing remaining Prometheus processes...${NC}"
        for pid in $PROM_PIDS; do
            sudo kill -9 $pid 2>/dev/null || true
        done
    fi
else
    echo -e "${GREEN}No Prometheus processes found.${NC}"
fi

# 3. Find and kill Node Exporter processes
echo -e "${YELLOW}Finding and terminating Node Exporter processes...${NC}"
NODE_PIDS=$(pgrep -f node_exporter || true)
if [ -n "$NODE_PIDS" ]; then
    echo -e "Found Node Exporter PIDs: $NODE_PIDS"
    for pid in $NODE_PIDS; do
        sudo kill -15 $pid 2>/dev/null || true
    done
    sleep 2
    # Force kill any remaining
    NODE_PIDS=$(pgrep -f node_exporter || true)
    if [ -n "$NODE_PIDS" ]; then
        echo -e "${YELLOW}Force killing remaining Node Exporter processes...${NC}"
        for pid in $NODE_PIDS; do
            sudo kill -9 $pid 2>/dev/null || true
        done
    fi
else
    echo -e "${GREEN}No Node Exporter processes found.${NC}"
fi

# 4. Check the ports and free them up
echo -e "${YELLOW}Freeing up ports 9090 and 9100...${NC}"
sudo fuser -k 9090/tcp 2>/dev/null || true
sudo fuser -k 9100/tcp 2>/dev/null || true

# 5. Verify all processes are stopped
sleep 1
PROM_PIDS=$(pgrep -f prometheus || true)
NODE_PIDS=$(pgrep -f node_exporter || true)

if [ -n "$PROM_PIDS" ] || [ -n "$NODE_PIDS" ]; then
    echo -e "${RED}Some monitoring processes are still running!${NC}"
    echo -e "Prometheus PIDs: ${PROM_PIDS:-None}"
    echo -e "Node Exporter PIDs: ${NODE_PIDS:-None}"
    
    # Final attempt - use the killall command
    echo -e "${YELLOW}Attempting final termination with killall...${NC}"
    sudo killall -9 prometheus 2>/dev/null || true
    sudo killall -9 node_exporter 2>/dev/null || true
    
    # Verify again
    sleep 1
    PROM_PIDS=$(pgrep -f prometheus || true)
    NODE_PIDS=$(pgrep -f node_exporter || true)
    
    if [ -n "$PROM_PIDS" ] || [ -n "$NODE_PIDS" ]; then
        echo -e "${RED}Failed to terminate all monitoring processes.${NC}"
        exit 1
    else
        echo -e "${GREEN}Successfully terminated all monitoring processes.${NC}"
    fi
else
    echo -e "${GREEN}All monitoring processes have been successfully terminated.${NC}"
fi

# 6. Remove PID files if they exist
rm -f /opt/sutazaiapp/.prometheus.pid /opt/sutazaiapp/.node_exporter.pid 2>/dev/null || true

echo -e "${GREEN}Cleanup complete!${NC}" 