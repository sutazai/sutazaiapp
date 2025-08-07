#!/bin/bash
#
# stop_monitoring.sh - Script to stop Prometheus and Node Exporter monitoring services
#

# Define color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-/opt/sutazaiapp}"
PROMETHEUS_PID_FILE="$PROJECT_ROOT/.prometheus.pid"
NODE_EXPORTER_PID_FILE="$PROJECT_ROOT/.node_exporter.pid"
PROMETHEUS_DATA_DIR="$PROJECT_ROOT/data/prometheus"

# Function to stop a process by PID file
stop_process() {
    local pid_file=$1
    local service_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping $service_name (PID: $pid)...${NC}"
            
            # Try graceful shutdown first
            kill -15 $pid 2>/dev/null || sudo kill -15 $pid 2>/dev/null
            
            # Wait for process to terminate
            local count=0
            while ps -p $pid > /dev/null 2>&1 && [ $count -lt 5 ]; do
                sleep 1
                ((count++))
            done
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${YELLOW}Process did not terminate gracefully. Forcing...${NC}"
                kill -9 $pid 2>/dev/null || sudo kill -9 $pid 2>/dev/null
                sleep 1
            fi
            
            if ! ps -p $pid > /dev/null 2>&1; then
                echo -e "${GREEN}$service_name stopped successfully.${NC}"
            else
                echo -e "${RED}Failed to stop $service_name.${NC}"
            fi
        else
            echo -e "${BLUE}$service_name (PID: $pid) is not running.${NC}"
        fi
        
        # Remove PID file
        rm -f "$pid_file"
    else
        echo -e "${BLUE}$service_name is not running (no PID file).${NC}"
    fi
}

# Check for Docker containers
check_docker_containers() {
    if command -v docker &> /dev/null; then
        echo -e "${BLUE}Checking for Docker containers...${NC}"
        
        # Check for Prometheus container
        if docker ps 2>/dev/null | grep -q prometheus; then
            echo -e "${YELLOW}Found Prometheus running in Docker. Stopping...${NC}"
            docker stop prometheus >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Prometheus Docker container stopped successfully.${NC}"
                # Remove container to clean up properly
                docker rm prometheus >/dev/null 2>&1 || true
            else
                echo -e "${RED}Failed to stop Prometheus Docker container.${NC}"
            fi
        fi
        
        # Check for Node Exporter container
        if docker ps 2>/dev/null | grep -q node_exporter; then
            echo -e "${YELLOW}Found Node Exporter running in Docker. Stopping...${NC}"
            docker stop node_exporter >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Node Exporter Docker container stopped successfully.${NC}"
                # Remove container to clean up properly
                docker rm node_exporter >/dev/null 2>&1 || true
            else
                echo -e "${RED}Failed to stop Node Exporter Docker container.${NC}"
            fi
        fi
        
        # Check for Docker network
        if docker network ls 2>/dev/null | grep -q "sutazai-monitoring"; then
            echo -e "${YELLOW}Found sutazai-monitoring Docker network. Checking if it can be removed...${NC}"
            
            # Check if any containers are using this network
            CONTAINERS_USING_NETWORK=$(docker network inspect sutazai-monitoring 2>/dev/null | grep -o '"Name": "[^"]*"' | grep -v "sutazai-monitoring" || true)
            
            if [ -z "$CONTAINERS_USING_NETWORK" ]; then
                echo -e "${YELLOW}Removing sutazai-monitoring Docker network...${NC}"
                docker network rm sutazai-monitoring >/dev/null 2>&1
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}Docker network removed successfully.${NC}"
                else
                    echo -e "${RED}Failed to remove Docker network.${NC}"
                fi
            else
                echo -e "${YELLOW}Network still in use by other containers. Skipping removal.${NC}"
            fi
        fi
    fi
}

# Main execution
echo -e "${BLUE}Stopping monitoring services...${NC}"

# First check for Docker containers
check_docker_containers

# Stop Prometheus
stop_process "$PROMETHEUS_PID_FILE" "Prometheus"

# Stop Node Exporter
stop_process "$NODE_EXPORTER_PID_FILE" "Node Exporter"

# Check for any remaining processes
PROM_PIDS=$(pgrep -f prometheus | grep -v "grep" | grep -v "stop_monitoring.sh" || true)
NODE_PIDS=$(pgrep -f node_exporter | grep -v "grep" | grep -v "stop_monitoring.sh" || true)

if [ -n "$PROM_PIDS" ] || [ -n "$NODE_PIDS" ]; then
    echo -e "${YELLOW}Found lingering monitoring processes. Force killing...${NC}"
    # Force kill any remaining Prometheus processes
    if [ -n "$PROM_PIDS" ]; then
        echo -e "Prometheus PIDs: $PROM_PIDS"
        for pid in $PROM_PIDS; do
            sudo kill -9 $pid 2>/dev/null || true
        done
    fi
    
    # Force kill any remaining Node Exporter processes
    if [ -n "$NODE_PIDS" ]; then
        echo -e "Node Exporter PIDs: $NODE_PIDS"
        for pid in $NODE_PIDS; do
            sudo kill -9 $pid 2>/dev/null || true
        done
    fi
    
    # Free up ports
    echo -e "${YELLOW}Freeing ports 9090 and 9100...${NC}"
    sudo fuser -k 9090/tcp 2>/dev/null || true
    sudo fuser -k 9100/tcp 2>/dev/null || true
fi

# Fix permissions on Prometheus data directory if it exists
if [ -d "$PROMETHEUS_DATA_DIR" ]; then
    echo -e "${YELLOW}Setting correct permissions for Prometheus data directory...${NC}"
    sudo chmod -R 777 "$PROMETHEUS_DATA_DIR" 2>/dev/null || true
fi

echo -e "${GREEN}All monitoring services stopped.${NC}" 