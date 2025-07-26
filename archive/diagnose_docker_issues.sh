#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== SutazAI Docker Diagnostics ===${NC}"
echo

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "ok" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "warn" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

# Check Docker daemon
echo -e "${BLUE}Checking Docker daemon...${NC}"
if docker info >/dev/null 2>&1; then
    print_status "ok" "Docker daemon is running"
else
    print_status "error" "Docker daemon is not running"
    echo "  Please start Docker with: sudo systemctl start docker"
    exit 1
fi

# Check Docker Compose
echo -e "\n${BLUE}Checking Docker Compose...${NC}"
if command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1; then
    print_status "ok" "Docker Compose is installed"
else
    print_status "error" "Docker Compose is not installed"
    exit 1
fi

# Check disk space
echo -e "\n${BLUE}Checking disk space...${NC}"
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 90 ]; then
    print_status "ok" "Disk usage is at ${DISK_USAGE}%"
else
    print_status "warn" "Disk usage is high: ${DISK_USAGE}%"
fi

# Check memory
echo -e "\n${BLUE}Checking memory...${NC}"
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
if [ -z "$AVAILABLE_MEM" ]; then
    AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $4}')
fi
print_status "ok" "Total memory: ${TOTAL_MEM}GB, Available: ${AVAILABLE_MEM}GB"

# Check for duplicate service definitions
echo -e "\n${BLUE}Checking docker-compose.yml for issues...${NC}"
if [ -f "docker-compose.yml" ]; then
    # Check for duplicate service definitions
    SERVICES=$(grep -E "^[[:space:]]*[a-zA-Z0-9_-]+:" docker-compose.yml | sed 's/://g' | sed 's/^[[:space:]]*//' | sort)
    DUPLICATES=$(echo "$SERVICES" | uniq -d)
    
    if [ -n "$DUPLICATES" ]; then
        print_status "error" "Found duplicate service definitions in docker-compose.yml:"
        echo "$DUPLICATES" | while read service; do
            echo "    - $service"
        done
        echo -e "\n  ${YELLOW}This is likely causing container restart issues!${NC}"
        echo -e "  ${GREEN}A fixed version has been created: docker-compose-fixed.yml${NC}"
    else
        print_status "ok" "No duplicate service definitions found"
    fi
else
    print_status "warn" "docker-compose.yml not found"
fi

# Check for running containers
echo -e "\n${BLUE}Checking for running containers...${NC}"
RUNNING_CONTAINERS=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep sutazai || echo "")
if [ -n "$RUNNING_CONTAINERS" ]; then
    print_status "ok" "Found running SutazAI containers:"
    echo "$RUNNING_CONTAINERS" | sed 's/^/    /'
else
    print_status "warn" "No SutazAI containers are currently running"
fi

# Check for containers in restart loop
echo -e "\n${BLUE}Checking for containers in restart loop...${NC}"
RESTARTING=$(docker ps -a --filter "name=sutazai" --format "{{.Names}} {{.Status}}" | grep -i "restarting" || echo "")
if [ -n "$RESTARTING" ]; then
    print_status "error" "Found containers in restart loop:"
    echo "$RESTARTING" | sed 's/^/    /'
    echo -e "\n  ${YELLOW}Checking container logs for errors...${NC}"
    
    # Get logs from restarting containers
    echo "$RESTARTING" | awk '{print $1}' | while read container; do
        echo -e "\n  ${BLUE}Last 10 lines from $container:${NC}"
        docker logs --tail 10 "$container" 2>&1 | sed 's/^/    /'
    done
else
    print_status "ok" "No containers in restart loop"
fi

# Check for port conflicts
echo -e "\n${BLUE}Checking for port conflicts...${NC}"
PORTS=(5432 6379 8000 8001 6333 11434 8501 9090 3000)
CONFLICTS=0
for port in "${PORTS[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        PROCESS=$(lsof -Pi :$port -sTCP:LISTEN 2>/dev/null | grep -v "^COMMAND" | head -1)
        if [[ ! "$PROCESS" =~ "docker" ]]; then
            print_status "error" "Port $port is already in use by non-Docker process"
            CONFLICTS=$((CONFLICTS + 1))
        fi
    fi
done
if [ $CONFLICTS -eq 0 ]; then
    print_status "ok" "No port conflicts detected"
fi

# Check Docker logs for errors
echo -e "\n${BLUE}Checking Docker daemon logs for recent errors...${NC}"
if command -v journalctl >/dev/null 2>&1; then
    DOCKER_ERRORS=$(journalctl -u docker --since "1 hour ago" --no-pager 2>/dev/null | grep -i error | tail -5)
    if [ -n "$DOCKER_ERRORS" ]; then
        print_status "warn" "Found recent Docker daemon errors:"
        echo "$DOCKER_ERRORS" | sed 's/^/    /'
    else
        print_status "ok" "No recent Docker daemon errors"
    fi
fi

# Recommendations
echo -e "\n${BLUE}=== Recommendations ===${NC}"
echo

if [ -f "docker-compose.yml" ] && [ -n "$DUPLICATES" ]; then
    echo -e "${YELLOW}1. Your docker-compose.yml has duplicate service definitions.${NC}"
    echo "   This is causing containers to restart. To fix:"
    echo -e "   ${GREEN}cp docker-compose-fixed.yml docker-compose.yml${NC}"
    echo -e "   ${GREEN}docker-compose down${NC}"
    echo -e "   ${GREEN}docker-compose up -d${NC}"
    echo
fi

echo -e "${YELLOW}2. To clean up and start fresh:${NC}"
echo "   docker-compose down -v          # Stop and remove volumes"
echo "   docker system prune -a          # Clean up unused resources"
echo "   docker-compose up -d            # Start services"
echo

echo -e "${YELLOW}3. To monitor container health:${NC}"
echo "   docker-compose ps               # Check service status"
echo "   docker-compose logs -f          # Follow logs"
echo "   docker stats                    # Monitor resource usage"
echo

echo -e "${BLUE}=== Diagnostics Complete ===${NC}"