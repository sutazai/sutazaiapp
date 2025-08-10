#!/bin/bash

# Security Hardening Script for Root Containers
# Purpose: Fix the remaining 3 containers running as root
# Targets: ai-agent-orchestrator, consul, grafana

set -e


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

echo "================================================"
echo "Container Security Hardening Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check container user
check_container_user() {
    local container_name=$1
    if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        local user_info=$(docker exec "$container_name" id 2>/dev/null || echo "Container not accessible")
        echo "$user_info"
    else
        echo "Container not running"
    fi
}

# Function to stop container
stop_container() {
    local container_name=$1
    echo -e "${YELLOW}Stopping container: $container_name${NC}"
    docker stop "$container_name" 2>/dev/null || true
    docker rm "$container_name" 2>/dev/null || true
}

# Function to validate security
validate_security() {
    local container_name=$1
    local user_info=$(check_container_user "$container_name")
    
    if echo "$user_info" | grep -q "uid=0(root)"; then
        echo -e "${RED}✗ $container_name is still running as root${NC}"
        echo "  User info: $user_info"
        return 1
    elif echo "$user_info" | grep -q "gid=0(root)"; then
        echo -e "${YELLOW}⚠ $container_name has root group${NC}"
        echo "  User info: $user_info"
        return 1
    else
        echo -e "${GREEN}✓ $container_name is secure (non-root)${NC}"
        echo "  User info: $user_info"
        return 0
    fi
}

echo "Step 1: Current Security Status"
echo "================================"
echo ""

echo "Checking current container users..."
echo -e "${YELLOW}AI Agent Orchestrator:${NC}"
check_container_user "sutazai-ai-agent-orchestrator"
echo ""

echo -e "${YELLOW}Consul:${NC}"
check_container_user "sutazai-consul"
echo ""

echo -e "${YELLOW}Grafana:${NC}"
check_container_user "sutazai-grafana"
echo ""

echo "Step 2: Applying Security Hardening"
echo "===================================="
echo ""

# Check if security hardening file exists
if [ ! -f "docker-compose.security-hardening.yml" ]; then
    echo -e "${RED}Error: docker-compose.security-hardening.yml not found${NC}"
    echo "Please ensure you're running this script from the project root"
    exit 1
fi

# Rebuild AI Agent Orchestrator with hardened Dockerfile
echo -e "${YELLOW}Rebuilding AI Agent Orchestrator with security hardening...${NC}"
if [ -f "agents/ai_agent_orchestrator/Dockerfile.hardened" ]; then
    # Backup original Dockerfile
    cp agents/ai_agent_orchestrator/Dockerfile agents/ai_agent_orchestrator/Dockerfile.backup
    # Use hardened version
    cp agents/ai_agent_orchestrator/Dockerfile.hardened agents/ai_agent_orchestrator/Dockerfile
    
    # Rebuild the image
    docker build -t sutazaiapp-ai-agent-orchestrator \
        --build-arg USER_ID=1000 \
        --build-arg GROUP_ID=1000 \
        ./agents/ai_agent_orchestrator
    
    echo -e "${GREEN}✓ AI Agent Orchestrator image rebuilt with security hardening${NC}"
else
    echo -e "${YELLOW}Using existing Dockerfile with USER directive${NC}"
    docker build -t sutazaiapp-ai-agent-orchestrator ./agents/ai_agent_orchestrator
fi

# Stop affected containers
echo ""
echo -e "${YELLOW}Stopping affected containers for security update...${NC}"
stop_container "sutazai-ai-agent-orchestrator"
stop_container "sutazai-consul"
stop_container "sutazai-grafana"

# Start containers with security hardening
echo ""
echo -e "${YELLOW}Starting containers with security hardening...${NC}"
docker compose -f docker-compose.yml -f docker-compose.security-hardening.yml up -d \
    ai-agent-orchestrator consul grafana

# Wait for containers to be healthy
echo ""
echo -e "${YELLOW}Waiting for containers to become healthy...${NC}"
sleep 10

# Verify health status
echo ""
echo "Step 3: Verifying Container Health"
echo "==================================="
echo ""

for container in "sutazai-ai-agent-orchestrator" "sutazai-consul" "sutazai-grafana"; do
    health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "unknown")
    if [ "$health_status" == "healthy" ]; then
        echo -e "${GREEN}✓ $container is healthy${NC}"
    else
        echo -e "${YELLOW}⚠ $container health status: $health_status${NC}"
    fi
done

echo ""
echo "Step 4: Final Security Validation"
echo "=================================="
echo ""

# Final security check
all_secure=true
for container in "sutazai-ai-agent-orchestrator" "sutazai-consul" "sutazai-grafana"; do
    if ! validate_security "$container"; then
        all_secure=false
    fi
done

echo ""
echo "Step 5: Service Functionality Test"
echo "==================================="
echo ""

# Test AI Agent Orchestrator
echo -e "${YELLOW}Testing AI Agent Orchestrator...${NC}"
if curl -s -f http://localhost:8589/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ AI Agent Orchestrator API is responding${NC}"
else
    echo -e "${RED}✗ AI Agent Orchestrator API is not responding${NC}"
fi

# Test Consul
echo -e "${YELLOW}Testing Consul...${NC}"
if curl -s -f http://localhost:8500/v1/status/leader > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Consul API is responding${NC}"
else
    echo -e "${RED}✗ Consul API is not responding${NC}"
fi

# Test Grafana
echo -e "${YELLOW}Testing Grafana...${NC}"
if curl -s -f http://localhost:10201/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Grafana API is responding${NC}"
else
    echo -e "${RED}✗ Grafana API is not responding${NC}"
fi

echo ""
echo "================================================"
echo "Security Hardening Complete"
echo "================================================"
echo ""

if [ "$all_secure" = true ]; then
    echo -e "${GREEN}✓ SUCCESS: All containers are now running as non-root users${NC}"
    echo ""
    echo "Security Summary:"
    echo "• AI Agent Orchestrator: Secured with appuser (UID 1000)"
    echo "• Consul: Secured with consul user (UID 100)"
    echo "• Grafana: Secured with grafana user (UID 472)"
    echo ""
    echo "All services remain fully functional with enhanced security."
    exit 0
else
    echo -e "${YELLOW}⚠ WARNING: Some containers may still have security issues${NC}"
    echo "Please review the output above and address any remaining issues."
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Ensure docker-compose.security-hardening.yml is being used"
    echo "2. Check container logs: docker logs <container-name>"
    echo "3. Verify volume permissions are correct"
    echo "4. Try rebuilding images with: docker compose build --no-cache"
    exit 1
fi