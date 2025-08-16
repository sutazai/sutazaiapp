#!/bin/bash
# Build all Docker images with Rule 11 compliance
# Secure, pinned versions, non-root users, health checks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Rule 11 compliant Docker images...${NC}"

# Create network if it doesn't exist
if ! docker network ls | grep -q sutazai-network; then
    echo -e "${YELLOW}Creating sutazai-network...${NC}"
    docker network create sutazai-network
fi

# Build base images first
echo -e "${GREEN}Building base images...${NC}"

# Build Python base
echo "Building Python base image..."
docker build -f docker/base/Dockerfile.python-base-secure -t sutazai-python-base:v1.0.0 docker/base/

# Build backend
echo -e "${GREEN}Building backend image...${NC}"
docker build -f backend/Dockerfile -t sutazaiapp-backend:v1.0.0 backend/

# Build frontend
echo -e "${GREEN}Building frontend image...${NC}"
docker build -f docker/frontend/Dockerfile -t sutazaiapp-frontend:v1.0.0 frontend/

# Build FAISS
echo -e "${GREEN}Building FAISS image...${NC}"
docker build -f docker/faiss/Dockerfile -t sutazaiapp-faiss:v1.0.0 docker/faiss/

# Build agent images
echo -e "${GREEN}Building agent images...${NC}"

AGENT_DIRS=(
    "hardware-resource-optimizer"
    "ai-agent-orchestrator"
    "jarvis-automation-agent"
    "jarvis-hardware-resource-optimizer"
    "ollama_integration"
    "resource_arbitration_agent"
    "task_assignment_coordinator"
)

for agent in "${AGENT_DIRS[@]}"; do
    if [ -d "docker/agents/$agent" ] && [ -f "docker/agents/$agent/Dockerfile" ]; then
        echo "Building $agent..."
        docker build -f "docker/agents/$agent/Dockerfile" -t "sutazaiapp-$agent:v1.0.0" "agents/$agent/" || echo -e "${YELLOW}Warning: Failed to build $agent${NC}"
    fi
done

# Verify images
echo -e "${GREEN}Verifying built images...${NC}"
docker images | grep sutazaiapp

# Security scan with trivy if available
if command -v trivy &> /dev/null; then
    echo -e "${GREEN}Running security scans...${NC}"
    for image in $(docker images --format "{{.Repository}}:{{.Tag}}" | grep sutazaiapp); do
        echo "Scanning $image..."
        trivy image --severity HIGH,CRITICAL "$image" || true
    done
else
    echo -e "${YELLOW}Trivy not installed. Skipping security scans.${NC}"
fi

echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}To deploy with secure configuration, run:${NC}"
echo -e "  docker-compose -f docker-compose.secure.yml up -d"