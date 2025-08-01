#\!/bin/bash
# Test deployment script - verify everything works

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧪 SutazAI Deployment Test Script${NC}"
echo "========================================"

# 1. Check Docker
echo -e "\n${BLUE}1. Testing Docker...${NC}"
if docker version >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker is running${NC}"
    docker version --format "   Server: {{.Server.Version}}"
else
    echo -e "${RED}❌ Docker is not running${NC}"
    exit 1
fi

# 2. Check Docker Compose
echo -e "\n${BLUE}2. Testing Docker Compose...${NC}"
if docker compose version >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker Compose v2 is available${NC}"
    docker compose version
else
    echo -e "${YELLOW}⚠️  Docker Compose v2 not found${NC}"
fi

# 3. Check environment files
echo -e "\n${BLUE}3. Checking environment files...${NC}"
required_files=(".env" "docker-compose.yml" "docker-compose-agents-complete.yml")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ Found: $file${NC}"
    else
        echo -e "${RED}❌ Missing: $file${NC}"
    fi
done

# 4. Test deployment script syntax
echo -e "\n${BLUE}4. Testing deployment script syntax...${NC}"
if bash -n scripts/deploy_complete_system.sh 2>/dev/null; then
    echo -e "${GREEN}✅ Deployment script syntax is valid${NC}"
else
    echo -e "${RED}❌ Deployment script has syntax errors${NC}"
    bash -n scripts/deploy_complete_system.sh
fi

# 5. Check disk space
echo -e "\n${BLUE}5. Checking disk space...${NC}"
available_gb=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$available_gb" -gt 50 ]; then
    echo -e "${GREEN}✅ Sufficient disk space: ${available_gb}GB available${NC}"
else
    echo -e "${YELLOW}⚠️  Low disk space: ${available_gb}GB available${NC}"
fi

# 6. Test a simple container
echo -e "\n${BLUE}6. Testing Docker with a simple container...${NC}"
if docker run --rm hello-world >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker can run containers successfully${NC}"
else
    echo -e "${RED}❌ Docker cannot run containers${NC}"
fi

echo -e "\n${GREEN}✅ All tests completed\!${NC}"
echo -e "${BLUE}You can now run: sudo ./scripts/deploy_complete_system.sh${NC}"
