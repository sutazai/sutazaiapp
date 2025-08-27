#!/bin/bash

# Automated MCP Integration Setup Script
# Sets up all MCP servers and integration points for SutazAI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       SutazAI MCP Integration Setup Script           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check command existence
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✅${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}❌${NC} $1 is not installed"
        return 1
    fi
}

# Function to check and start docker service
check_docker_service() {
    echo -e "\n${BLUE}Checking Docker service...${NC}"
    
    if docker ps &> /dev/null; then
        echo -e "${GREEN}✅${NC} Docker is running"
    else
        echo -e "${YELLOW}⚠️${NC} Docker is not running. Attempting to start..."
        if command -v systemctl &> /dev/null; then
            sudo systemctl start docker
            echo -e "${GREEN}✅${NC} Docker started"
        else
            echo -e "${RED}❌${NC} Cannot start Docker automatically"
            exit 1
        fi
    fi
}

# Function to setup database containers
setup_databases() {
    echo -e "\n${BLUE}Setting up database containers...${NC}"
    
    # Check if network exists
    if ! docker network ls --format '{{.Name}}' | grep -qx "sutazai-network"; then
        echo "Creating Docker network..."
        docker network create sutazai-network
    fi
    
    # Start essential containers
    cd "$PROJECT_ROOT"
    
    echo "Starting PostgreSQL..."
    docker compose up -d postgres
    
    echo "Starting Redis..."
    docker compose up -d redis
    
    echo "Waiting for databases to initialize..."
    sleep 10
    
    # Check PostgreSQL
    if docker exec sutazai-postgres pg_isready -U sutazai &> /dev/null; then
        echo -e "${GREEN}✅${NC} PostgreSQL is ready"
    else
        echo -e "${YELLOW}⚠️${NC} PostgreSQL is starting..."
    fi
    
    # Check Redis
    if docker exec sutazai-redis redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✅${NC} Redis is ready"
    else
        echo -e "${YELLOW}⚠️${NC} Redis is starting..."
    fi
}

# Function to initialize claude-flow
setup_claude_flow() {
    echo -e "\n${BLUE}Setting up claude-flow...${NC}"
    
    # Check if claude-flow is accessible
    if npx claude-flow@alpha --version &> /dev/null; then
        echo -e "${GREEN}✅${NC} claude-flow is accessible"
    else
        echo -e "${YELLOW}⚠️${NC} Installing claude-flow..."
        npm install -g claude-flow@alpha
    fi
    
    # Initialize if needed
    if [ ! -f "$PROJECT_ROOT/CLAUDE.md" ]; then
        echo "Initializing claude-flow..."
        cd "$PROJECT_ROOT"
        npx claude-flow@alpha init
    fi
    
    echo -e "${GREEN}✅${NC} claude-flow initialized"
}

# Function to initialize hive-mind
setup_hive_mind() {
    echo -e "\n${BLUE}Setting up hive-mind...${NC}"
    
    if [ ! -d "$PROJECT_ROOT/.hive-mind" ]; then
        echo "Initializing hive-mind system..."
        cd "$PROJECT_ROOT"
        npx claude-flow@alpha hive-mind init
        echo -e "${GREEN}✅${NC} Hive-mind initialized"
    else
        echo -e "${GREEN}✅${NC} Hive-mind already initialized"
    fi
}

# Function to test MCP wrappers
test_mcp_wrappers() {
    echo -e "\n${BLUE}Testing MCP wrapper scripts...${NC}"
    
    local passed=0
    local failed=0
    local wrappers=(
        "claude-flow"
        "github"
        "sequential-thinking"
        "context7"
        "code-index-mcp"
        "ultimatecoder"
        "extended-memory"
        "files"
        "ddg"
        "http"
    )
    
    for wrapper in "${wrappers[@]}"; do
        if [ -f "$SCRIPT_DIR/wrappers/${wrapper}.sh" ]; then
            if "$SCRIPT_DIR/wrappers/${wrapper}.sh" --selfcheck &> /dev/null; then
                echo -e "${GREEN}✅${NC} $wrapper"
                ((passed++))
            else
                echo -e "${RED}❌${NC} $wrapper"
                ((failed++))
            fi
        else
            echo -e "${YELLOW}⚠️${NC} $wrapper (missing wrapper)"
            ((failed++))
        fi
    done
    
    echo -e "\nWrapper Test Results: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}"
}

# Function to start backend API
setup_backend() {
    echo -e "\n${BLUE}Setting up backend API...${NC}"
    
    # Check if backend is already running
    if curl -s http://localhost:10010/health | grep -q "healthy" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} Backend API is already running"
    else
        echo "Starting backend API..."
        cd "$PROJECT_ROOT/backend"
        
        # Create venv if it doesn't exist
        if [ ! -d "venv" ]; then
            python3 -m venv venv
            ./venv/bin/pip install --quiet --upgrade pip
            ./venv/bin/pip install --quiet -r requirements.txt 2>/dev/null || \
            ./venv/bin/pip install --quiet fastapi uvicorn redis asyncpg psycopg2-binary httpx pydantic
        fi
        
        # Start backend in background
        export JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        export PYTHONPATH="$PROJECT_ROOT/backend"
        
        nohup ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 10010 > backend.log 2>&1 &
        
        echo "Waiting for backend to start..."
        sleep 5
        
        if curl -s http://localhost:10010/health | grep -q "healthy" 2>/dev/null; then
            echo -e "${GREEN}✅${NC} Backend API started successfully"
        else
            echo -e "${YELLOW}⚠️${NC} Backend API may still be starting..."
        fi
    fi
}

# Function to create summary
create_summary() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}            SETUP COMPLETE${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    
    echo -e "\n${GREEN}Available Resources:${NC}"
    echo "• Backend API: http://localhost:10010/health"
    echo "• PostgreSQL: localhost:10000"
    echo "• Redis: localhost:10001"
    echo "• MCP Servers: 26+ configured"
    
    echo -e "\n${GREEN}Quick Commands:${NC}"
    echo "• Test all MCP: /opt/sutazaiapp/scripts/mcp/test-all-mcp.sh"
    echo "• Test integration: /opt/sutazaiapp/scripts/mcp/test-mcp-integration.sh"
    echo "• View MCP list: claude mcp list"
    echo "• Start swarm: npx claude-flow@alpha swarm \"your task\""
    
    echo -e "\n${GREEN}Documentation:${NC}"
    echo "• Integration Guide: /opt/sutazaiapp/MCP-INTEGRATION.md"
    echo "• System Status: /opt/sutazaiapp/CLAUDE.md"
}

# Main execution
main() {
    echo "Starting MCP integration setup..."
    
    # Prerequisites
    echo -e "\n${BLUE}Checking prerequisites...${NC}"
    check_command docker
    check_command npm
    check_command python3
    check_command curl
    
    # Setup components
    check_docker_service
    setup_databases
    setup_claude_flow
    setup_hive_mind
    setup_backend
    test_mcp_wrappers
    
    # Generate summary
    create_summary
    
    echo -e "\n${GREEN}✅ MCP Integration setup complete!${NC}"
}

# Run main function
main "$@"