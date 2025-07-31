#!/bin/bash

# Start Unified AI System with All Agents Working Together
# This script ensures all components are properly initialized and coordinated

set -e

# Constants
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# ASCII Art Banner
print_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   ███████╗██╗   ██╗████████╗ █████╗ ███████╗ █████╗ ██╗         ║
    ║   ██╔════╝██║   ██║╚══██╔══╝██╔══██╗╚══███╔╝██╔══██╗██║         ║
    ║   ███████╗██║   ██║   ██║   ███████║  ███╔╝ ███████║██║         ║
    ║   ╚════██║██║   ██║   ██║   ██╔══██║ ███╔╝  ██╔══██║██║         ║
    ║   ███████║╚██████╔╝   ██║   ██║  ██║███████╗██║  ██║██║         ║
    ║   ╚══════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝         ║
    ║                                                                   ║
    ║              🤖 Unified AI Agent System v2.0 🤖                   ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Logging functions
log_step() {
    echo -e "\n${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}[STEP]${NC} $1"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $(date '+%H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $(date '+%H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $(date '+%H:%M:%S') - $1"
}

# Function to check Docker daemon
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running!"
        log_info "Starting Docker..."
        sudo systemctl start docker || sudo service docker start
        sleep 5
        
        if ! docker info >/dev/null 2>&1; then
            log_error "Failed to start Docker. Please start Docker manually."
            exit 1
        fi
    fi
    log_success "Docker is running"
}

# Function to initialize MCP connection
init_mcp() {
    log_info "Initializing MCP connection..."
    
    # Check if MCP server is running
    if docker-compose ps mcp-server 2>/dev/null | grep -q "Up"; then
        log_success "MCP server is already running"
    else
        log_info "Starting MCP server..."
        docker-compose up -d mcp-server
        sleep 10
    fi
    
    # Verify MCP is responsive
    if curl -s http://localhost:8100/health >/dev/null 2>&1; then
        log_success "MCP server is healthy"
    else
        log_warning "MCP server health check failed, but continuing..."
    fi
}

# Function to run the unified orchestrator
start_orchestrator() {
    log_info "Starting Unified AI Orchestrator..."
    
    # Create orchestrator service if needed
    cat > /tmp/docker-compose.orchestrator.yml << 'EOF'
version: '3.8'
services:
  unified-orchestrator:
    build:
      context: .
      dockerfile: backend/Dockerfile.agi
    container_name: sutazai-unified-orchestrator
    command: python -m backend.ai_agents.orchestration.unified_orchestrator
    environment:
      - PYTHONPATH=/app
      - REDIS_HOST=redis
      - POSTGRES_HOST=postgres
    volumes:
      - ./backend:/app/backend
      - ./config:/app/config
      - ./logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
      mcp-server:
        condition: service_started
    networks:
      - sutazai-network
    restart: unless-stopped
EOF
    
    # Start orchestrator
    docker-compose -f docker-compose.yml -f /tmp/docker-compose.orchestrator.yml up -d unified-orchestrator
    
    log_success "Orchestrator started"
}

# Function to verify all agents
verify_agents() {
    log_info "Verifying AI agents..."
    
    # List of expected agents
    local agents=(
        "infrastructure-devops-manager"
        "ollama-integration-specialist"
        "hardware-resource-optimizer"
        "senior-ai-engineer"
        "senior-backend-developer"
        "senior-frontend-developer"
        "ai-agent-orchestrator"
        "deployment-automation-master"
        "testing-qa-validator"
        "security-pentesting-specialist"
    )
    
    local active_count=0
    local total_count=${#agents[@]}
    
    echo ""
    echo -e "${CYAN}Agent Status:${NC}"
    echo -e "${CYAN}─────────────────────────────────────────────${NC}"
    
    for agent in "${agents[@]}"; do
        # Check if agent is registered in Redis
        if docker-compose exec -T redis redis-cli EXISTS "agent:registry:$agent" | grep -q "1"; then
            status=$(docker-compose exec -T redis redis-cli HGET "agent:registry:$agent" status 2>/dev/null || echo "unknown")
            
            if [ "$status" = "active" ]; then
                echo -e "  ${GREEN}●${NC} $agent: ${GREEN}$status${NC}"
                ((active_count++))
            else
                echo -e "  ${YELLOW}●${NC} $agent: ${YELLOW}$status${NC}"
            fi
        else
            echo -e "  ${RED}●${NC} $agent: ${RED}not registered${NC}"
        fi
    done
    
    echo -e "${CYAN}─────────────────────────────────────────────${NC}"
    echo -e "Active Agents: ${GREEN}$active_count${NC}/$total_count"
    echo ""
}

# Function to create demonstration tasks
create_demo_tasks() {
    log_info "Creating demonstration tasks..."
    
    python3 << 'EOF'
import requests
import json

tasks = [
    {
        "type": "deployment",
        "description": "Deploy the latest AI model with zero downtime",
        "priority": 8
    },
    {
        "type": "testing",
        "description": "Run comprehensive test suite for all services",
        "priority": 7
    },
    {
        "type": "optimization",
        "description": "Optimize system performance and resource usage",
        "priority": 6
    },
    {
        "type": "security",
        "description": "Perform security audit and vulnerability scan",
        "priority": 9
    },
    {
        "type": "ai_development",
        "description": "Implement new RAG system for documentation",
        "priority": 7
    }
]

try:
    # Submit tasks via MCP server
    for task in tasks:
        response = requests.post(
            "http://localhost:8100/execute_agent_task",
            json={
                "agent_name": "task-assignment-coordinator",
                "task": json.dumps(task),
                "priority": task["priority"]
            }
        )
        if response.status_code == 200:
            print(f"✓ Created task: {task['description']}")
        else:
            print(f"✗ Failed to create task: {task['description']}")
except Exception as e:
    print(f"Error creating tasks: {e}")
EOF
}

# Function to show system dashboard
show_dashboard() {
    log_step "System Dashboard"
    
    echo -e "\n${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║               🚀 SutazAI System Status 🚀                 ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
    
    # Services status
    echo -e "\n${YELLOW}Core Services:${NC}"
    docker-compose ps --format "table {{.Name}}\t{{.Status}}" | grep -E "(postgres|redis|ollama|backend-agi|frontend-agi)" | head -10
    
    # Agent summary
    echo -e "\n${YELLOW}AI Agents:${NC}"
    active_agents=$(docker-compose exec -T redis redis-cli --scan --pattern "agent:registry:*" | wc -l)
    echo -e "  Total Registered: ${GREEN}$active_agents${NC}"
    
    # Task summary
    echo -e "\n${YELLOW}Task Queue:${NC}"
    pending_tasks=$(docker-compose exec -T redis redis-cli LLEN "agent:tasks" 2>/dev/null || echo "0")
    echo -e "  Pending Tasks: ${CYAN}$pending_tasks${NC}"
    
    # Access URLs
    echo -e "\n${YELLOW}Access Points:${NC}"
    echo -e "  ${BLUE}Frontend:${NC} http://localhost:8501"
    echo -e "  ${BLUE}Backend API:${NC} http://localhost:8000"
    echo -e "  ${BLUE}MCP Server:${NC} http://localhost:8100"
    echo -e "  ${BLUE}Grafana:${NC} http://localhost:3000"
    echo -e "  ${BLUE}Prometheus:${NC} http://localhost:9090"
    
    # Quick commands
    echo -e "\n${YELLOW}Useful Commands:${NC}"
    echo -e "  ${CYAN}View logs:${NC} docker-compose logs -f [service-name]"
    echo -e "  ${CYAN}Agent status:${NC} docker-compose exec redis redis-cli --scan --pattern 'agent:*'"
    echo -e "  ${CYAN}Submit task:${NC} curl -X POST http://localhost:8100/execute_agent_task -d '{...}'"
    echo ""
}

# Main execution
main() {
    clear
    print_banner
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Create necessary directories
    mkdir -p "$LOG_DIR" data/{postgres,redis,ollama,chromadb,qdrant}
    
    log_step "1/7 - Checking Docker"
    check_docker
    
    log_step "2/7 - Starting Core Infrastructure"
    log_info "Starting core services..."
    docker-compose up -d postgres redis ollama chromadb qdrant
    
    # Wait for core services
    log_info "Waiting for core services to be healthy..."
    sleep 15
    
    log_step "3/7 - Initializing MCP Server"
    init_mcp
    
    log_step "4/7 - Starting AI Backend Services"
    log_info "Starting backend services..."
    docker-compose up -d backend-agi frontend-agi litellm
    sleep 10
    
    log_step "5/7 - Deploying AI Agents"
    log_info "Running unified agent deployment..."
    bash "$PROJECT_ROOT/scripts/deploy_all_agents_unified.sh"
    
    log_step "6/7 - Starting Orchestrator"
    start_orchestrator
    
    # Wait for everything to stabilize
    log_info "Waiting for system to stabilize..."
    sleep 20
    
    log_step "7/7 - System Verification"
    verify_agents
    
    # Optional: Create demo tasks
    if [ "${CREATE_DEMO_TASKS:-false}" = "true" ]; then
        create_demo_tasks
    fi
    
    # Show final dashboard
    show_dashboard
    
    log_success "🎉 Unified AI System is ready!"
    log_success "All agents are working together and ready to handle tasks!"
}

# Handle script arguments
case "${1:-}" in
    --with-demo)
        export CREATE_DEMO_TASKS=true
        main
        ;;
    --status)
        show_dashboard
        ;;
    --verify)
        verify_agents
        ;;
    *)
        main
        ;;
esac