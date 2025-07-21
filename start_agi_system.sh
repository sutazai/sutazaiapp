#!/bin/bash
#
# Quick Start Script for SutazAI AGI/ASI System
# For full deployment, use deploy_agi_complete.sh
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_ROOT="/opt/sutazaiapp"

# Header
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            SutazAI AGI/ASI System Startup                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Change to project directory
cd "$PROJECT_ROOT"

# Check prerequisites
if [ ! -f ".env" ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo -e "${YELLOW}Run ./deploy_agi_complete.sh first for initial setup${NC}"
    exit 1
fi

if [ ! -f "docker-compose-complete-agi.yml" ]; then
    echo -e "${RED}ERROR: docker-compose-complete-agi.yml not found!${NC}"
    echo -e "${YELLOW}Run ./deploy_agi_complete.sh to generate required files${NC}"
    exit 1
fi

# Check Docker
if ! docker ps &> /dev/null; then
    echo -e "${RED}ERROR: Cannot connect to Docker daemon${NC}"
    echo -e "${YELLOW}Make sure Docker is running and you have proper permissions${NC}"
    exit 1
fi

# Stop any existing services
echo -e "${BLUE}Stopping existing services...${NC}"
docker-compose down 2>/dev/null || true
docker-compose -f docker-compose-complete-agi.yml down 2>/dev/null || true

# Start all services
echo -e "${BLUE}Starting all AGI services...${NC}"
docker-compose -f docker-compose-complete-agi.yml up -d

# Function to check service health with spinner
check_service_health() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    printf "${BLUE}Checking %-20s${NC}" "$service..."
    
    while [ $attempt -lt $max_attempts ]; do
        if nc -z localhost "$port" 2>/dev/null; then
            echo -e " ${GREEN}✓ Ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        printf "."
        sleep 2
    done
    
    echo -e " ${RED}✗ Failed${NC}"
    return 1
}

# Wait for critical services
echo ""
echo -e "${BLUE}Waiting for services to initialize...${NC}"
echo ""

# Core infrastructure
check_service_health "PostgreSQL" 5432
check_service_health "Redis" 6379
check_service_health "Neo4j" 7474

# Vector databases
check_service_health "ChromaDB" 8001
check_service_health "Qdrant" 6333

# Model serving
check_service_health "Ollama" 11434

# Application services  
check_service_health "Backend API" 8000
check_service_health "Frontend" 8501

# Monitoring
check_service_health "Prometheus" 9090
check_service_health "Grafana" 3003

# Service status table
echo ""
echo -e "${BLUE}Service Status:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai | head -20 || echo "No services found"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Backend health check
echo ""
echo -e "${BLUE}System Health Check:${NC}"
if response=$(curl -s http://localhost:8000/health 2>/dev/null); then
    if echo "$response" | grep -q "healthy"; then
        echo -e "${GREEN}✓ Backend API is healthy${NC}"
        
        # Parse and display key metrics
        if command -v jq &> /dev/null; then
            agents_healthy=$(echo "$response" | jq -r '.agents_healthy // 0')
            agents_total=$(echo "$response" | jq -r '.agents_total // 0')
            gpu_available=$(echo "$response" | jq -r '.gpu_available // false')
            
            echo -e "  Agents: ${agents_healthy}/${agents_total} healthy"
            echo -e "  GPU: $([ "$gpu_available" = "true" ] && echo "Available" || echo "Not available")"
        fi
    else
        echo -e "${YELLOW}⚠ Backend API returned unexpected response${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Backend API health check failed${NC}"
    echo -e "${YELLOW}  The system may still be initializing...${NC}"
fi

# Success message
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${GREEN}        ✅ SutazAI AGI/ASI System Started Successfully!      ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"

# Access information
echo ""
echo -e "${GREEN}Access Points:${NC}"
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ ${BLUE}Main UI:${NC}          http://localhost:8501                     │"
echo "│ ${BLUE}API:${NC}              http://localhost:8000                     │"
echo "│ ${BLUE}API Docs:${NC}         http://localhost:8000/docs                │"
echo "│ ${BLUE}Neo4j Browser:${NC}    http://localhost:7474                     │"
echo "│ ${BLUE}Prometheus:${NC}       http://localhost:9090                     │"
echo "│ ${BLUE}Grafana:${NC}          http://localhost:3003                     │"
echo "└─────────────────────────────────────────────────────────────┘"

# Useful commands
echo ""
echo -e "${YELLOW}Useful Commands:${NC}"
echo "  ${BLUE}View logs:${NC}     docker-compose -f docker-compose-complete-agi.yml logs -f [service]"
echo "  ${BLUE}Stop system:${NC}   docker-compose -f docker-compose-complete-agi.yml down"
echo "  ${BLUE}Restart:${NC}       docker-compose -f docker-compose-complete-agi.yml restart [service]"
echo "  ${BLUE}Status:${NC}        docker ps | grep sutazai"
echo "  ${BLUE}Exec into:${NC}     docker exec -it sutazai-[service] bash"

# Tips
echo ""
echo -e "${YELLOW}Tips:${NC}"
echo "  • First startup may take several minutes to initialize all services"
echo "  • Check logs if any service fails to start properly"
echo "  • Credentials are stored in the .env file"
echo "  • For full deployment options, use: ./deploy_agi_complete.sh"

# Optional: Open browser
if command -v xdg-open &> /dev/null; then
    echo ""
    read -p "Would you like to open the UI in your browser? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open http://localhost:8501 &
    fi
fi