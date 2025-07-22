#!/bin/bash
# SutazAI Live Logging System
# Centralized monitoring of all SutazAI containers

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
log_with_color() {
    local color=$1
    local service=$2
    local message=$3
    echo -e "${color}[$(date +'%H:%M:%S')] [${service}]${NC} ${message}"
}

# Function to check container status
check_container_status() {
    local container=$1
    if docker ps --filter "name=${container}" --format "{{.Names}}" | grep -q "${container}"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
}

# Function to get container health
get_container_health() {
    local container=$1
    local health=$(docker inspect --format='{{.State.Health.Status}}' $container 2>/dev/null)
    if [ "$health" = "healthy" ]; then
        echo -e "${GREEN}healthy${NC}"
    elif [ "$health" = "unhealthy" ]; then
        echo -e "${RED}unhealthy${NC}"
    elif [ "$health" = "starting" ]; then
        echo -e "${YELLOW}starting${NC}"
    else
        echo -e "${BLUE}no-check${NC}"
    fi
}

# Function to display system overview
show_system_overview() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    SUTAZAI LIVE MONITORING                   ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Service status table
    echo -e "${BLUE}Service Status:${NC}"
    echo "┌─────────────────────┬────────┬──────────┬─────────────┐"
    echo "│ Container           │ Status │ Health   │ Ports       │"
    echo "├─────────────────────┼────────┼──────────┼─────────────┤"
    
    containers=(
        "sutazai-postgres:5432"
        "sutazai-redis:6379"
        "sutazai-neo4j:7474,7687"
        "sutazai-chromadb:8001"
        "sutazai-qdrant:6333"
        "sutazai-ollama:11434"
        "sutazai-backend-agi:8000"
        "sutazai-frontend-agi:8501"
    )
    
    for container_info in "${containers[@]}"; do
        IFS=':' read -r container ports <<< "$container_info"
        status=$(check_container_status "$container")
        health=$(get_container_health "$container")
        printf "│ %-19s │ %-6s │ %-8s │ %-11s │\n" "$container" "$status" "$health" "$ports"
    done
    
    echo "└─────────────────────┴────────┴──────────┴─────────────┘"
    echo ""
    
    # Quick API tests
    echo -e "${BLUE}API Connectivity:${NC}"
    
    # Test backend health
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "Backend API (8000): ${GREEN}✓ Responding${NC}"
    else
        echo -e "Backend API (8000): ${RED}✗ Not responding${NC}"
    fi
    
    # Test frontend
    if curl -f http://localhost:8501/healthz >/dev/null 2>&1; then
        echo -e "Frontend (8501): ${GREEN}✓ Responding${NC}"
    else
        echo -e "Frontend (8501): ${RED}✗ Not responding${NC}"
    fi
    
    # Test Ollama
    if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo -e "Ollama API (11434): ${GREEN}✓ Responding${NC}"
    else
        echo -e "Ollama API (11434): ${RED}✗ Not responding${NC}"
    fi
    
    echo ""
    echo -e "${PURPLE}Access URLs:${NC}"
    echo "• Frontend: http://172.31.77.193:8501"
    echo "• Backend API: http://172.31.77.193:8000"
    echo "• API Docs: http://172.31.77.193:8000/docs"
    echo ""
}

# Function to show live logs
show_live_logs() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     SUTAZAI LIVE LOGS                       ║${NC}"
    echo -e "${CYAN}║                  Press Ctrl+C to exit                       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Start background log monitoring
    (
        docker logs -f sutazai-backend-agi 2>&1 | while IFS= read -r line; do
            log_with_color "$GREEN" "BACKEND" "$line"
        done
    ) &
    BACKEND_PID=$!
    
    (
        docker logs -f sutazai-frontend-agi 2>&1 | while IFS= read -r line; do
            log_with_color "$BLUE" "FRONTEND" "$line"
        done
    ) &
    FRONTEND_PID=$!
    
    (
        docker logs -f sutazai-ollama 2>&1 | while IFS= read -r line; do
            log_with_color "$YELLOW" "OLLAMA" "$line"
        done
    ) &
    OLLAMA_PID=$!
    
    # Wait for interrupt
    trap 'kill $BACKEND_PID $FRONTEND_PID $OLLAMA_PID 2>/dev/null; exit 0' INT TERM
    wait
}

# Function to test API endpoints
test_api_endpoints() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    API ENDPOINT TESTING                     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    endpoints=(
        "GET:http://localhost:8000/health:Health Check"
        "GET:http://localhost:8000/:System Info"
        "GET:http://localhost:8000/agents:Agent List"
        "GET:http://localhost:8000/models:Model List"
        "POST:http://localhost:8000/simple-chat:Simple Chat"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r method url description <<< "$endpoint_info"
        echo -n "Testing $description ($method $url)... "
        
        if [ "$method" = "GET" ]; then
            if curl -f "$url" >/dev/null 2>&1; then
                echo -e "${GREEN}✓ OK${NC}"
            else
                echo -e "${RED}✗ FAILED${NC}"
            fi
        elif [ "$method" = "POST" ]; then
            if curl -f -X POST "$url" -H "Content-Type: application/json" -d '{"message":"test"}' >/dev/null 2>&1; then
                echo -e "${GREEN}✓ OK${NC}"
            else
                echo -e "${RED}✗ FAILED${NC}"
            fi
        fi
        sleep 0.5
    done
    
    echo ""
    echo "Testing frontend connectivity..."
    if curl -f http://172.31.77.193:8501/healthz >/dev/null 2>&1; then
        echo -e "Frontend Health: ${GREEN}✓ OK${NC}"
    else
        echo -e "Frontend Health: ${RED}✗ FAILED${NC}"
    fi
}

# Function to show container stats
show_container_stats() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   CONTAINER STATISTICS                      ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.PIDs}}" \
        sutazai-postgres sutazai-redis sutazai-neo4j sutazai-chromadb \
        sutazai-qdrant sutazai-ollama sutazai-backend-agi sutazai-frontend-agi 2>/dev/null
}

# Main menu
show_menu() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   SUTAZAI MONITORING MENU                   ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "1. System Overview"
    echo "2. Live Logs (All Services)"
    echo "3. Test API Endpoints"
    echo "4. Container Statistics"
    echo "5. Restart All Services"
    echo "6. Exit"
    echo ""
    read -p "Select option (1-6): " choice
    
    case $choice in
        1) show_system_overview; read -p "Press Enter to continue..."; show_menu ;;
        2) show_live_logs ;;
        3) test_api_endpoints; read -p "Press Enter to continue..."; show_menu ;;
        4) show_container_stats; read -p "Press Enter to continue..."; show_menu ;;
        5) 
            echo "Restarting all SutazAI services..."
            docker-compose -f /opt/sutazaiapp/docker-compose-consolidated.yml restart
            echo "All services restarted!"
            read -p "Press Enter to continue..."
            show_menu
            ;;
        6) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid option"; show_menu ;;
    esac
}

# Start the monitoring system
if [ "$1" = "--overview" ]; then
    show_system_overview
elif [ "$1" = "--logs" ]; then
    show_live_logs
elif [ "$1" = "--test" ]; then
    test_api_endpoints
elif [ "$1" = "--stats" ]; then
    show_container_stats
else
    show_menu
fi