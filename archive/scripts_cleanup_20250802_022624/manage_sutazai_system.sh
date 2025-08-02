#!/bin/bash
# 🚀 SutazAI Complete System Manager
# Master control script for the entire 38-agent AI infrastructure

set -euo pipefail

# Configuration
PROJECT_ROOT=$(pwd)
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# ===============================================
# 🚀 LOGGING AND DISPLAY FUNCTIONS
# ===============================================

log_info() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${BLUE}ℹ️  [$timestamp] $message${NC}"
}

log_success() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${GREEN}✅ [$timestamp] $message${NC}"
}

log_warn() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${YELLOW}⚠️  [$timestamp] WARNING: $message${NC}"
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${RED}❌ [$timestamp] ERROR: $message${NC}"
}

show_banner() {
    clear
    echo -e "${CYAN}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███████╗██╗   ██╗████████╗ █████╗ ███████╗ █████╗ ██╗                     ║
║   ██╔════╝██║   ██║╚══██╔══╝██╔══██╗╚══███╔╝██╔══██╗██║                     ║
║   ███████╗██║   ██║   ██║   ███████║  ███╔╝ ███████║██║                     ║
║   ╚════██║██║   ██║   ██║   ██╔══██║ ███╔╝  ██╔══██║██║                     ║
║   ███████║╚██████╔╝   ██║   ██║  ██║███████╗██║  ██║██║                     ║
║   ╚══════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝                     ║
║                                                                              ║
║              🧠 COMPLETE AI AGENT INFRASTRUCTURE MANAGER 🧠                 ║
║                          38 AI Agents • Full AGI/ASI System                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

show_system_status() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                           🔍 SYSTEM STATUS                                  ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    
    # Running containers
    local running_containers=$(docker ps --format "{{.Names}}" | grep "^sutazai-" | wc -l)
    local total_expected=40  # Core services + 38 agents + infrastructure
    
    echo -e "${BLUE}📊 Container Status:${NC}"
    echo -e "   Running: ${GREEN}$running_containers${NC}/$total_expected containers"
    
    # Core services status
    echo -e "\n${BLUE}🏗️ Core Services:${NC}"
    local core_services=("postgres" "redis" "neo4j" "chromadb" "qdrant" "ollama")
    for service in "${core_services[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "sutazai-$service"; then
            echo -e "   $service: ${GREEN}●${NC} Running"
        else
            echo -e "   $service: ${RED}●${NC} Stopped"
        fi
    done
    
    # Agent communication
    echo -e "\n${BLUE}📡 Agent Communication:${NC}"
    if curl -s -f "http://localhost:8299/health" >/dev/null 2>&1; then
        echo -e "   Message Bus: ${GREEN}●${NC} Healthy"
    else
        echo -e "   Message Bus: ${RED}●${NC} Unhealthy"
    fi
    
    if curl -s -f "http://localhost:8300/health" >/dev/null 2>&1; then
        echo -e "   Agent Registry: ${GREEN}●${NC} Healthy"
        local registered_agents=$(curl -s "http://localhost:8300/stats" 2>/dev/null | jq -r '.total_agents // "0"' 2>/dev/null || echo "0")
        echo -e "   Registered Agents: ${CYAN}$registered_agents${NC}"
    else
        echo -e "   Agent Registry: ${RED}●${NC} Unhealthy"
    fi
    
    # Monitoring
    echo -e "\n${BLUE}📊 Monitoring:${NC}"
    local monitoring_services=("prometheus:9090" "grafana:3000" "loki:3100")
    for service_port in "${monitoring_services[@]}"; do
        local service="${service_port%:*}"
        local port="${service_port#*:}"
        if nc -z localhost "$port" 2>/dev/null; then
            echo -e "   $service: ${GREEN}●${NC} Available (http://localhost:$port)"
        else
            echo -e "   $service: ${RED}●${NC} Unavailable"
        fi
    done
    
    # System resources
    echo -e "\n${BLUE}💻 System Resources:${NC}"
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' | sed 's/[^0-9.]//g' || echo "unknown")
    local memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}' 2>/dev/null || echo "unknown")
    local disk_usage=$(df -h . | awk 'NR==2{print $5}' 2>/dev/null || echo "unknown")
    
    echo -e "   CPU Usage: ${CYAN}${cpu_usage}%${NC}"
    echo -e "   Memory Usage: ${CYAN}${memory_usage}%${NC}"
    echo -e "   Disk Usage: ${CYAN}${disk_usage}${NC}"
    
    echo ""
}

# ===============================================
# 🚀 SYSTEM MANAGEMENT FUNCTIONS
# ===============================================

deploy_system() {
    log_info "Starting complete system deployment..."
    
    if [[ -f "./scripts/deploy_complete_agent_infrastructure.sh" ]]; then
        log_info "Running deployment script..."
        ./scripts/deploy_complete_agent_infrastructure.sh
    else
        log_warn "Deployment script not found, using basic Docker Compose..."
        docker compose -f docker-compose.yml -f docker-compose-complete-agents.yml up -d
    fi
}

start_system() {
    log_info "Starting SutazAI system..."
    
    # Start core infrastructure first
    log_info "Starting core infrastructure..."
    docker compose up -d postgres redis neo4j chromadb qdrant ollama
    
    # Wait for core services
    log_info "Waiting for core services to be ready..."
    sleep 30
    
    # Start agent communication infrastructure
    log_info "Starting agent communication infrastructure..."
    docker compose up -d agent-message-bus agent-registry
    
    # Wait for communication infrastructure
    sleep 15
    
    # Start all agents
    log_info "Starting AI agents..."
    docker compose -f docker-compose-complete-agents.yml up -d
    
    # Start monitoring
    log_info "Starting monitoring stack..."
    docker compose up -d prometheus grafana loki promtail
    
    log_success "System startup initiated. Use 'status' to check progress."
}

stop_system() {
    log_info "Stopping SutazAI system..."
    
    # Stop agents first
    log_info "Stopping AI agents..."
    docker compose -f docker-compose-complete-agents.yml down
    
    # Stop monitoring
    log_info "Stopping monitoring..."
    docker compose stop prometheus grafana loki promtail
    
    # Stop core services
    log_info "Stopping core services..."
    docker compose down
    
    log_success "System stopped."
}

restart_system() {
    log_info "Restarting SutazAI system..."
    stop_system
    sleep 10
    start_system
}

cleanup_system() {
    log_warn "This will remove all containers, volumes, and data. Are you sure? (y/N)"
    read -r confirm
    
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        log_info "Cleaning up system..."
        
        # Stop everything
        docker compose -f docker-compose.yml -f docker-compose-complete-agents.yml down -v --remove-orphans
        
        # Remove images
        log_info "Removing SutazAI images..."
        docker images --format "{{.Repository}}:{{.Tag}}" | grep sutazai | xargs -r docker rmi -f
        
        # Clean up Docker system
        docker system prune -af
        
        log_success "System cleanup completed."
    else
        log_info "Cleanup cancelled."
    fi
}

# ===============================================
# 🚀 MONITORING AND MAINTENANCE
# ===============================================

monitor_system() {
    log_info "Starting system monitoring..."
    
    if [[ -f "./scripts/monitor_agents.sh" ]]; then
        ./scripts/monitor_agents.sh
    else
        log_error "Monitoring script not found"
        exit 1
    fi
}

validate_system() {
    log_info "Running system validation..."
    
    if [[ -f "./scripts/validate_agent_deployment.sh" ]]; then
        ./scripts/validate_agent_deployment.sh
    else
        log_error "Validation script not found"
        exit 1
    fi
}

backup_system() {
    log_info "Creating system backup..."
    
    if [[ -f "./scripts/backup_recovery_system.sh" ]]; then
        ./scripts/backup_recovery_system.sh backup full
    else
        log_error "Backup script not found"
        exit 1
    fi
}

show_logs() {
    local service="${1:-all}"
    
    if [[ "$service" == "all" ]]; then
        log_info "Showing logs for all services..."
        docker compose logs --tail=50 -f
    else
        log_info "Showing logs for service: $service"
        docker compose logs --tail=50 -f "$service"
    fi
}

# ===============================================
# 🚀 INTERACTIVE MENU
# ===============================================

show_menu() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                              🎮 MAIN MENU                                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}🚀 System Operations:${NC}"
    echo -e "   ${GREEN}1)${NC} Deploy Complete System        ${GREEN}2)${NC} Start System"
    echo -e "   ${GREEN}3)${NC} Stop System                   ${GREEN}4)${NC} Restart System"
    echo -e "   ${GREEN}5)${NC} System Status                 ${GREEN}6)${NC} Cleanup System"
    echo ""
    echo -e "${YELLOW}🔍 Monitoring & Maintenance:${NC}"
    echo -e "   ${GREEN}7)${NC} Monitor Agents                ${GREEN}8)${NC} Validate Deployment"
    echo -e "   ${GREEN}9)${NC} Create Backup                 ${GREEN}10)${NC} View Logs"
    echo ""
    echo -e "${YELLOW}🌐 Quick Access:${NC}"
    echo -e "   ${GREEN}11)${NC} Open Frontend (http://localhost:8501)"
    echo -e "   ${GREEN}12)${NC} Open Grafana (http://localhost:3000)"
    echo -e "   ${GREEN}13)${NC} Open Agent Registry (http://localhost:8300)"
    echo ""
    echo -e "   ${RED}q)${NC} Quit"
    echo ""
}

handle_menu_choice() {
    local choice="$1"
    
    case "$choice" in
        "1")
            deploy_system
            ;;
        "2")
            start_system
            ;;
        "3")
            stop_system
            ;;
        "4")
            restart_system
            ;;
        "5")
            show_system_status
            ;;
        "6")
            cleanup_system
            ;;
        "7")
            monitor_system
            ;;
        "8")
            validate_system
            ;;
        "9")
            backup_system
            ;;
        "10")
            echo "Enter service name (or 'all' for all services):"
            read -r service_name
            show_logs "${service_name:-all}"
            ;;
        "11")
            log_info "Opening frontend in browser..."
            if command -v xdg-open >/dev/null; then
                xdg-open http://localhost:8501
            elif command -v open >/dev/null; then
                open http://localhost:8501
            else
                log_info "Please open http://localhost:8501 in your browser"
            fi
            ;;
        "12")
            log_info "Opening Grafana in browser..."
            if command -v xdg-open >/dev/null; then
                xdg-open http://localhost:3000
            elif command -v open >/dev/null; then
                open http://localhost:3000
            else
                log_info "Please open http://localhost:3000 in your browser (admin/sutazai_grafana)"
            fi
            ;;
        "13")
            log_info "Opening Agent Registry in browser..."
            if command -v xdg-open >/dev/null; then
                xdg-open http://localhost:8300
            elif command -v open >/dev/null; then
                open http://localhost:8300
            else
                log_info "Please open http://localhost:8300 in your browser"
            fi
            ;;
        "q"|"Q")
            log_success "Goodbye!"
            exit 0
            ;;
        *)
            log_error "Invalid choice: $choice"
            ;;
    esac
}

interactive_mode() {
    while true; do
        show_banner
        show_system_status
        show_menu
        
        echo -ne "${CYAN}Choose an option: ${NC}"
        read -r choice
        
        echo ""
        handle_menu_choice "$choice"
        
        if [[ "$choice" != "5" ]]; then  # Don't pause for status updates
            echo ""
            echo -e "${YELLOW}Press Enter to continue...${NC}"
            read -r
        fi
    done
}

# ===============================================
# 🚀 COMMAND LINE INTERFACE
# ===============================================

show_help() {
    echo "SutazAI Complete AI Agent Infrastructure Manager"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy      Deploy complete system"
    echo "  start       Start the system"
    echo "  stop        Stop the system"
    echo "  restart     Restart the system"
    echo "  status      Show system status"
    echo "  monitor     Start monitoring"
    echo "  validate    Validate deployment"
    echo "  backup      Create system backup"
    echo "  cleanup     Clean up system (removes all data)"
    echo "  logs [SVC]  Show logs for service (or all)"
    echo "  menu        Interactive menu (default)"
    echo "  help        Show this help"
    echo ""
    echo "Examples:"
    echo "  $0              # Interactive menu"
    echo "  $0 deploy       # Deploy complete system"
    echo "  $0 status       # Show status"
    echo "  $0 logs redis   # Show Redis logs"
    echo ""
    echo "URLs:"
    echo "  Frontend:       http://localhost:8501"
    echo "  Grafana:        http://localhost:3000"
    echo "  Prometheus:     http://localhost:9090"
    echo "  Agent Registry: http://localhost:8300"
    echo "  Message Bus:    http://localhost:8299"
    echo ""
}

# ===============================================
# 🚀 MAIN EXECUTION
# ===============================================

main() {
    # Ensure we're in the project directory
    if [[ ! -f "docker-compose.yml" ]]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    local command="${1:-menu}"
    
    case "$command" in
        "deploy")
            show_banner
            deploy_system
            ;;
        "start")
            show_banner
            start_system
            ;;
        "stop")
            show_banner
            stop_system
            ;;
        "restart")
            show_banner
            restart_system
            ;;
        "status")
            show_banner
            show_system_status
            ;;
        "monitor")
            monitor_system
            ;;
        "validate")
            validate_system
            ;;
        "backup")
            backup_system
            ;;
        "cleanup")
            show_banner
            cleanup_system
            ;;
        "logs")
            show_logs "${2:-all}"
            ;;
        "menu")
            interactive_mode
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Operation interrupted by user${NC}"; exit 0' INT

# Run main function
main "$@"