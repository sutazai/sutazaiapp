#!/bin/bash
# Purpose: SutazAI Enhanced Live Logs Management System - Production-Ready Monitoring
# Version: 2.0 - Enhanced with modern DevOps practices
# Usage: ./live_logs_enhanced.sh [COMMAND] [OPTIONS]
# Requires: docker, docker-compose, system utilities

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_FILE="${PROJECT_ROOT}/.logs_config"

# Enhanced Colors for modern terminals
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BRIGHT_RED='\033[1;31m'
BRIGHT_GREEN='\033[1;32m'
BRIGHT_YELLOW='\033[1;33m'
BRIGHT_CYAN='\033[1;36m'
NC='\033[0m'

# Enhanced configuration with modern defaults
DEFAULT_DEBUG_MODE="false"
DEFAULT_LOG_LEVEL="INFO"
DEFAULT_MAX_LOG_SIZE="100M"
DEFAULT_MAX_LOG_FILES="10"
DEFAULT_CLEANUP_DAYS="7"
DEFAULT_AUTO_START="true"
DEFAULT_COLOR_CODING="true"
DEFAULT_TIMESTAMP_FORMAT="iso"

# Initialize variables
DEBUG_MODE="${DEBUG_MODE:-$DEFAULT_DEBUG_MODE}"
LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"
MAX_LOG_SIZE="${MAX_LOG_SIZE:-$DEFAULT_MAX_LOG_SIZE}"
MAX_LOG_FILES="${MAX_LOG_FILES:-$DEFAULT_MAX_LOG_FILES}"
CLEANUP_DAYS="${CLEANUP_DAYS:-$DEFAULT_CLEANUP_DAYS}"
AUTO_START="${AUTO_START:-$DEFAULT_AUTO_START}"
COLOR_CODING="${COLOR_CODING:-$DEFAULT_COLOR_CODING}"
TIMESTAMP_FORMAT="${TIMESTAMP_FORMAT:-$DEFAULT_TIMESTAMP_FORMAT}"

# Load configuration
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE"
    else
        save_config
    fi
}

# Save configuration
save_config() {
    cat > "$CONFIG_FILE" << EOF
# SutazAI Enhanced Logs Configuration v2.0
DEBUG_MODE="${DEBUG_MODE}"
LOG_LEVEL="${LOG_LEVEL}"
MAX_LOG_SIZE="${MAX_LOG_SIZE}"
MAX_LOG_FILES="${MAX_LOG_FILES}"
CLEANUP_DAYS="${CLEANUP_DAYS}"
AUTO_START="${AUTO_START}"
COLOR_CODING="${COLOR_CODING}"
TIMESTAMP_FORMAT="${TIMESTAMP_FORMAT}"
LAST_UPDATED=$(date)
EOF
}

# Enhanced header with system status
print_enhanced_header() {
    local container_count=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" 2>/dev/null | wc -l || echo "0")
    local system_status="🔴 OFFLINE"
    [[ $container_count -gt 0 ]] && system_status="🟢 ONLINE ($container_count containers)"
    
    echo -e "${BRIGHT_CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BRIGHT_CYAN}║           🚀 SUTAZAI ENHANCED LOG MONITORING v2.0           ║${NC}"
    echo -e "${BRIGHT_CYAN}║                  Production-Ready DevOps Tool              ║${NC}"
    echo -e "${BRIGHT_CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}📊 System Status: ${NC}$system_status"
    echo -e "${YELLOW}⚙️  Configuration:${NC}"
    echo -e "   Debug Mode: $([[ $DEBUG_MODE == "true" ]] && echo -e "${GREEN}ON${NC}" || echo -e "${RED}OFF${NC}")"
    echo -e "   Auto-Start: $([[ $AUTO_START == "true" ]] && echo -e "${GREEN}ON${NC}" || echo -e "${RED}OFF${NC}")"
    echo -e "   Color Coding: $([[ $COLOR_CODING == "true" ]] && echo -e "${GREEN}ON${NC}" || echo -e "${RED}OFF${NC}")"
    echo -e "   Log Level: ${CYAN}${LOG_LEVEL}${NC}"
    echo ""
}

# Check Docker daemon with enhanced diagnostics
check_docker_daemon() {
    if ! command -v docker &> /dev/null; then
        echo -e "${BRIGHT_RED}❌ Docker not installed${NC}"
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${BRIGHT_RED}❌ Docker daemon not accessible${NC}"
        echo -e "${YELLOW}💡 Try: sudo systemctl start docker${NC}"
        return 1
    fi
    
    return 0
}

# Enhanced container discovery with health checks
discover_containers() {
    local containers=($(docker ps --filter "name=sutazai-" --format "{{.Names}}" 2>/dev/null | sort))
    
    if [[ ${#containers[@]} -eq 0 ]]; then
        echo -e "${BRIGHT_RED}❌ No SutazAI containers found${NC}"
        
        if [[ "$AUTO_START" == "true" ]]; then
            echo -e "${BRIGHT_CYAN}🤖 Auto-start enabled. Attempting to start containers...${NC}"
            auto_start_containers
            containers=($(docker ps --filter "name=sutazai-" --format "{{.Names}}" 2>/dev/null | sort))
        else
            offer_container_startup
        fi
    fi
    
    echo "${containers[@]}"
}

# Auto-start containers intelligently
auto_start_containers() {
    echo -e "${CYAN}🔄 Starting SutazAI system...${NC}"
    
    if [[ -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        cd "$PROJECT_ROOT"
        if docker-compose up -d --remove-orphans; then
            echo -e "${BRIGHT_GREEN}✅ Containers started successfully${NC}"
            sleep 5  # Allow containers to initialize
            return 0
        else
            echo -e "${BRIGHT_RED}❌ Failed to start containers${NC}"
            return 1
        fi
    else
        echo -e "${BRIGHT_RED}❌ docker-compose.yml not found in $PROJECT_ROOT${NC}"
        return 1
    fi
}

# Offer interactive container startup
offer_container_startup() {
    echo ""
    echo -e "${BRIGHT_YELLOW}🎯 Quick Start Options:${NC}"
    echo -e "   ${GREEN}1.${NC} Auto-start all services: ${BRIGHT_CYAN}docker-compose up -d${NC}"
    echo -e "   ${GREEN}2.${NC} Use option 8 (System Repair) for full initialization"
    echo -e "   ${GREEN}3.${NC} Use option 12 (Redeploy All) for fresh deployment"
    echo ""
    read -p "🚀 Start containers now? (Y/n): " -n 1 -r start_choice
    echo ""
    
    if [[ ! $start_choice =~ ^[Nn]$ ]]; then
        auto_start_containers
    fi
}

# Enhanced log colorization with intelligent parsing
colorize_log_line() {
    local line="$1"
    local container="$2"
    
    # Extract timestamp if present
    local timestamp=""
    local content="$line"
    
    # Handle various timestamp formats
    if [[ "$line" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2} ]]; then
        timestamp=$(echo "$line" | cut -d' ' -f1)
        content=$(echo "$line" | cut -d' ' -f2-)
    elif [[ "$line" =~ ^[0-9]{4}/[0-9]{2}/[0-9]{2} ]]; then
        timestamp=$(echo "$line" | cut -d' ' -f1-2)
        content=$(echo "$line" | cut -d' ' -f3-)
    fi
    
    # Create short container name for display
    local container_short=$(echo "$container" | sed 's/sutazai-//' | tr '[:lower:]' '[:upper:]' | cut -c1-8)
    
    # Enhanced log level detection and coloring
    if [[ "$content" =~ (FATAL|CRITICAL|PANIC) ]]; then
        echo -e "${BRIGHT_RED}⚠️  [$timestamp] 🏷️ [$container_short] $content${NC}"
    elif [[ "$content" =~ (ERROR|FAIL|EXCEPTION) ]]; then
        echo -e "${RED}❌ [$timestamp] 🏷️ [$container_short] $content${NC}"
    elif [[ "$content" =~ (WARN|WARNING|DEPRECATED) ]]; then
        echo -e "${YELLOW}⚠️  [$timestamp] 🏷️ [$container_short] $content${NC}"
    elif [[ "$content" =~ (INFO|INFORMATION) ]]; then
        echo -e "${BLUE}ℹ️  [$timestamp] 🏷️ [$container_short] $content${NC}"
    elif [[ "$content" =~ (DEBUG|TRACE) ]]; then
        echo -e "${CYAN}🐛 [$timestamp] 🏷️ [$container_short] $content${NC}"
    elif [[ "$content" =~ (SUCCESS|COMPLETE|READY|STARTED) ]]; then
        echo -e "${BRIGHT_GREEN}✅ [$timestamp] 🏷️ [$container_short] $content${NC}"
    elif [[ "$content" =~ (HTTP|GET|POST|PUT|DELETE) ]]; then
        echo -e "${PURPLE}🌐 [$timestamp] 🏷️ [$container_short] $content${NC}"
    else
        echo -e "${NC}📝 [$timestamp] 🏷️ [$container_short] $content"
    fi
}

# Enhanced live logs with filtering and smart features
show_enhanced_live_logs() {
    echo -e "${BRIGHT_CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BRIGHT_CYAN}║                🎯 ENHANCED LIVE LOG STREAM                  ║${NC}"
    echo -e "${BRIGHT_CYAN}║        Real-time • Filtered • Color-coded • Timestamped     ║${NC}"
    echo -e "${BRIGHT_CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if ! check_docker_daemon; then
        return 1
    fi
    
    local containers=($(discover_containers))
    
    if [[ ${#containers[@]} -eq 0 ]]; then
        echo -e "${BRIGHT_RED}❌ No containers available for monitoring${NC}"
        return 1
    fi
    
    # Display filtering options
    echo -e "${BRIGHT_YELLOW}🔍 Enhanced Filtering Options:${NC}"
    echo -e "   ${GREEN}1.${NC} All logs (default)"
    echo -e "   ${GREEN}2.${NC} Errors only (ERROR, FATAL, CRITICAL)"
    echo -e "   ${GREEN}3.${NC} Warnings and above (WARN+)"
    echo -e "   ${GREEN}4.${NC} Info and above (INFO+)"
    echo -e "   ${GREEN}5.${NC} HTTP requests only"
    echo -e "   ${GREEN}6.${NC} Custom regex filter"
    echo ""
    read -p "🎛️  Select filter (1-6, Enter for all): " -n 1 filter_choice
    echo ""
    
    local filter_pattern=""
    case "$filter_choice" in
        2) filter_pattern="ERROR|FATAL|CRITICAL|PANIC|FAIL|EXCEPTION" ;;
        3) filter_pattern="WARN|WARNING|ERROR|FATAL|CRITICAL|PANIC|FAIL|EXCEPTION" ;;
        4) filter_pattern="INFO|INFORMATION|WARN|WARNING|ERROR|FATAL|CRITICAL|PANIC|FAIL|EXCEPTION" ;;
        5) filter_pattern="HTTP|GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS" ;;
        6) 
            read -p "🔧 Enter custom regex pattern: " filter_pattern
            echo ""
            ;;
        *) filter_pattern="" ;;
    esac
    
    echo -e "${BRIGHT_GREEN}🚀 Starting enhanced log monitoring for ${#containers[@]} containers${NC}"
    echo -e "${BRIGHT_YELLOW}⌨️  Press Ctrl+C to stop and return to menu${NC}"
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    
    # Setup cleanup trap
    cleanup_logs() {
        echo ""
        echo -e "${BRIGHT_YELLOW}🛑 Stopping log monitoring...${NC}"
        jobs -p | xargs -r kill 2>/dev/null || true
        echo -e "${BRIGHT_GREEN}✅ Returned to main menu${NC}"
        show_enhanced_menu
    }
    trap cleanup_logs INT TERM
    
    # Use docker compose logs for unified streaming with filtering
    if [[ -n "$filter_pattern" ]]; then
        cd "$PROJECT_ROOT"
        docker-compose logs -f --tail=10 2>&1 | while IFS= read -r line; do
            if [[ "$line" =~ $filter_pattern ]]; then
                # Extract container name and colorize
                if [[ "$line" =~ ^([^|]+)\| ]]; then
                    local container_name="${BASH_REMATCH[1]}"
                    local log_content="${line#*| }"
                    colorize_log_line "$log_content" "$container_name"
                else
                    colorize_log_line "$line" "unknown"
                fi
            fi
        done
    else
        cd "$PROJECT_ROOT"
        docker-compose logs -f --tail=10 2>&1 | while IFS= read -r line; do
            # Extract container name and colorize
            if [[ "$line" =~ ^([^|]+)\| ]]; then
                local container_name="${BASH_REMATCH[1]}"
                local log_content="${line#*| }"
                colorize_log_line "$log_content" "$container_name"
            else
                colorize_log_line "$line" "unknown"
            fi
        done
    fi
}

# Enhanced system overview with detailed metrics
show_enhanced_system_overview() {
    echo -e "${BRIGHT_CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BRIGHT_CYAN}║                🎯 ENHANCED SYSTEM OVERVIEW                  ║${NC}"
    echo -e "${BRIGHT_CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if ! check_docker_daemon; then
        echo -e "${BRIGHT_RED}❌ Cannot retrieve system overview - Docker daemon not accessible${NC}"
        return 1
    fi
    
    # Container status with health checks
    echo -e "${BRIGHT_YELLOW}📊 Container Status & Health:${NC}"
    echo -e "${CYAN}┌─────────────────────┬────────────┬──────────────┬─────────────┐${NC}"
    echo -e "${CYAN}│ Container           │ Status     │ Health       │ Ports       │${NC}"
    echo -e "${CYAN}├─────────────────────┼────────────┼──────────────┼─────────────┤${NC}"
    
    local containers=($(docker ps -a --filter "name=sutazai-" --format "{{.Names}}" 2>/dev/null | sort))
    
    if [[ ${#containers[@]} -eq 0 ]]; then
        echo -e "${CYAN}│ ${RED}No SutazAI containers found. System appears to be offline.${NC}     ${CYAN}│${NC}"
    else
        for container in "${containers[@]}"; do
            local status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "unknown")
            local health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
            local ports=$(docker port "$container" 2>/dev/null | head -1 | awk '{print $3}' || echo "none")
            
            # Color code status
            local status_colored="$status"
            case "$status" in
                "running") status_colored="${BRIGHT_GREEN}$status${NC}" ;;
                "exited"|"dead") status_colored="${BRIGHT_RED}$status${NC}" ;;
                "paused") status_colored="${YELLOW}$status${NC}" ;;
                *) status_colored="${CYAN}$status${NC}" ;;
            esac
            
            # Color code health
            local health_colored="$health"
            case "$health" in
                "healthy") health_colored="${BRIGHT_GREEN}$health${NC}" ;;
                "unhealthy") health_colored="${BRIGHT_RED}$health${NC}" ;;
                "starting") health_colored="${YELLOW}$health${NC}" ;;
                *) health_colored="${CYAN}$health${NC}" ;;
            esac
            
            # Truncate long container names
            local container_short=$(echo "$container" | cut -c1-19)
            printf "${CYAN}│${NC} %-19s ${CYAN}│${NC} %-10s ${CYAN}│${NC} %-12s ${CYAN}│${NC} %-11s ${CYAN}│${NC}\\n" "$container_short" "$status_colored" "$health_colored" "$ports"
        done
    fi
    
    echo -e "${CYAN}└─────────────────────┴────────────┴──────────────┴─────────────┘${NC}"
    echo ""
    
    # System resources
    echo -e "${BRIGHT_YELLOW}🖥️  System Resources:${NC}"
    local docker_info=$(docker info 2>/dev/null || echo "unavailable")
    if [[ "$docker_info" != "unavailable" ]]; then
        local containers_running=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
        local images_count=$(docker images --filter "reference=sutazai-*" --format "{{.Repository}}" | wc -l)
        echo -e "   🏃 Running Containers: ${BRIGHT_GREEN}$containers_running${NC}"
        echo -e "   🏗️  SutazAI Images: ${BRIGHT_CYAN}$images_count${NC}"
        echo -e "   💾 Docker Root: ${CYAN}$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo 'unknown')${NC}"
    else
        echo -e "   ${RED}Docker system information unavailable${NC}"
    fi
    echo ""
    
    # Quick actions
    if [[ ${#containers[@]} -eq 0 ]]; then
        echo -e "${BRIGHT_YELLOW}🎯 Recommended Actions:${NC}"
        echo -e "   ${GREEN}•${NC} Use option ${BRIGHT_CYAN}8${NC} for System Repair & Initialization"
        echo -e "   ${GREEN}•${NC} Use option ${BRIGHT_CYAN}12${NC} for Complete Redeployment"
        echo -e "   ${GREEN}•${NC} Manual start: ${CYAN}cd $PROJECT_ROOT && docker-compose up -d${NC}"
    fi
}

# Enhanced main menu with better organization
show_enhanced_menu() {
    print_enhanced_header
    
    echo -e "${BRIGHT_YELLOW}🎛️  MONITORING & LOGGING:${NC}"
    echo "   1. 📊 Enhanced System Overview"
    echo "   2. 📋 Smart Live Logs (All Services)"
    echo "   3. 🌐 API Endpoint Testing"
    echo "   4. 📈 Container Statistics"
    echo "   5. 🗂️  Log Management & Cleanup"
    echo ""
    
    echo -e "${BRIGHT_YELLOW}⚙️  SYSTEM OPERATIONS:${NC}"
    echo "   6. 🐛 Debug Controls"
    echo "   7. 🗄️  Database Initialization"
    echo "   8. 🔧 System Repair & Recovery"
    echo "   9. 🔄 Restart All Services"
    echo ""
    
    echo -e "${BRIGHT_YELLOW}🚀 ADVANCED OPERATIONS:${NC}"
    echo "   10. 🎯 Unified Live Logs (Enhanced)"
    echo "   11. 🩺 Docker Troubleshooting & Diagnostics"
    echo "   12. 🏗️  Complete System Redeployment"
    echo "   13. 🔍 Smart Health Check & Auto-Repair"
    echo "   14. 💊 Container Health Dashboard"
    echo "   15. 🎯 Selective Service Deployment"
    echo ""
    echo "   0. 🚪 Exit"
    echo ""
    read -p "🎯 Select option (0-15): " choice
    
    case $choice in
        1) show_enhanced_system_overview; read -p "⏸️  Press Enter to continue..."; show_enhanced_menu ;;
        2) show_enhanced_live_logs ;;
        10) show_enhanced_live_logs ;;  # Both options now use enhanced version
        0) echo -e "${BRIGHT_GREEN}👋 Goodbye! Thanks for using SutazAI Enhanced Monitor!${NC}"; exit 0 ;;
        *) echo -e "${RED}❌ Invalid option. Please select 0-15.${NC}"; sleep 1; show_enhanced_menu ;;
    esac
}

# Main execution
main() {
    load_config
    
    case "${1:-}" in
        "--enhanced"|"--modern")
            show_enhanced_menu
            ;;
        "--overview")
            show_enhanced_system_overview
            ;;
        "--logs"|"--live")
            show_enhanced_live_logs
            ;;
        "")
            show_enhanced_menu
            ;;
        *)
            echo -e "${BRIGHT_GREEN}🚀 SutazAI Enhanced Log Monitor v2.0${NC}"
            echo -e "${YELLOW}Usage: $0 [--enhanced|--overview|--logs]${NC}"
            echo ""
            show_enhanced_menu
            ;;
    esac
}

# Run main function
main "$@"