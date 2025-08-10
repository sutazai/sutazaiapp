#!/bin/bash
# 🎨 SutazAI ASCII Logo Display Script
# Standalone script to display the SutazAI branding logo

set -euo pipefail


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

display_sutazai_logo() {
    # Color definitions for professional ASCII art display
    local CYAN='\033[0;36m'
    local BRIGHT_CYAN='\033[1;36m'
    local GREEN='\033[0;32m'
    local BRIGHT_GREEN='\033[1;32m'
    local YELLOW='\033[1;33m'
    local WHITE='\033[1;37m'
    local BLUE='\033[0;34m'
    local BRIGHT_BLUE='\033[1;34m'
    local RED='\033[0;31m'
    local BRIGHT_RED='\033[1;31m'
    local MAGENTA='\033[0;35m'
    local BRIGHT_MAGENTA='\033[1;35m'
    local RESET='\033[0m'
    local BOLD='\033[1m'
    
    clear
    echo ""
    echo -e "${BRIGHT_CYAN}════════════════════════════════════════════════════════════════════════════${RESET}"
    echo -e "${BRIGHT_GREEN} _________       __                   _____  .___${RESET}"
    echo -e "${BRIGHT_GREEN}/   _____/__ ___/  |______  ________ /  _  \\ |   |${RESET}"
    echo -e "${BRIGHT_GREEN}\\_____  \\|  |  \\   __\\__  \\ \\___   //  /_\\  \\|   |${RESET}"
    echo -e "${BRIGHT_GREEN}/        \\  |  /|  |  / __ \\_/    //    |    \\   |${RESET}"
    echo -e "${BRIGHT_GREEN}/_______  /____/ |__| (____  /_____ \\____|__  /___|${RESET}"
    echo -e "${BRIGHT_GREEN}        \\/                 \\/      \\/       \\/     ${RESET}"
    echo ""
    echo -e "${BRIGHT_CYAN}           🚀 Enterprise automation/advanced automation Autonomous System 🚀${RESET}"
    echo -e "${CYAN}                     Comprehensive AI Platform${RESET}"
    echo ""
    echo -e "${YELLOW}    • 50+ AI Services  • Vector Databases  • Model Management${RESET}"
    echo -e "${YELLOW}    • Agent Orchestration  • Enterprise Security  • 100% Local${RESET}"
    echo ""
    echo -e "${BRIGHT_BLUE}═══════════════════════════════════════════════════════════════════════════${RESET}"
    echo ""
    echo -e "${WHITE}🌟 ${BOLD}Advanced Local AI Infrastructure Platform${RESET}"
    echo -e "${WHITE}🔒 Secure • 🚀 Fast • 🧠 Intelligent • 🏢 Enterprise-Ready${RESET}"
    echo ""
    echo -e "${BRIGHT_MAGENTA}🎯 System Status:${RESET}"
    
    # Check system status
    if command -v docker >/dev/null 2>&1; then
        echo -e "${GREEN}   ✅ Docker: Available${RESET}"
    else
        echo -e "${RED}   ❌ Docker: Not found${RESET}"
    fi
    
    if docker ps >/dev/null 2>&1; then
        local running_containers=$(docker ps --format "{{.Names}}" | grep -c "sutazai" 2>/dev/null || echo "0")
        echo -e "${GREEN}   ✅ Docker Service: Running (${running_containers} SutazAI containers)${RESET}"
    else
        echo -e "${YELLOW}   ⚠️  Docker Service: Not accessible${RESET}"
    fi
    
    if [ -f "/opt/sutazaiapp/docker-compose.yml" ]; then
        echo -e "${GREEN}   ✅ SutazAI Configuration: Found${RESET}"
    else
        echo -e "${RED}   ❌ SutazAI Configuration: Missing${RESET}"
    fi
    
    echo ""
    echo -e "${BRIGHT_CYAN}════════════════════════════════════════════════════════════════════════════${RESET}"
    echo ""
}

# Handle command line arguments
case "${1:-display}" in
    "display"|"show"|"logo")
        display_sutazai_logo
        ;;
    "simple"|"plain")
        echo " _________       __                   _____  .___"
        echo "/   _____/__ ___/  |______  ________ /  _  \\ |   |"
        echo "\\_____  \\|  |  \\   __\\__  \\ \\___   //  /_\\  \\|   |"
        echo "/        \\  |  /|  |  / __ \\_/    //    |    \\   |"
        echo "/_______  /____/ |__| (____  /_____ \\____|__  /___|"
        echo "        \\/                 \\/      \\/       \\/     "
        ;;
    "help"|"-h"|"--help")
        echo "SutazAI Logo Display Script"
        echo "Usage: $0 [option]"
        echo "Options:"
        echo "  display, show, logo  - Show full colored logo with system status (default)"
        echo "  simple, plain        - Show plain ASCII logo only"
        echo "  help, -h, --help     - Show this help message"
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 