#!/bin/bash
# 🚀 SutazAI Quick Deploy Wrapper
# Simple wrapper for the main deployment script with automatic root elevation

set -euo pipefail

# Display SutazAI logo with colors
display_logo() {
    local BRIGHT_GREEN='\033[1;32m'
    local BRIGHT_CYAN='\033[1;36m'
    local YELLOW='\033[1;33m'
    local WHITE='\033[1;37m'
    local RESET='\033[0m'
    
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
    echo -e "${BRIGHT_CYAN}🚀 Quick Deploy Wrapper - Enterprise AGI/ASI System 🚀${RESET}"
    echo -e "${BRIGHT_CYAN}════════════════════════════════════════════════════════════════════════════${RESET}"
    echo -e "${YELLOW}🎯 This wrapper will automatically:${RESET}"
    echo -e "${WHITE}   • Check and elevate to root privileges if needed${RESET}"
    echo -e "${WHITE}   • Deploy all recent changes and improvements${RESET}"
    echo -e "${WHITE}   • Verify deployment success${RESET}"
    echo ""
}

display_logo

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/deploy_complete_system.sh"

# Check if main deployment script exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ ERROR: Main deployment script not found at: $MAIN_SCRIPT"
    echo "💡 Please ensure you're running this from the correct directory"
    exit 1
fi

# Check if main script is executable
if [ ! -x "$MAIN_SCRIPT" ]; then
    echo "🔧 Setting execute permissions on main deployment script..."
    chmod +x "$MAIN_SCRIPT" 2>/dev/null || {
        echo "⚠️  Cannot set permissions. Trying with sudo..."
        sudo chmod +x "$MAIN_SCRIPT"
    }
fi

echo "🔄 Executing main deployment script with all recent changes..."
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Execute the main deployment script with all arguments
exec "$MAIN_SCRIPT" "${@:-deploy}"