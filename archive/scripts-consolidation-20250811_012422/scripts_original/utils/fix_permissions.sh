#!/bin/bash

# SutazAI Permission Fix Script
# Resolves Docker permission issues for development files

set -euo pipefail

# Colors for output

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

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="/opt/sutazaiapp"
CURRENT_USER="${USER:-$(whoami)}"

echo -e "${BLUE}🔧 SutazAI Permission Fix Script${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  Running as root. This script should be run as the development user.${NC}"
    echo "Please run as: sudo -u ai $0"
    exit 1
fi

# Check if user is in sudo group
if ! groups "$CURRENT_USER" | grep -q sudo; then
    echo -e "${YELLOW}⚠️  User $CURRENT_USER is not in sudo group.${NC}"
    echo "Permission fixes may fail."
fi

echo -e "${BLUE}Current user: $CURRENT_USER${NC}"
echo -e "${BLUE}Project root: $PROJECT_ROOT${NC}"
echo ""

# Function to fix ownership
fix_ownership() {
    local path=$1
    local description=$2
    
    if [[ -e "$path" ]]; then
        echo -e "${BLUE}Fixing: $description${NC}"
        sudo chown -R "$CURRENT_USER:$CURRENT_USER" "$path"
        echo -e "${GREEN}✅ Fixed: $path${NC}"
    else
        echo -e "${YELLOW}⚠️  Not found: $path${NC}"
    fi
}

# Fix core project files
echo -e "${BLUE}Fixing core project files...${NC}"
fix_ownership "$PROJECT_ROOT/docker-compose.yml" "Docker Compose configuration"
fix_ownership "$PROJECT_ROOT/.mcp.json" "MCP configuration"
fix_ownership "$PROJECT_ROOT/.env" "Environment variables"
fix_ownership "$PROJECT_ROOT/package.json" "Node.js package file"
fix_ownership "$PROJECT_ROOT/requirements.txt" "Python requirements"

# Fix directories
echo ""
echo -e "${BLUE}Fixing directories...${NC}"
fix_ownership "$PROJECT_ROOT/mcp_server" "MCP server directory"
fix_ownership "$PROJECT_ROOT/logs" "Logs directory"
fix_ownership "$PROJECT_ROOT/data" "Data directory"
fix_ownership "$PROJECT_ROOT/scripts" "Scripts directory"

# Fix backup files
echo ""
echo -e "${BLUE}Fixing backup files...${NC}"
for backup_file in "$PROJECT_ROOT"/.env.backup.*; do
    if [[ -f "$backup_file" ]]; then
        sudo chown "$CURRENT_USER:$CURRENT_USER" "$backup_file"
    fi
done

# Set proper permissions for scripts
echo ""
echo -e "${BLUE}Setting executable permissions for scripts...${NC}"
if [[ -d "$PROJECT_ROOT/scripts" ]]; then
    find "$PROJECT_ROOT/scripts" -name "*.sh" -exec chmod +x {} \;
    echo -e "${GREEN}✅ Script permissions set${NC}"
fi

# Fix MCP server permissions specifically
if [[ -d "$PROJECT_ROOT/mcp_server" ]]; then
    chmod +x "$PROJECT_ROOT/mcp_server/setup.sh" 2>/dev/null || true
    chmod +x "$PROJECT_ROOT/mcp_server/index.js" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}🎉 Permission fixes completed!${NC}"
echo ""
echo -e "${BLUE}Prevention tips:${NC}"
echo -e "${BLUE}1. Always run docker-compose as the development user${NC}"
echo -e "${BLUE}2. Avoid using 'sudo docker-compose' unless necessary${NC}"
echo -e "${BLUE}3. Use 'sudo usermod -aG docker $CURRENT_USER' to add user to docker group${NC}"
echo -e "${BLUE}4. Run this script whenever permission issues occur${NC}"
echo ""

# Check Docker group membership
if ! groups "$CURRENT_USER" | grep -q docker; then
    echo -e "${YELLOW}💡 Tip: Add $CURRENT_USER to docker group to avoid sudo:${NC}"
    echo -e "${YELLOW}   sudo usermod -aG docker $CURRENT_USER${NC}"
    echo -e "${YELLOW}   Then log out and back in${NC}"
    echo ""
fi

# Verify key files are now writable
echo -e "${BLUE}Verifying write permissions...${NC}"
for file in "docker-compose.yml" ".mcp.json" ".env"; do
    if [[ -w "$PROJECT_ROOT/$file" ]]; then
        echo -e "${GREEN}✅ $file is writable${NC}"
    else
        echo -e "${YELLOW}⚠️  $file may still have permission issues${NC}"
    fi
done

echo ""
echo -e "${GREEN}🚀 Ready to continue development!${NC}" 