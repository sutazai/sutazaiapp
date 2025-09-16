#!/bin/bash
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Script: reorganize_scripts.sh
# Purpose: Reorganize existing scripts into Rule 7 compliant structure
# Author: Sutazai System
# Date: 2025-09-03
# Version: 1.0.0
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail
IFS=$'\n\t'

# Color codes
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

echo -e "${BLUE}Reorganizing existing scripts into Rule 7 structure...${NC}\n"

# Function to move and create symlink
move_script() {
    local source="$1"
    local dest="$2"
    local dest_dir=$(dirname "$dest")
    
    if [ -f "$source" ]; then
        # Create destination directory if needed
        mkdir -p "$dest_dir"
        
        # Copy to new location (preserve original for safety)
        cp "$source" "$dest"
        
        # Create symlink for backward compatibility
        ln -sf "$dest" "$source"
        
        echo -e "${GREEN}✓${NC} Moved: $(basename $source) → $dest"
    else
        echo -e "${YELLOW}⚠${NC}  Not found: $source"
    fi
}

# Reorganize deployment scripts
echo -e "${YELLOW}=== Deployment Scripts ===${NC}"
move_script "deploy/start-infrastructure.sh" "deploy/infrastructure/start-infrastructure.sh"
move_script "deploy/stop-infrastructure.sh" "deploy/infrastructure/stop-infrastructure.sh"
move_script "deploy/configure-kong-routes.sh" "deploy/infrastructure/configure-kong-routes.sh"
move_script "deploy/register-consul-services.sh" "deploy/infrastructure/register-consul-services.sh"
move_script "deploy_jarvis.sh" "deploy/environments/development.sh"

# Reorganize development scripts
echo -e "\n${YELLOW}=== Development Scripts ===${NC}"
move_script "start_all_services.sh" "dev/services/start-services.sh"

# Reorganize maintenance scripts
echo -e "\n${YELLOW}=== Maintenance Scripts ===${NC}"
move_script "docker-fix-infrastructure.sh" "utils/maintenance/docker-fix-infrastructure.sh"
move_script "fix-docker-issues.sh" "utils/maintenance/fix-docker-issues.sh"
move_script "maintenance/fix-unhealthy-services.sh" "maintenance/fix-unhealthy-services.sh"
move_script "maintenance/cleanup-excessive-changelogs.sh" "data/maintenance/cleanup-changelogs.sh"
move_script "maintenance/create-all-changelogs.sh" "data/maintenance/create-changelogs.sh"
move_script "maintenance/check-compliance.sh" "utils/validation/check-compliance.sh"
move_script "maintenance/fix-compliance-violations.py" "utils/validation/fix-compliance-violations.py"
move_script "maintenance/fix-mcp-bridge.py" "utils/integration/fix-mcp-bridge.py"

# Reorganize monitoring scripts
echo -e "\n${YELLOW}=== Monitoring Scripts ===${NC}"
move_script "monitoring/comprehensive_system_audit.sh" "monitoring/system-audit.sh"
move_script "monitoring/live_logs.sh" "monitoring/live-logs.sh"
move_script "monitoring/health-monitor-daemon.sh" "monitoring/health-monitor-daemon.sh"
move_script "monitoring/fix-unhealthy-services.sh" "maintenance/fix-unhealthy-services.sh"
move_script "monitoring/ollama-health-fix.sh" "maintenance/ollama-health-fix.sh"
move_script "monitoring/fix-ollama-semgrep.sh" "maintenance/fix-ollama-semgrep.sh"

# Reorganize testing scripts
echo -e "\n${YELLOW}=== Testing Scripts ===${NC}"
move_script "test_frontend_playwright.py" "test/e2e/test-frontend-playwright.py"
move_script "verify_all_components.py" "test/integration/verify-all-components.py"

# Reorganize security scripts
echo -e "\n${YELLOW}=== Security Scripts ===${NC}"
move_script "security/initialize_secrets.py" "deploy/security/initialize-secrets.py"

# Reorganize MCP-related scripts
echo -e "\n${YELLOW}=== MCP Integration Scripts ===${NC}"
move_script "mcp/fix-all-wrappers.sh" "utils/integration/mcp/fix-all-wrappers.sh"
move_script "mcp/fix-failing-servers.sh" "utils/integration/mcp/fix-failing-servers.sh"

echo -e "\n${GREEN}Script reorganization completed!${NC}"
