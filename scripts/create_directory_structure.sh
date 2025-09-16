#!/bin/bash
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Script: create_directory_structure.sh
# Purpose: Create Rule 7 compliant directory structure for scripts
# Author: Sutazai System
# Date: 2025-09-03
# Version: 1.0.0
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail
IFS=$'\n\t'

# Color codes for output
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

echo -e "${YELLOW}Creating Rule 7 compliant directory structure...${NC}"

# Create main directory structure
directories=(
    # Development directories
    "dev/setup"
    "dev/database"
    "dev/testing"
    "dev/services"
    "dev/cleanup"
    
    # Deployment directories
    "deploy/environments"
    "deploy/infrastructure"
    "deploy/containers"
    "deploy/database"
    "deploy/security"
    "deploy/monitoring"
    
    # Data management directories
    "data/backup"
    "data/migration"
    "data/processing"
    "data/maintenance"
    "data/sync"
    
    # Utility directories
    "utils/system"
    "utils/network"
    "utils/security"
    "utils/validation"
    "utils/maintenance"
    "utils/integration/mcp"  # For MCP wrappers
    
    # Testing directories
    "test/unit"
    "test/integration"
    "test/e2e"
    "test/performance"
    "test/security"
    "test/automation"
    
    # Monitoring directory
    "monitoring"
    
    # Maintenance directory
    "maintenance"
    
    # Template directory
    "templates/examples"
)

# Create all directories
for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}✓${NC} Created: $dir"
    else
        echo "  Already exists: $dir"
    fi
done

echo -e "${GREEN}Directory structure creation completed!${NC}"
