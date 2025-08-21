#!/bin/bash
# Unified Cleanup Script - Consolidates 21 cleanup scripts
# Created: 2025-08-21
# Replaces: emergency_container_cleanup.sh, cleanup-duplicates.sh, ultra_cleanup_commands.sh, etc.

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup modes
MODE=${1:-safe}  # safe, aggressive, emergency
DRY_RUN=${DRY_RUN:-false}

echo -e "${GREEN}=== Unified Cleanup Script ===${NC}"
echo "Mode: $MODE | Dry Run: $DRY_RUN"

# Function: Clean Docker resources
cleanup_docker() {
    echo -e "${YELLOW}Cleaning Docker resources...${NC}"
    
    if [ "$MODE" = "emergency" ] || [ "$MODE" = "aggressive" ]; then
        # Stop all containers except critical
        docker ps -q | grep -v -E "postgres|redis|neo4j|backend|frontend" | xargs -r docker stop
        
        # Remove stopped containers
        docker container prune -f
        
        # Clean unused images
        docker image prune -a -f --filter "until=24h"
        
        # Clean volumes (careful!)
        if [ "$MODE" = "aggressive" ]; then
            docker volume prune -f
        fi
        
        # Clean networks
        docker network prune -f
    else
        # Safe cleanup
        docker container prune -f
        docker image prune -f
    fi
    
    echo "Docker cleanup complete. Freed: $(docker system df | grep 'Total reclamation' | awk '{print $3}')"
}

# Function: Clean Python cache
cleanup_python() {
    echo -e "${YELLOW}Cleaning Python cache...${NC}"
    
    find /opt/sutazaiapp -type d -name "__pycache__" ! -path "*/venv/*" -exec rm -rf {} + 2>/dev/null || true
    find /opt/sutazaiapp -name "*.pyc" ! -path "*/venv/*" -delete 2>/dev/null || true
    find /opt/sutazaiapp -name "*.pyo" ! -path "*/venv/*" -delete 2>/dev/null || true
    find /opt/sutazaiapp -name ".pytest_cache" ! -path "*/venv/*" -exec rm -rf {} + 2>/dev/null || true
    
    echo "Python cache cleaned"
}

# Function: Clean Node modules
cleanup_node() {
    echo -e "${YELLOW}Cleaning Node artifacts...${NC}"
    
    # Clean node_modules in non-critical locations
    if [ "$MODE" = "aggressive" ]; then
        find /opt/sutazaiapp -name "node_modules" -type d ! -path "*/frontend/*" ! -path "*/backend/*" -exec rm -rf {} + 2>/dev/null || true
    fi
    
    # Clean npm cache
    npm cache clean --force 2>/dev/null || true
    
    echo "Node artifacts cleaned"
}

# Function: Clean logs
cleanup_logs() {
    echo -e "${YELLOW}Cleaning old logs...${NC}"
    
    # Rotate and compress logs
    find /opt/sutazaiapp -name "*.log" -size +100M -exec gzip {} \; 2>/dev/null || true
    
    # Remove old logs
    find /opt/sutazaiapp -name "*.log" -mtime +7 -delete 2>/dev/null || true
    find /opt/sutazaiapp -name "*.log.gz" -mtime +30 -delete 2>/dev/null || true
    
    echo "Logs cleaned"
}

# Function: Clean temporary files
cleanup_temp() {
    echo -e "${YELLOW}Cleaning temporary files...${NC}"
    
    find /opt/sutazaiapp -name "*.tmp" -o -name "*.temp" -o -name "*~" -o -name "*.bak" -o -name "*.old" | \
        grep -v -E "venv|node_modules" | xargs -r rm -f 2>/dev/null || true
    
    # Clean /tmp
    find /tmp -mtime +1 -name "mcp-*" -o -name "sutazai-*" | xargs -r rm -rf 2>/dev/null || true
    
    echo "Temporary files cleaned"
}

# Function: Report disk usage
report_usage() {
    echo -e "${GREEN}=== Disk Usage Report ===${NC}"
    df -h /opt/sutazaiapp
    echo ""
    echo "Top 10 space consumers:"
    du -h /opt/sutazaiapp --max-depth=2 2>/dev/null | sort -rh | head -10
}

# Main execution
main() {
    if [ "$DRY_RUN" = "true" ]; then
        echo -e "${YELLOW}DRY RUN MODE - No changes will be made${NC}"
        return 0
    fi
    
    # Record initial disk usage
    INITIAL_USAGE=$(df /opt/sutazaiapp | tail -1 | awk '{print $3}')
    
    # Execute cleanup functions based on mode
    cleanup_docker
    cleanup_python
    cleanup_node
    cleanup_logs
    cleanup_temp
    
    # Calculate space freed
    FINAL_USAGE=$(df /opt/sutazaiapp | tail -1 | awk '{print $3}')
    FREED=$((INITIAL_USAGE - FINAL_USAGE))
    
    echo -e "${GREEN}=== Cleanup Complete ===${NC}"
    echo "Space freed: $((FREED / 1024)) MB"
    
    # Show final report
    report_usage
}

# Run main function
main