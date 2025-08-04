#!/bin/bash
# Purpose: Run standalone hygiene check without interfering with main app
# Usage: ./run-hygiene-check.sh [--report-only]
# Requires: Docker, Docker Compose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Standalone Hygiene Check ===${NC}"
echo -e "${BLUE}This runs independently from the main Sutazai application${NC}\n"

# Check if main app is running (just for info)
if docker ps | grep -q "sutazai-backend"; then
    echo -e "${YELLOW}Note: Main Sutazai application is running${NC}"
    echo -e "${YELLOW}Hygiene check will not interfere with it${NC}\n"
fi

# Create reports directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/reports"

# Function to run hygiene scan
run_hygiene_scan() {
    echo -e "${BLUE}Starting hygiene scan...${NC}"
    
    # Run the hygiene scanner
    docker-compose -f "$PROJECT_ROOT/docker-compose.hygiene-standalone.yml" \
        run --rm hygiene-scanner
    
    # Check exit code
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Hygiene scan completed successfully${NC}"
    else
        echo -e "${RED}✗ Hygiene issues detected${NC}"
    fi
}

# Function to start report viewer
start_report_viewer() {
    echo -e "\n${BLUE}Starting report viewer...${NC}"
    
    docker-compose -f "$PROJECT_ROOT/docker-compose.hygiene-standalone.yml" \
        up -d hygiene-reporter
    
    echo -e "${GREEN}✓ Report viewer available at: http://localhost:9080${NC}"
    echo -e "${YELLOW}  Latest report: http://localhost:9080/hygiene-report.html${NC}"
}

# Function to stop report viewer
stop_report_viewer() {
    echo -e "\n${BLUE}Stopping report viewer...${NC}"
    
    docker-compose -f "$PROJECT_ROOT/docker-compose.hygiene-standalone.yml" \
        down
    
    echo -e "${GREEN}✓ Report viewer stopped${NC}"
}

# Function to run rule validation
run_rule_validation() {
    echo -e "\n${BLUE}Running CLAUDE.md rule validation...${NC}"
    
    docker-compose -f "$PROJECT_ROOT/docker-compose.hygiene-standalone.yml" \
        run --rm hygiene-validator
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Rule validation completed${NC}"
    else
        echo -e "${YELLOW}⚠ Some rules may need attention${NC}"
    fi
}

# Main execution
case "${1:-scan}" in
    scan)
        run_hygiene_scan
        start_report_viewer
        echo -e "\n${BLUE}To stop the report viewer, run:${NC}"
        echo -e "  $0 stop"
        ;;
    
    validate)
        run_rule_validation
        ;;
    
    full)
        run_hygiene_scan
        run_rule_validation
        start_report_viewer
        echo -e "\n${BLUE}Full hygiene check complete!${NC}"
        ;;
    
    report-only)
        start_report_viewer
        ;;
    
    stop)
        stop_report_viewer
        ;;
    
    clean)
        echo -e "${BLUE}Cleaning up hygiene containers and volumes...${NC}"
        docker-compose -f "$PROJECT_ROOT/docker-compose.hygiene-standalone.yml" \
            down -v --remove-orphans
        echo -e "${GREEN}✓ Cleanup complete${NC}"
        ;;
    
    *)
        echo "Usage: $0 {scan|validate|full|report-only|stop|clean}"
        echo ""
        echo "Commands:"
        echo "  scan        - Run hygiene scan and show report (default)"
        echo "  validate    - Validate CLAUDE.md rules compliance"
        echo "  full        - Run both scan and validation"
        echo "  report-only - Just start the report viewer"
        echo "  stop        - Stop the report viewer"
        echo "  clean       - Remove all hygiene containers and volumes"
        exit 1
        ;;
esac

# Show recent reports
if [ -d "$PROJECT_ROOT/reports" ] && [ "$(ls -A $PROJECT_ROOT/reports 2>/dev/null)" ]; then
    echo -e "\n${BLUE}Recent reports:${NC}"
    ls -lt "$PROJECT_ROOT/reports" | head -6
fi