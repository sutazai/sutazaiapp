#!/bin/bash
#
# Hardware Optimizer Test Runner
# Purpose: Quick start script for running integration tests
# Usage: ./run_tests.sh [all|single|continuous|dashboard]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if agent is running
check_agent() {
    echo -e "${YELLOW}Checking if Hardware Optimizer agent is running...${NC}"
    
    if curl -s http://localhost:8116/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Agent is running${NC}"
        return 0
    else
        echo -e "${RED}✗ Agent is not running${NC}"
        echo "Please start the agent first:"
        echo "  cd /opt/sutazaiapp/agents/hardware-resource-optimizer"
        echo "  python3 app.py"
        return 1
    fi
}

# Run all integration tests
run_all_tests() {
    echo -e "${GREEN}Running all integration test scenarios...${NC}"
    python3 integration_test_suite.py
}

# Run single scenario
run_single_test() {
    echo -e "${GREEN}Available test scenarios:${NC}"
    echo "  - full_system  : Full system optimization"
    echo "  - storage      : Storage workflow"  
    echo "  - pressure     : Resource pressure"
    echo "  - docker       : Docker lifecycle"
    echo "  - concurrent   : Concurrent operations"
    echo "  - errors       : Error recovery"
    echo
    
    read -p "Enter scenario name: " scenario
    
    echo -e "${GREEN}Running $scenario scenario...${NC}"
    python3 integration_test_suite.py --scenario "$scenario"
}

# Run continuous validator
run_continuous() {
    echo -e "${GREEN}Starting continuous test validator...${NC}"
    echo "Tests will run every 60 minutes. Press Ctrl+C to stop."
    python3 continuous_validator.py --interval 60
}

# Run with dashboard
run_dashboard() {
    echo -e "${GREEN}Starting continuous validator with dashboard...${NC}"
    echo "Dashboard will be available at: http://localhost:8117"
    echo "Press Ctrl+C to stop."
    python3 continuous_validator.py --interval 60 --dashboard
}

# Install as service
install_service() {
    echo -e "${GREEN}Installing continuous test service...${NC}"
    
    if [ "$EUID" -ne 0 ]; then 
        echo -e "${RED}Please run as root to install service${NC}"
        exit 1
    fi
    
    cp hardware-optimizer-tests.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable hardware-optimizer-tests.service
    systemctl start hardware-optimizer-tests.service
    
    echo -e "${GREEN}✓ Service installed and started${NC}"
    echo "View logs: journalctl -u hardware-optimizer-tests -f"
}

# Show test reports
show_reports() {
    echo -e "${GREEN}Recent test reports:${NC}"
    ls -la continuous_test_reports/*.json 2>/dev/null | tail -10 || echo "No reports found"
    
    if [ -f continuous_test_reports/summary.txt ]; then
        echo
        echo -e "${GREEN}Latest summary:${NC}"
        tail -50 continuous_test_reports/summary.txt
    fi
}

# Main menu
main() {
    if ! check_agent; then
        exit 1
    fi
    
    echo
    echo -e "${GREEN}Hardware Optimizer Test Suite${NC}"
    echo "=============================="
    echo
    
    case "${1:-menu}" in
        all)
            run_all_tests
            ;;
        single)
            run_single_test
            ;;
        continuous)
            run_continuous
            ;;
        dashboard)
            run_dashboard
            ;;
        install)
            install_service
            ;;
        reports)
            show_reports
            ;;
        menu|*)
            echo "Options:"
            echo "  1) Run all test scenarios"
            echo "  2) Run single scenario"
            echo "  3) Run continuous tests (no dashboard)"
            echo "  4) Run continuous tests with dashboard"
            echo "  5) Install as system service"
            echo "  6) Show test reports"
            echo "  7) Exit"
            echo
            
            read -p "Select option [1-7]: " choice
            
            case $choice in
                1) run_all_tests ;;
                2) run_single_test ;;
                3) run_continuous ;;
                4) run_dashboard ;;
                5) install_service ;;
                6) show_reports ;;
                7) exit 0 ;;
                *) echo -e "${RED}Invalid option${NC}" ;;
            esac
            ;;
    esac
}

# Run main function
main "$@"