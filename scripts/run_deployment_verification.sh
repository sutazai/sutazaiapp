#!/bin/bash
# SutazAI Deployment Verification Runner
# Testing QA Validator Agent
# Version: 1.0.0

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_banner() {
    echo -e "\n${CYAN}===============================================================================${NC}"
    echo -e "${CYAN}${BOLD}🔍 SutazAI Deployment Verification Runner${NC}"
    echo -e "${CYAN}Testing QA Validator Agent | Version 1.0.0${NC}"
    echo -e "${CYAN}===============================================================================${NC}\n"
}

check_python() {
    if command -v python3 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Python3 found: $(python3 --version)${NC}"
        return 0
    else
        echo -e "${RED}❌ Python3 not found${NC}"
        return 1
    fi
}

install_dependencies() {
    echo -e "\n${YELLOW}📦 Installing Python dependencies...${NC}"
    
    # Check if virtual environment exists
    if [[ ! -d "$PROJECT_ROOT/venv" ]]; then
        echo -e "${BLUE}Creating virtual environment...${NC}"
        python3 -m venv "$PROJECT_ROOT/venv"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Install base requirements if they exist
    if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        echo -e "${BLUE}Installing base requirements...${NC}"
        pip install -r "$PROJECT_ROOT/requirements.txt" >/dev/null 2>&1 || true
    fi
    
    # Install verification-specific requirements
    if [[ -f "$SCRIPT_DIR/requirements-verification.txt" ]]; then
        echo -e "${BLUE}Installing verification requirements...${NC}"
        pip install -r "$SCRIPT_DIR/requirements-verification.txt" >/dev/null 2>&1
    else
        echo -e "${BLUE}Installing required packages...${NC}"
        pip install aiohttp asyncpg redis psutil docker neo4j pyyaml >/dev/null 2>&1
    fi
    
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

run_quick_check() {
    echo -e "\n${YELLOW}🚀 Running quick deployment check...${NC}"
    "$SCRIPT_DIR/quick_deployment_check.sh"
    return $?
}

run_comprehensive_check() {
    echo -e "\n${YELLOW}🔬 Running comprehensive deployment verification...${NC}"
    
    # Activate virtual environment if it exists
    if [[ -d "$PROJECT_ROOT/venv" ]]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
    
    python3 "$SCRIPT_DIR/comprehensive_deployment_verification.py"
    return $?
}

show_help() {
    echo -e "${CYAN}SutazAI Deployment Verification Runner${NC}"
    echo -e ""
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 [option]"
    echo -e ""
    echo -e "${YELLOW}Options:${NC}"
    echo -e "  --quick, -q       Run quick shell-based verification only"
    echo -e "  --full, -f        Run comprehensive Python-based verification only"
    echo -e "  --both, -b        Run both quick and comprehensive (default)"
    echo -e "  --install, -i     Install dependencies only"
    echo -e "  --help, -h        Show this help message"
    echo -e ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  $0                # Run both verifications"
    echo -e "  $0 --quick        # Quick check only"
    echo -e "  $0 --full         # Comprehensive check only"
    echo -e "  $0 --install      # Install dependencies only"
}

main() {
    local mode="both"
    
    # Parse command line arguments
    case "${1:-}" in
        --quick|-q)
            mode="quick"
            ;;
        --full|-f)
            mode="full"
            ;;
        --both|-b)
            mode="both"
            ;;
        --install|-i)
            mode="install"
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        "")
            mode="both"
            ;;
        *)
            echo -e "${RED}❌ Invalid option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
    
    print_banner
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Check Python availability for comprehensive mode
    if [[ "$mode" == "full" || "$mode" == "both" || "$mode" == "install" ]]; then
        if ! check_python; then
            echo -e "${RED}❌ Python3 is required for comprehensive verification${NC}"
            if [[ "$mode" == "full" ]]; then
                exit 1
            else
                echo -e "${YELLOW}⚠️ Falling back to quick verification only${NC}"
                mode="quick"
            fi
        fi
    fi
    
    case "$mode" in
        "install")
            install_dependencies
            echo -e "\n${GREEN}✅ Dependencies installed successfully${NC}"
            exit 0
            ;;
        "quick")
            run_quick_check
            exit $?
            ;;
        "full")
            install_dependencies
            run_comprehensive_check
            exit $?
            ;;
        "both")
            echo -e "${BLUE}Running both quick and comprehensive verifications...${NC}"
            
            # Run quick check first
            echo -e "\n${CYAN}=== PHASE 1: QUICK VERIFICATION ===${NC}"
            quick_result=0
            run_quick_check || quick_result=$?
            
            # Run comprehensive check if Python is available
            echo -e "\n${CYAN}=== PHASE 2: COMPREHENSIVE VERIFICATION ===${NC}"
            comprehensive_result=0
            install_dependencies
            run_comprehensive_check || comprehensive_result=$?
            
            # Summary
            echo -e "\n${CYAN}===============================================================================${NC}"
            echo -e "${CYAN}${BOLD}📊 VERIFICATION SUMMARY${NC}"
            echo -e "${CYAN}===============================================================================${NC}"
            
            if [[ $quick_result -eq 0 ]]; then
                echo -e "${GREEN}✅ Quick verification: PASSED${NC}"
            else
                echo -e "${RED}❌ Quick verification: FAILED (exit code: $quick_result)${NC}"
            fi
            
            if [[ $comprehensive_result -eq 0 ]]; then
                echo -e "${GREEN}✅ Comprehensive verification: PASSED${NC}"
            else
                echo -e "${RED}❌ Comprehensive verification: FAILED (exit code: $comprehensive_result)${NC}"
            fi
            
            # Return the worst result
            if [[ $quick_result -ne 0 ]]; then
                exit $quick_result
            else
                exit $comprehensive_result
            fi
            ;;
    esac
}

# Handle script interruption
trap 'echo -e "\n${RED}Verification interrupted${NC}"; exit 130' INT TERM

# Run main function
main "$@"