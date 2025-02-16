#!/bin/bash

# Enhanced Pre-Deployment Checklist for SutazAI

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/media/ai/SutazAI_Storage/SutazAI/v1"

# Logging
LOG_FILE="$HOME/sutazai_predeploy_checklist.log"
RECOMMENDATIONS_FILE="$HOME/sutazai_deployment_recommendations.txt"

# Clear previous logs
> "$LOG_FILE"
> "$RECOMMENDATIONS_FILE"

# Logging function
log_message() {
    local level="$1"
    local message="$2"
    local color=""

    case "$level" in
        "INFO") color="$BLUE" ;;
        "WARN") color="$YELLOW" ;;
        "ERROR") color="$RED" ;;
        "SUCCESS") color="$GREEN" ;;
        *) color="$NC" ;;
    esac

    echo -e "${color}[$level] $message${NC}" | tee -a "$LOG_FILE"
}

# Recommendation function
add_recommendation() {
    echo "- $1" >> "$RECOMMENDATIONS_FILE"
}

check_os_compatibility() {
    log_message "INFO" "Checking Operating System Compatibility..."
    
    OS=$(uname -s)
    DISTRO=$(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
    
    if [[ "$OS" != "Linux" ]]; then
        log_message "ERROR" "Unsupported operating system: $OS"
        add_recommendation "Consider using a Linux-based system for optimal compatibility"
        return 1
    fi
    
    log_message "SUCCESS" "Operating System: $DISTRO"
}

check_docker() {
    log_message "INFO" "Checking Docker Installation..."
    
    if ! command -v docker &> /dev/null; then
        log_message "ERROR" "Docker is not installed"
        add_recommendation "Install Docker: sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io"
        return 1
    fi
    
    docker info > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Docker daemon is not running"
        add_recommendation "Start Docker daemon: sudo systemctl start docker"
        return 1
    fi
    
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | sed 's/,//')
    log_message "SUCCESS" "Docker installed (version $DOCKER_VERSION)"
}

check_docker_compose() {
    log_message "INFO" "Checking Docker Compose..."
    
    if ! command -v docker-compose &> /dev/null; then
        log_message "WARN" "Docker Compose not found"
        add_recommendation "Install Docker Compose: sudo curl -L 'https://github.com/docker/compose/releases/latest/download/docker-compose-Linux-x86_64' -o /usr/local/bin/docker-compose"
        return 1
    fi
    
    COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | sed 's/,//')
    log_message "SUCCESS" "Docker Compose installed (version $COMPOSE_VERSION)"
}

check_python_version() {
    log_message "INFO" "Checking Python Version..."
    
    REQUIRED_PYTHON_VERSION="3.11"
    CURRENT_PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    
    if [[ "$(printf '%s\n' "$REQUIRED_PYTHON_VERSION" "$CURRENT_PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_PYTHON_VERSION" ]]; then
        log_message "ERROR" "Python version must be >= $REQUIRED_PYTHON_VERSION. Current: $CURRENT_PYTHON_VERSION"
        add_recommendation "Install Python $REQUIRED_PYTHON_VERSION using pyenv or from source"
        return 1
    fi
    
    log_message "SUCCESS" "Python version is compatible: $CURRENT_PYTHON_VERSION"
}

check_dependencies() {
    log_message "INFO" "Checking Project Dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        log_message "WARN" "Virtual environment not found. Creating..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Check pip requirements
    pip install -r requirements-prod.txt --dry-run > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Dependency installation failed"
        add_recommendation "Manually run: pip install -r requirements-prod.txt"
        return 1
    fi
    
    log_message "SUCCESS" "All dependencies are resolvable"
}

check_environment_file() {
    log_message "INFO" "Checking Environment Configuration..."
    
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_message "WARN" "Creating .env file from example..."
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    fi
    
    # Basic .env validation
    if ! grep -q "SECRET_KEY" "$PROJECT_ROOT/.env"; then
        log_message "ERROR" "Missing SECRET_KEY in .env file"
        add_recommendation "Manually add SECRET_KEY to .env file or use config.py to generate"
        return 1
    fi
    
    log_message "SUCCESS" "Environment configuration looks good"
}

check_system_resources() {
    log_message "INFO" "Checking System Resources..."
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    if [ "$CPU_CORES" -lt 4 ]; then
        log_message "WARN" "Less than 4 CPU cores detected"
        add_recommendation "Consider upgrading system with more CPU cores for better performance"
    fi
    
    # Check available memory
    TOTAL_MEMORY=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEMORY" -lt 8 ]; then
        log_message "WARN" "Less than 8GB RAM detected"
        add_recommendation "Consider adding more RAM for optimal system performance"
    fi
    
    log_message "SUCCESS" "System resources check completed"
}

main() {
    log_message "INFO" "Starting SutazAI Pre-Deployment Checklist"
    
    # Run all checks
    check_os_compatibility
    check_docker
    check_docker_compose
    check_python_version
    check_dependencies
    check_environment_file
    check_system_resources
    
    # Check if any checks failed
    if [ $? -eq 0 ]; then
        log_message "SUCCESS" "Pre-Deployment Checklist Passed Successfully!"
        
        # Display recommendations if any
        if [ -s "$RECOMMENDATIONS_FILE" ]; then
            echo -e "\n${YELLOW}Recommendations:${NC}"
            cat "$RECOMMENDATIONS_FILE"
        fi
        
        exit 0
    else
        log_message "ERROR" "Pre-Deployment Checklist Failed. Please resolve the issues."
        
        # Display recommendations
        echo -e "\n${YELLOW}Recommendations:${NC}"
        cat "$RECOMMENDATIONS_FILE"
        
        exit 1
    fi
}

main