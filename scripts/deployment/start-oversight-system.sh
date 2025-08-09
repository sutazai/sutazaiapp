#!/bin/bash
# Start SutazAI Human Oversight System
# This script starts all oversight components

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OVERSIGHT_DIR="$PROJECT_ROOT/backend/oversight"
LOG_DIR="$PROJECT_ROOT/logs/oversight"
PID_DIR="$PROJECT_ROOT/var/run"

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"
mkdir -p "$OVERSIGHT_DIR/reports"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    log "Python version: $(python3 --version)"
}

# Check and install required Python packages
check_dependencies() {
    log "Checking Python dependencies..."
    
    # Required packages
    local packages=(
        "aiohttp"
        "aiohttp-cors"
        "sqlite3"
        "jinja2"
        "matplotlib"
        "seaborn"
        "pandas"
        "plotly"
        "streamlit"
    )
    
    for package in "${packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            warning "Installing missing package: $package"
            pip3 install "$package" || {
                error "Failed to install $package"
                exit 1
            }
        fi
    done
    
    success "All dependencies are satisfied"
}

# Check if oversight system is already running
check_running() {
    if [ -f "$PID_DIR/oversight_orchestrator.pid" ]; then
        local pid=$(cat "$PID_DIR/oversight_orchestrator.pid")
        if kill -0 "$pid" 2>/dev/null; then
            error "Oversight system is already running (PID: $pid)"
            echo "Use './scripts/stop-oversight-system.sh' to stop it first"
            exit 1
        else
            warning "Removing stale PID file"
            rm -f "$PID_DIR/oversight_orchestrator.pid"
        fi
    fi
}

# Validate configuration files
validate_config() {
    log "Validating configuration files..."
    
    # Check main config
    if [ ! -f "$OVERSIGHT_DIR/config.json" ]; then
        error "Main configuration file not found: $OVERSIGHT_DIR/config.json"
        exit 1
    fi
    
    # Validate JSON syntax
    if ! python3 -m json.tool "$OVERSIGHT_DIR/config.json" > /dev/null; then
        error "Invalid JSON in main configuration file"
        exit 1
    fi
    
    # Check alert config
    if [ ! -f "$OVERSIGHT_DIR/alert_config.json" ]; then
        error "Alert configuration file not found: $OVERSIGHT_DIR/alert_config.json"
        exit 1
    fi
    
    # Validate alert config JSON
    if ! python3 -m json.tool "$OVERSIGHT_DIR/alert_config.json" > /dev/null; then
        error "Invalid JSON in alert configuration file"
        exit 1
    fi
    
    success "Configuration files validated"
}

# Check database directory permissions
check_permissions() {
    log "Checking permissions..."
    
    if [ ! -w "$OVERSIGHT_DIR" ]; then
        error "No write permission for oversight directory: $OVERSIGHT_DIR"
        exit 1
    fi
    
    if [ ! -w "$LOG_DIR" ]; then
        error "No write permission for log directory: $LOG_DIR"
        exit 1
    fi
    
    success "Permissions OK"
}

# Initialize database if needed
init_database() {
    log "Initializing oversight database..."
    
    python3 -c "
import sqlite3
import os
from pathlib import Path

db_path = Path('$OVERSIGHT_DIR/oversight.db')
if not db_path.exists():
    print('Creating new oversight database...')
    # The database will be created automatically when the system starts
else:
    print('Oversight database already exists')
"
    
    success "Database initialization complete"
}

# Start the oversight orchestrator
start_orchestrator() {
    log "Starting SutazAI Human Oversight Orchestrator..."
    
    cd "$PROJECT_ROOT"
    
    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Start orchestrator in background
    nohup python3 -m backend.oversight.oversight_orchestrator \
        --config "$OVERSIGHT_DIR/config.json" \
        --log-level INFO \
        > "$LOG_DIR/orchestrator.log" 2>&1 &
    
    local pid=$!
    echo $pid > "$PID_DIR/oversight_orchestrator.pid"
    
    # Wait a moment and check if it's still running
    sleep 3
    if kill -0 "$pid" 2>/dev/null; then
        success "Oversight Orchestrator started successfully (PID: $pid)"
        log "Dashboard will be available at: http://localhost:8095"
        log "Logs are being written to: $LOG_DIR/"
    else
        error "Failed to start Oversight Orchestrator"
        cat "$LOG_DIR/orchestrator.log"
        exit 1
    fi
}

# Display system information
show_info() {
    echo ""
    echo -e "${GREEN}=== SutazAI Human Oversight System ===${NC}"
    echo ""
    echo "📊 Dashboard URL:     http://localhost:8095"
    echo "📁 Configuration:     $OVERSIGHT_DIR/config.json"
    echo "📁 Alert Config:      $OVERSIGHT_DIR/alert_config.json"
    echo "📁 Database:          $OVERSIGHT_DIR/oversight.db"
    echo "📁 Reports Directory: $OVERSIGHT_DIR/reports/"
    echo "📁 Log Directory:     $LOG_DIR/"
    echo "📁 PID File:          $PID_DIR/oversight_orchestrator.pid"
    echo ""
    echo -e "${BLUE}Key Features:${NC}"
    echo "• Real-time monitoring of 69 AI agents"
    echo "• Human intervention controls (pause/resume/override)"
    echo "• Emergency stop mechanisms"
    echo "• Compliance reporting (GDPR, HIPAA, AI Ethics, etc.)"
    echo "• Alert and notification system"
    echo "• Comprehensive audit trail"
    echo "• Performance monitoring and analytics"
    echo ""
    echo -e "${YELLOW}Available Commands:${NC}"
    echo "• View logs:          tail -f $LOG_DIR/orchestrator.log"
    echo "• Stop system:        ./scripts/stop-oversight-system.sh"
    echo "• Check status:       ./scripts/check-oversight-status.sh"
    echo "• Generate report:    ./scripts/generate-compliance-report.sh"
    echo ""
}

# Cleanup function
cleanup() {
    if [ -f "$PID_DIR/oversight_orchestrator.pid" ]; then
        local pid=$(cat "$PID_DIR/oversight_orchestrator.pid")
        if kill -0 "$pid" 2>/dev/null; then
            warning "Stopping oversight system due to error..."
            kill "$pid"
            rm -f "$PID_DIR/oversight_orchestrator.pid"
        fi
    fi
}

# Set up error handling
trap cleanup EXIT

# Main execution
main() {
    log "Starting SutazAI Human Oversight System initialization..."
    
    check_python
    check_dependencies
    check_running
    validate_config
    check_permissions
    init_database
    start_orchestrator
    show_info
    
    success "SutazAI Human Oversight System started successfully!"
    
    # Don't cleanup on successful exit
    trap - EXIT
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Start the SutazAI Human Oversight System"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --check        Check if system is running"
        echo "  --force        Force start (stop existing instance)"
        echo ""
        exit 0
        ;;
    --check)
        if [ -f "$PID_DIR/oversight_orchestrator.pid" ]; then
            local pid=$(cat "$PID_DIR/oversight_orchestrator.pid")
            if kill -0 "$pid" 2>/dev/null; then
                success "Oversight system is running (PID: $pid)"
                exit 0
            else
                error "Oversight system is not running (stale PID file)"
                exit 1
            fi
        else
            error "Oversight system is not running"
            exit 1
        fi
        ;;
    --force)
        if [ -f "$PID_DIR/oversight_orchestrator.pid" ]; then
            local pid=$(cat "$PID_DIR/oversight_orchestrator.pid")
            if kill -0 "$pid" 2>/dev/null; then
                warning "Stopping existing oversight system..."
                kill "$pid"
                sleep 5
                rm -f "$PID_DIR/oversight_orchestrator.pid"
            fi
        fi
        main
        ;;
    "")
        main
        ;;
    *)
        error "Unknown option: $1"
        error "Use '$0 --help' for usage information"
        exit 1
        ;;
esac