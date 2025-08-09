#!/bin/bash
#
# Start Hygiene Monitoring System
# Purpose: Launch the complete real-time hygiene monitoring infrastructure
# Author: AI Observability and Monitoring Engineer
# Version: 1.0.0 - Production Startup Script
#

set -e  # Exit on any error

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
MONITORING_DIR="$PROJECT_ROOT/monitoring"
LOGS_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/run"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    mkdir -p "$LOGS_DIR"
    mkdir -p "$PID_DIR"
    mkdir -p "$MONITORING_DIR"
    success "Directories created successfully"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check required Python packages
    python3 -c "import asyncio, aiohttp, psutil, sqlite3, watchdog" 2>/dev/null || {
        warn "Installing required Python packages..."
        pip3 install aiohttp psutil watchdog aiofiles --quiet
    }
    
    success "Dependencies verified"
}

# Check if processes are already running
check_running_processes() {
    log "Checking for existing monitoring processes..."
    
    # Check for hygiene monitor backend
    if pgrep -f "hygiene-monitor-backend.py" > /dev/null; then
        warn "Hygiene monitor backend is already running"
        return 1
    fi
    
    # Check for agent orchestrator
    if pgrep -f "agent-orchestrator.py" > /dev/null; then
        warn "Agent orchestrator is already running"
        return 1
    fi
    
    return 0
}

# Stop existing processes
stop_processes() {
    log "Stopping existing monitoring processes..."
    
    # Stop hygiene monitor backend
    pkill -f "hygiene-monitor-backend.py" 2>/dev/null || true
    
    # Stop agent orchestrator
    pkill -f "agent-orchestrator.py" 2>/dev/null || true
    
    # Stop logging infrastructure
    pkill -f "logging-infrastructure.py" 2>/dev/null || true
    
    # Wait for processes to stop
    sleep 2
    
    success "Existing processes stopped"
}

# Start logging infrastructure
start_logging() {
    log "Starting logging infrastructure..."
    
    cd "$MONITORING_DIR"
    nohup python3 logging-infrastructure.py > "$LOGS_DIR/logging-infrastructure.log" 2>&1 &
    echo $! > "$PID_DIR/logging-infrastructure.pid"
    
    # Wait for logging to initialize
    sleep 2
    
    if pgrep -f "logging-infrastructure.py" > /dev/null; then
        success "Logging infrastructure started successfully"
        return 0
    else
        error "Failed to start logging infrastructure"
        return 1
    fi
}

# Start agent orchestrator
start_orchestrator() {
    log "Starting agent orchestrator..."
    
    cd "$MONITORING_DIR" 
    nohup python3 agent-orchestrator.py > "$LOGS_DIR/agent-orchestrator.log" 2>&1 &
    echo $! > "$PID_DIR/agent-orchestrator.pid"
    
    # Wait for orchestrator to initialize
    sleep 3
    
    if pgrep -f "agent-orchestrator.py" > /dev/null; then
        success "Agent orchestrator started successfully"
        return 0
    else
        error "Failed to start agent orchestrator"
        return 1
    fi
}

# Start hygiene monitor backend
start_backend() {
    log "Starting hygiene monitor backend..."
    
    cd "$MONITORING_DIR"
    nohup python3 hygiene-monitor-backend.py > "$LOGS_DIR/hygiene-monitor-backend.log" 2>&1 &
    echo $! > "$PID_DIR/hygiene-monitor-backend.pid"
    
    # Wait for backend to initialize and check if it's responding
    sleep 5
    
    if pgrep -f "hygiene-monitor-backend.py" > /dev/null; then
        # Test if the API is responding
        if curl -s http://localhost:8080/api/hygiene/status > /dev/null; then
            success "Hygiene monitor backend started successfully"
            success "API is responding on http://localhost:8080"
            return 0
        else
            warn "Backend process started but API is not responding yet"
            return 0
        fi
    else
        error "Failed to start hygiene monitor backend"
        return 1
    fi
}

# Verify all services are running
verify_services() {
    log "Verifying all services are running..."
    
    local all_running=true
    
    # Check logging infrastructure
    if ! pgrep -f "logging-infrastructure.py" > /dev/null; then
        error "Logging infrastructure is not running"
        all_running=false
    fi
    
    # Check agent orchestrator
    if ! pgrep -f "agent-orchestrator.py" > /dev/null; then
        error "Agent orchestrator is not running"
        all_running=false
    fi
    
    # Check hygiene monitor backend
    if ! pgrep -f "hygiene-monitor-backend.py" > /dev/null; then
        error "Hygiene monitor backend is not running"
        all_running=false
    fi
    
    if [ "$all_running" = true ]; then
        success "All monitoring services are running successfully"
        return 0
    else
        error "Some services failed to start"
        return 1
    fi
}

# Show service status
show_status() {
    echo ""
    log "=== Hygiene Monitoring System Status ==="
    echo ""
    
    echo "Service Status:"
    if pgrep -f "logging-infrastructure.py" > /dev/null; then
        echo -e "  ${GREEN}✓${NC} Logging Infrastructure"
    else
        echo -e "  ${RED}✗${NC} Logging Infrastructure"
    fi
    
    if pgrep -f "agent-orchestrator.py" > /dev/null; then
        echo -e "  ${GREEN}✓${NC} Agent Orchestrator"
    else
        echo -e "  ${RED}✗${NC} Agent Orchestrator"
    fi
    
    if pgrep -f "hygiene-monitor-backend.py" > /dev/null; then
        echo -e "  ${GREEN}✓${NC} Hygiene Monitor Backend"
    else
        echo -e "  ${RED}✗${NC} Hygiene Monitor Backend"
    fi
    
    echo ""
    echo "Endpoints:"
    echo "  Dashboard: file://$PROJECT_ROOT/dashboard/hygiene-monitor/index.html"
    echo "  API:       http://localhost:8080/api/hygiene/status"
    echo "  WebSocket: ws://localhost:8080/ws"
    echo ""
    
    echo "Log Files:"
    echo "  Main logs: $LOGS_DIR/"
    echo "  Backend:   $LOGS_DIR/hygiene-monitor-backend.log"
    echo "  Orchestrator: $LOGS_DIR/agent-orchestrator.log"
    echo "  Logging:   $LOGS_DIR/logging-infrastructure.log"
    echo ""
    
    echo "Process IDs:"
    if [ -f "$PID_DIR/hygiene-monitor-backend.pid" ]; then
        echo "  Backend PID:      $(cat $PID_DIR/hygiene-monitor-backend.pid)"
    fi
    if [ -f "$PID_DIR/agent-orchestrator.pid" ]; then
        echo "  Orchestrator PID: $(cat $PID_DIR/agent-orchestrator.pid)"
    fi
    if [ -f "$PID_DIR/logging-infrastructure.pid" ]; then
        echo "  Logging PID:      $(cat $PID_DIR/logging-infrastructure.pid)"
    fi
    echo ""
}

# Main execution
main() {
    echo ""
    log "=== Starting Hygiene Monitoring System ==="
    echo ""
    
    # Handle command line arguments
    case "${1:-start}" in
        "start")
            create_directories
            check_dependencies
            
            if check_running_processes; then
                log "No existing processes found, starting fresh..."
            else
                warn "Existing processes detected, stopping them first..."
                stop_processes
            fi
            
            # Start services in order
            if start_logging && start_orchestrator && start_backend; then
                sleep 3
                if verify_services; then
                    success "Hygiene Monitoring System started successfully!"
                    show_status
                    
                    log "You can now:"
                    log "1. Open the dashboard: file://$PROJECT_ROOT/dashboard/hygiene-monitor/index.html"
                    log "2. Test the API: curl http://localhost:8080/api/hygiene/status"
                    log "3. View logs: tail -f $LOGS_DIR/hygiene-monitor-backend.log"
                    log "4. Stop services: $0 stop"
                else
                    error "System failed to start properly"
                    exit 1
                fi
            else
                error "Failed to start all services"
                exit 1
            fi
            ;;
            
        "stop")
            log "Stopping Hygiene Monitoring System..."
            stop_processes
            
            # Clean up PID files
            rm -f "$PID_DIR"/*.pid
            
            success "Hygiene Monitoring System stopped"
            ;;
            
        "restart")
            log "Restarting Hygiene Monitoring System..."
            "$0" stop
            sleep 2
            "$0" start
            ;;
            
        "status")
            show_status
            ;;
            
        "logs")
            log "Tailing monitoring logs (Ctrl+C to stop)..."
            tail -f "$LOGS_DIR"/hygiene-monitor-backend.log
            ;;
            
        *)
            echo "Usage: $0 {start|stop|restart|status|logs}"
            echo ""
            echo "Commands:"
            echo "  start   - Start the monitoring system"
            echo "  stop    - Stop the monitoring system" 
            echo "  restart - Restart the monitoring system"
            echo "  status  - Show system status"
            echo "  logs    - Tail the backend logs"
            exit 1
            ;;
    esac
}

# Trap signals for graceful shutdown
trap 'log "Received signal, shutting down..."; stop_processes; exit 0' SIGTERM SIGINT

# Run main function
main "$@"