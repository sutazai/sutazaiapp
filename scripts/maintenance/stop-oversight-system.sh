#!/bin/bash
# Stop SutazAI Human Oversight System

set -e

# Configuration

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_ROOT/var/run"

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

# Stop the oversight orchestrator
stop_orchestrator() {
    local pid_file="$PID_DIR/oversight_orchestrator.pid"
    
    if [ ! -f "$pid_file" ]; then
        warning "PID file not found: $pid_file"
        warning "Oversight system may not be running"
        return 1
    fi
    
    local pid=$(cat "$pid_file")
    
    if ! kill -0 "$pid" 2>/dev/null; then
        warning "Process $pid is not running"
        rm -f "$pid_file"
        return 1
    fi
    
    log "Stopping Oversight Orchestrator (PID: $pid)..."
    
    # Send SIGTERM first
    kill "$pid"
    
    # Wait for graceful shutdown
    local count=0
    while kill -0 "$pid" 2>/dev/null && [ $count -lt 30 ]; do
        sleep 1
        count=$((count + 1))
        if [ $((count % 5)) -eq 0 ]; then
            log "Waiting for graceful shutdown... ($count/30 seconds)"
        fi
    done
    
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        warning "Process did not stop gracefully, sending SIGKILL..."
        kill -9 "$pid"
        sleep 2
    fi
    
    # Clean up PID file
    rm -f "$pid_file"
    
    if kill -0 "$pid" 2>/dev/null; then
        error "Failed to stop process $pid"
        return 1
    else
        success "Oversight Orchestrator stopped successfully"
        return 0
    fi
}

# Main execution
main() {
    log "Stopping SutazAI Human Oversight System..."
    
    if stop_orchestrator; then
        success "SutazAI Human Oversight System stopped successfully"
    else
        error "Failed to stop oversight system properly"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Stop the SutazAI Human Oversight System"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --force        Force stop using SIGKILL"
        echo ""
        exit 0
        ;;
    --force)
        log "Force stopping SutazAI Human Oversight System..."
        pid_file="$PID_DIR/oversight_orchestrator.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid"
                success "Process $pid force killed"
            fi
            rm -f "$pid_file"
        fi
        success "Force stop completed"
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