#!/bin/bash
# title        :stop_sutazai.sh
# description  :This script stops all SutazAI services using Docker Compose
# author       :SutazAI Team
# version      :3.0
# usage        :bash scripts/stop_sutazai.sh [--force] [--remove-volumes]

# Change to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Parse command-line arguments
FORCE=false
REMOVE_VOLUMES=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --force) FORCE=true; shift ;;
        --remove-volumes) REMOVE_VOLUMES=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log "Docker is not running. Services may already be stopped."
    exit 0
fi

log "Stopping SutazAI services..."

# Stop all services
if [ "$REMOVE_VOLUMES" = true ]; then
    log "Stopping services and removing volumes..."
    docker compose down -v
elif [ "$FORCE" = true ]; then
    log "Force stopping services..."
    docker compose down --remove-orphans
else
    log "Stopping services gracefully..."
    docker compose down
fi

# Show remaining containers
REMAINING=$(docker ps -q --filter "name=sutazai-" | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    log "WARNING: $REMAINING SutazAI containers are still running"
    docker ps --filter "name=sutazai-"
    
    if [ "$FORCE" = true ]; then
        log "Force stopping remaining containers..."
        docker ps -q --filter "name=sutazai-" | xargs -r docker stop
        docker ps -q --filter "name=sutazai-" | xargs -r docker rm
    fi
else
    log "All SutazAI services stopped successfully"
fi

log ""
log "To restart services, run: ./scripts/start_sutazai.sh"
log "To deploy complete system, run: ./scripts/deploy_complete_system.sh"

exit 0