#!/bin/bash

# SutazAI Docker Cleanup Script
# Version: 1.0
# Date: 2025-08-05
# Description: Remove old, conflicting, and phantom Docker compose files

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[CLEANUP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

# Create backup directory
create_backup() {
    local backup_dir="archive/docker-compose-cleanup-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    echo "$backup_dir"
}

# List of old/problematic Docker compose files to remove
OLD_COMPOSE_FILES=(
    "docker-compose-optimized.yml"
    "docker-compose.agents-20.yml"
    "docker-compose.agents-critical-fixed.yml"
    "docker-compose.agents-deploy.yml"
    "docker-compose.agents-final.yml"
    "docker-compose.agents-fix.yml"
    "docker-compose.agents-fixed.yml"
    "docker-compose.agi.yml"
    "docker-compose.auth.yml"
    "docker-compose.autoscaling.yml"
    "docker-compose.critical-immediate.yml"
    "docker-compose.distributed-ai.yml"
    "docker-compose.distributed-ollama.yml"
    "docker-compose.distributed.yml"
    "docker-compose.external-integration.yml"
    "docker-compose.fusion.yml"
    "docker-compose.gpu.yml"
    "docker-compose.health-final.yml"
    "docker-compose.health-fix.yml"
    "docker-compose.health-fixed.yml"
    "docker-compose.health-override.yml"
    "docker-compose.healthfix-override.yml"
    "docker-compose.healthfix.yml"
    "docker-compose.hygiene-monitor.yml"
    "docker-compose.hygiene-standalone.yml"
    "docker-compose.infrastructure.yml"
    "docker-compose.jarvis-simple.yml"
    "docker-compose.jarvis.yml"
    "docker-compose.minimal.yml"
    "docker-compose.missing-agents-optimized.yml"
    "docker-compose.missing-agents.yml"
    "docker-compose.missing-services.yml"
    "docker-compose.network-secure.yml"
    "docker-compose.ollama-cluster-optimized.yml"
    "docker-compose.ollama-cluster.yml"
    "docker-compose.ollama-fix.yml"
    "docker-compose.ollama-optimized.yml"
    "docker-compose.optimized.yml"
    "docker-compose.orchestration-agents.yml"
    "docker-compose.phase1-activation.yml"
    "docker-compose.phase1-critical-activation.yml"
    "docker-compose.phase1-critical.yml"
    "docker-compose.phase2-specialized.yml"
    "docker-compose.phase3-auxiliary.yml"
    "docker-compose.resource-optimized.yml"
    "docker-compose.secure.yml"
    "docker-compose.security.yml"
    "docker-compose.self-healing-critical.yml"
    "docker-compose.self-healing.yml"
    "docker-compose.service-mesh.yml"
    "docker-compose.simple-health.yml"
)

main() {
    print_header "SUTAZAI DOCKER CLEANUP"
    
    # Create backup directory
    backup_dir=$(create_backup)
    print_status "Created backup directory: $backup_dir"
    
    # Stop any running containers from old compose files
    print_header "STOPPING OLD CONTAINERS"
    for compose_file in "${OLD_COMPOSE_FILES[@]}"; do
        if [ -f "$compose_file" ]; then
            print_status "Stopping containers from $compose_file"
            docker-compose -f "$compose_file" down --remove-orphans &>/dev/null || true
        fi
    done
    
    # Also stop from main compose files
    if [ -f "docker-compose.yml" ]; then
        print_status "Stopping containers from docker-compose.yml"
        docker-compose -f "docker-compose.yml" down --remove-orphans &>/dev/null || true
    fi
    
    if [ -f "docker-compose.agents.yml" ]; then
        print_status "Stopping containers from docker-compose.agents.yml"
        docker-compose -f "docker-compose.agents.yml" down --remove-orphans &>/dev/null || true
    fi
    
    if [ -f "docker-compose.monitoring.yml" ]; then
        print_status "Stopping containers from docker-compose.monitoring.yml"
        docker-compose -f "docker-compose.monitoring.yml" down --remove-orphans &>/dev/null || true
    fi
    
    # Move old compose files to backup
    print_header "ARCHIVING OLD COMPOSE FILES"
    files_moved=0
    
    for compose_file in "${OLD_COMPOSE_FILES[@]}"; do
        if [ -f "$compose_file" ]; then
            mv "$compose_file" "$backup_dir/"
            print_status "Archived: $compose_file"
            files_moved=$((files_moved + 1))
        fi
    done
    
    # Backup main compose files (but don't remove them yet)
    if [ -f "docker-compose.yml" ]; then
        cp "docker-compose.yml" "$backup_dir/docker-compose.yml.backup"
        print_status "Backed up: docker-compose.yml"
    fi
    
    if [ -f "docker-compose.agents.yml" ]; then
        cp "docker-compose.agents.yml" "$backup_dir/docker-compose.agents.yml.backup"
        print_status "Backed up: docker-compose.agents.yml"
    fi
    
    if [ -f "docker-compose.monitoring.yml" ]; then
        cp "docker-compose.monitoring.yml" "$backup_dir/docker-compose.monitoring.yml.backup"
        print_status "Backed up: docker-compose.monitoring.yml"
    fi
    
    # Clean up orphaned containers
    print_header "CLEANING UP ORPHANED CONTAINERS"
    orphaned_containers=$(docker ps -a --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
    if [ "$orphaned_containers" -gt 0 ]; then
        print_status "Found $orphaned_containers containers with 'sutazai-' prefix"
        docker ps -a --filter "name=sutazai-" --format "{{.Names}}" | xargs -r docker rm -f
        print_status "Removed orphaned containers"
    else
        print_status "No orphaned containers found"
    fi
    
    # Clean up orphaned volumes
    print_header "CLEANING UP ORPHANED VOLUMES"
    orphaned_volumes=$(docker volume ls --filter "name=sutazai" --format "{{.Name}}" | wc -l)
    if [ "$orphaned_volumes" -gt 0 ]; then
        print_warning "Found $orphaned_volumes volumes with 'sutazai' prefix"
        print_warning "These will NOT be automatically removed to preserve data"
        print_warning "If you want to remove them, run: docker volume prune"
    else
        print_status "No orphaned volumes found"
    fi
    
    # Clean up orphaned networks
    print_header "CLEANING UP ORPHANED NETWORKS"
    if docker network ls | grep -q "sutazai"; then
        # Don't remove the main sutazai-network as it's still needed
        print_status "Sutazai networks found - keeping main network for new deployment"
    fi
    
    # Remove unused images
    print_header "CLEANING UP UNUSED IMAGES"
    print_status "Removing dangling images..."
    docker image prune -f &>/dev/null || true
    print_status "Unused images cleaned up"
    
    # Summary
    print_header "CLEANUP SUMMARY"
    print_status "Archived $files_moved old Docker Compose files"
    print_status "Backup location: $backup_dir"
    print_status "Orphaned containers removed"
    print_status "System cleaned up"
    
    echo ""
    print_status "Your system is now ready for the consolidated deployment!"
    print_status "Run: ./deploy-consolidated.sh"
    
    # Show current Docker state
    print_header "CURRENT DOCKER STATE"
    echo "Running containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo -e "\nDocker networks:"
    docker network ls --filter "name=sutazai" --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}"
    
    echo -e "\nDocker volumes:"
    docker volume ls --filter "name=sutazai" --format "table {{.Name}}\t{{.Driver}}"
}

# Command line interface
case "${1:-cleanup}" in
    "cleanup"|"clean")
        main
        ;;
    "dry-run"|"preview")
        print_header "DRY RUN - NO CHANGES WILL BE MADE"
        echo "Files that would be archived:"
        for compose_file in "${OLD_COMPOSE_FILES[@]}"; do
            if [ -f "$compose_file" ]; then
                echo "  - $compose_file"
            fi
        done
        echo ""
        echo "Containers that would be removed:"
        docker ps -a --filter "name=sutazai-" --format "{{.Names}}" | sed 's/^/  - /'
        ;;
    "help"|"-h"|"--help")
        echo "SutazAI Docker Cleanup Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  cleanup     Perform cleanup (default)"
        echo "  dry-run     Preview what would be cleaned up"
        echo "  help        Show this help message"
        echo ""
        echo "This script will:"
        echo "  - Stop containers from old compose files"
        echo "  - Archive old Docker Compose files"
        echo "  - Remove orphaned containers"
        echo "  - Clean up unused Docker images"
        echo "  - Preserve data volumes (safe)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac