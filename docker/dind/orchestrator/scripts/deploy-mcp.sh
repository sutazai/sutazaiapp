#!/bin/bash
set -euo pipefail

# MCP Container Deployment Script for DinD Environment
# Deploys MCP containers within Docker-in-Docker orchestrator

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_DIR="${SCRIPT_DIR}/../mcp-manifests"
MANAGER_URL="${MCP_MANAGER_URL:-http://localhost:18081}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if MCP Manager is ready
check_manager_health() {
    log "Checking MCP Manager health..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "${MANAGER_URL}/health" > /dev/null 2>&1; then
            success "MCP Manager is healthy"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: MCP Manager not ready, waiting..."
        sleep 2
        ((attempt++))
    done
    
    error "MCP Manager failed to become healthy after $max_attempts attempts"
    return 1
}

# Convert YAML manifest to JSON for API
yaml_to_json() {
    local yaml_file="$1"
    python3 -c "
import yaml
import json
import sys

with open('$yaml_file', 'r') as f:
    data = yaml.safe_load(f)

# Extract container configuration from manifest
spec = data['spec']
mcp_config = {
    'name': spec['container_name'],
    'image': spec['image'],
    'ports': spec.get('ports', {}),
    'environment': spec.get('environment', {}),
    'volumes': spec.get('volumes', {}),
    'restart_policy': spec.get('restart_policy', 'unless-stopped')
}

print(json.dumps(mcp_config, indent=2))
"
}

# Deploy individual MCP container
deploy_mcp_container() {
    local manifest_file="$1"
    local mcp_name=$(basename "$manifest_file" .yml)
    
    log "Deploying MCP container from manifest: $manifest_file"
    
    # Convert YAML to JSON
    local json_config
    if ! json_config=$(yaml_to_json "$manifest_file"); then
        error "Failed to parse manifest: $manifest_file"
        return 1
    fi
    
    # Deploy via API
    local response
    if response=$(curl -sf -X POST \
        -H "Content-Type: application/json" \
        -d "$json_config" \
        "${MANAGER_URL}/containers" 2>/dev/null); then
        
        local status=$(echo "$response" | jq -r '.status')
        local container_id=$(echo "$response" | jq -r '.container_id // "unknown"')
        
        case "$status" in
            "deployed")
                success "MCP container deployed: $mcp_name (ID: ${container_id:0:12})"
                ;;
            "already_running")
                warn "MCP container already running: $mcp_name (ID: ${container_id:0:12})"
                ;;
            *)
                warn "Unexpected deployment status: $status"
                ;;
        esac
        return 0
    else
        error "Failed to deploy MCP container: $mcp_name"
        return 1
    fi
}

# Deploy all MCP containers
deploy_all_mcps() {
    log "Starting deployment of all MCP containers..."
    
    local deployed=0
    local failed=0
    
    for manifest in "${MANIFEST_DIR}"/*.yml; do
        if [ -f "$manifest" ]; then
            if deploy_mcp_container "$manifest"; then
                ((deployed++))
            else
                ((failed++))
            fi
        fi
    done
    
    log "Deployment summary: $deployed deployed, $failed failed"
    
    if [ $failed -gt 0 ]; then
        return 1
    fi
    return 0
}

# List deployed containers
list_containers() {
    log "Listing deployed MCP containers..."
    
    local response
    if response=$(curl -sf "${MANAGER_URL}/containers" 2>/dev/null); then
        echo "$response" | jq -r '.[] | "\(.name)\t\(.status)\t\(.image)\t\(.id[0:12])"' | \
            column -t -s $'\t' -N "NAME,STATUS,IMAGE,CONTAINER_ID"
    else
        error "Failed to list containers"
        return 1
    fi
}

# Cleanup orphaned containers
cleanup_orphaned() {
    log "Cleaning up orphaned MCP containers..."
    
    local response
    if response=$(curl -sf -X POST "${MANAGER_URL}/cleanup" 2>/dev/null); then
        local cleaned=$(echo "$response" | jq -r '.cleaned')
        local running=$(echo "$response" | jq -r '.running')
        
        success "Cleanup complete: $cleaned orphaned containers removed, $running currently running"
    else
        error "Failed to cleanup orphaned containers"
        return 1
    fi
}

# Get orchestrator status
get_status() {
    log "Getting MCP orchestrator status..."
    
    local response
    if response=$(curl -sf "${MANAGER_URL}/status" 2>/dev/null); then
        echo "$response" | jq '.'
    else
        error "Failed to get orchestrator status"
        return 1
    fi
}

# Main function
main() {
    local command="${1:-deploy}"
    
    case "$command" in
        "deploy")
            check_manager_health || exit 1
            deploy_all_mcps
            ;;
        "list")
            list_containers
            ;;
        "cleanup")
            cleanup_orphaned
            ;;
        "status")
            get_status
            ;;
        "health")
            check_manager_health
            ;;
        *)
            echo "Usage: $0 {deploy|list|cleanup|status|health}"
            echo ""
            echo "Commands:"
            echo "  deploy  - Deploy all MCP containers from manifests"
            echo "  list    - List currently deployed MCP containers"
            echo "  cleanup - Clean up orphaned containers"
            echo "  status  - Get orchestrator status"
            echo "  health  - Check manager health"
            exit 1
            ;;
    esac
}

# Check dependencies
command -v python3 >/dev/null 2>&1 || { error "python3 is required but not installed"; exit 1; }
command -v jq >/dev/null 2>&1 || { error "jq is required but not installed"; exit 1; }
command -v curl >/dev/null 2>&1 || { error "curl is required but not installed"; exit 1; }

# Run main function
main "$@"