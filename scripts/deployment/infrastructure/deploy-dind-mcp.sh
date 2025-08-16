#!/bin/bash
set -euo pipefail

# Master deployment script for Docker-in-Docker MCP orchestration
# Implements proper container isolation and management

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
DIND_DIR="${PROJECT_ROOT}/docker/dind"

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

# Check if running in correct environment
check_environment() {
    log "Checking deployment environment..."
    
    # Check if we're in the correct directory structure
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        error "Project root not found at: $PROJECT_ROOT"
        return 1
    fi
    
    if [[ ! -d "$DIND_DIR" ]]; then
        error "DinD directory not found at: $DIND_DIR"
        return 1
    fi
    
    # Check if we're not already running inside a container
    if [[ -f /.dockerenv ]]; then
        warn "Running inside a container - this may cause issues with DinD"
    fi
    
    success "Environment check passed"
}

# Clean up existing orphaned containers
cleanup_orphaned_containers() {
    log "Cleaning up orphaned MCP containers..."
    
    # Find containers with MCP-related names or labels
    local orphaned_containers
    orphaned_containers=$(docker ps -a --filter "name=mcp" --format "{{.Names}}" 2>/dev/null || true)
    
    if [[ -n "$orphaned_containers" ]]; then
        warn "Found orphaned MCP containers:"
        echo "$orphaned_containers"
        
        read -p "Remove these orphaned containers? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$orphaned_containers" | xargs -r docker rm -f
            success "Orphaned containers removed"
        else
            warn "Orphaned containers left running - may cause conflicts"
        fi
    else
        log "No orphaned MCP containers found"
    fi
}

# Pre-flight checks
preflight_checks() {
    log "Running pre-flight checks..."
    
    # Check disk space
    local available_space
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local required_space=$((2 * 1024 * 1024))  # 2GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        error "Insufficient disk space. Required: 2GB, Available: $((available_space / 1024 / 1024))GB"
        return 1
    fi
    
    # Check memory
    local available_memory
    available_memory=$(free -m | awk 'NR==2{print $7}')
    local required_memory=2048  # 2GB
    
    if [[ $available_memory -lt $required_memory ]]; then
        warn "Low available memory. Required: ${required_memory}MB, Available: ${available_memory}MB"
    fi
    
    # Check Docker resources
    local docker_info
    if ! docker_info=$(docker system info 2>/dev/null); then
        error "Cannot get Docker system information"
        return 1
    fi
    
    success "Pre-flight checks passed"
}

# Deploy DinD infrastructure
deploy_dind_infrastructure() {
    log "Deploying Docker-in-Docker infrastructure..."
    
    # Make scripts executable
    chmod +x "${DIND_DIR}/orchestrator/scripts"/*.sh
    
    # Setup and start DinD environment
    if "${DIND_DIR}/orchestrator/scripts/setup-dind.sh" start; then
        success "DinD infrastructure deployed successfully"
    else
        error "Failed to deploy DinD infrastructure"
        return 1
    fi
    
    # Wait for services to stabilize
    log "Waiting for services to stabilize..."
    sleep 10
    
    # Verify deployment
    if "${DIND_DIR}/orchestrator/scripts/setup-dind.sh" test; then
        success "DinD infrastructure verification passed"
    else
        error "DinD infrastructure verification failed"
        return 1
    fi
}

# Deploy MCP containers within DinD
deploy_mcp_containers() {
    log "Deploying MCP containers within DinD environment..."
    
    # Deploy MCP containers
    if "${DIND_DIR}/orchestrator/scripts/deploy-mcp.sh" deploy; then
        success "MCP containers deployed successfully"
    else
        error "Failed to deploy MCP containers"
        return 1
    fi
    
    # List deployed containers
    log "Currently deployed MCP containers:"
    "${DIND_DIR}/orchestrator/scripts/deploy-mcp.sh" list
}

# Setup service discovery and mesh integration
setup_mesh_integration() {
    log "Setting up service discovery and mesh integration..."
    
    # Register DinD MCP services with Consul
    local consul_url="http://localhost:10006"
    
    # Check if Consul is available
    if ! curl -sf "${consul_url}/v1/status/leader" >/dev/null 2>&1; then
        warn "Consul not available - service discovery will be limited"
        return 0
    fi
    
    # Register MCP Orchestrator
    curl -sf -X PUT "${consul_url}/v1/agent/service/register" \
        -d '{
            "ID": "mcp-orchestrator",
            "Name": "mcp-orchestrator",
            "Tags": ["mcp", "dind", "orchestration"],
            "Address": "localhost",
            "Port": 18080,
            "Check": {
                "HTTP": "http://localhost:18081/health",
                "Interval": "30s"
            }
        }' || warn "Failed to register MCP orchestrator with Consul"
    
    # Register MCP Manager
    curl -sf -X PUT "${consul_url}/v1/agent/service/register" \
        -d '{
            "ID": "mcp-manager",
            "Name": "mcp-manager",
            "Tags": ["mcp", "dind", "management"],
            "Address": "localhost",
            "Port": 18081,
            "Check": {
                "HTTP": "http://localhost:18081/health",
                "Interval": "30s"
            }
        }' || warn "Failed to register MCP manager with Consul"
    
    success "Service discovery and mesh integration setup complete"
}

# Validate deployment
validate_deployment() {
    log "Validating complete DinD MCP deployment..."
    
    local validation_failed=0
    
    # Check DinD orchestrator health
    if curl -sf http://localhost:18081/health >/dev/null; then
        success "✓ MCP Manager is healthy"
    else
        error "✗ MCP Manager is not healthy"
        validation_failed=1
    fi
    
    # Check container isolation
    local dind_containers
    dind_containers=$(curl -sf http://localhost:18081/containers 2>/dev/null | jq -r '.[].name' | wc -l)
    
    if [[ $dind_containers -gt 0 ]]; then
        success "✓ MCP containers running in DinD: $dind_containers"
    else
        error "✗ No MCP containers found in DinD"
        validation_failed=1
    fi
    
    # Check network isolation
    local dind_network
    if dind_network=$(docker network ls --filter name=sutazai-dind-internal --format "{{.Name}}"); then
        if [[ -n "$dind_network" ]]; then
            success "✓ DinD internal network exists: $dind_network"
        else
            error "✗ DinD internal network not found"
            validation_failed=1
        fi
    fi
    
    # Check volume management
    local dind_volumes
    dind_volumes=$(docker volume ls --filter name=mcp- --format "{{.Name}}" | wc -l)
    
    if [[ $dind_volumes -gt 0 ]]; then
        success "✓ DinD volumes created: $dind_volumes"
    else
        warn "⚠ No DinD volumes found"
    fi
    
    if [[ $validation_failed -eq 0 ]]; then
        success "🎉 DinD MCP deployment validation PASSED"
        return 0
    else
        error "❌ DinD MCP deployment validation FAILED"
        return 1
    fi
}

# Show deployment summary
show_deployment_summary() {
    log "=== DinD MCP Deployment Summary ==="
    echo ""
    echo "📊 Service Endpoints:"
    echo "  - MCP Orchestrator API: http://localhost:18080"
    echo "  - MCP Manager UI: http://localhost:18081"
    echo "  - Docker Daemon (TLS): tcp://localhost:12376"
    echo "  - Metrics: http://localhost:19090"
    echo ""
    echo "🐳 Container Architecture:"
    echo "  Host Docker"
    echo "  └── DinD Container (MCP Orchestrator)"
    echo "      ├── MCP Container (PostgreSQL)"
    echo "      ├── MCP Container (Files)"
    echo "      ├── MCP Container (HTTP)"
    echo "      └── ... (other MCPs)"
    echo ""
    echo "🔧 Management Commands:"
    echo "  - List containers: ${DIND_DIR}/orchestrator/scripts/deploy-mcp.sh list"
    echo "  - Deploy new MCP: ${DIND_DIR}/orchestrator/scripts/deploy-mcp.sh deploy"
    echo "  - Cleanup orphaned: ${DIND_DIR}/orchestrator/scripts/deploy-mcp.sh cleanup"
    echo "  - DinD status: ${DIND_DIR}/orchestrator/scripts/setup-dind.sh status"
    echo ""
    echo "📁 Important Paths:"
    echo "  - DinD configs: ${DIND_DIR}"
    echo "  - MCP manifests: ${DIND_DIR}/orchestrator/mcp-manifests/"
    echo "  - Orchestrator scripts: ${DIND_DIR}/orchestrator/scripts/"
    echo ""
}

# Main deployment function
main() {
    local command="${1:-deploy}"
    
    case "$command" in
        "deploy")
            log "🚀 Starting Docker-in-Docker MCP orchestration deployment..."
            
            check_environment || exit 1
            cleanup_orphaned_containers
            preflight_checks || exit 1
            deploy_dind_infrastructure || exit 1
            deploy_mcp_containers || exit 1
            setup_mesh_integration
            
            if validate_deployment; then
                show_deployment_summary
                success "🎉 DinD MCP deployment completed successfully!"
            else
                error "❌ DinD MCP deployment validation failed"
                exit 1
            fi
            ;;
        "validate")
            validate_deployment
            ;;
        "cleanup")
            cleanup_orphaned_containers
            "${DIND_DIR}/orchestrator/scripts/setup-dind.sh" cleanup
            ;;
        "status")
            "${DIND_DIR}/orchestrator/scripts/setup-dind.sh" status
            "${DIND_DIR}/orchestrator/scripts/deploy-mcp.sh" list
            ;;
        *)
            echo "Usage: $0 {deploy|validate|cleanup|status}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Full DinD MCP deployment"
            echo "  validate - Validate existing deployment"
            echo "  cleanup  - Clean up all DinD resources"
            echo "  status   - Show current deployment status"
            exit 1
            ;;
    esac
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    warn "Running as root - this may cause permission issues"
fi

# Run main function
main "$@"