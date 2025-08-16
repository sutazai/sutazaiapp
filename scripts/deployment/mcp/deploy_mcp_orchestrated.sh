#!/bin/bash
# MCP Orchestrated Deployment Script
# Cleans up chaos and deploys MCPs with proper container orchestration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/opt/sutazaiapp"
ORCHESTRATOR="${SCRIPT_DIR}/orchestrator.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to cleanup chaos
cleanup_chaos() {
    log_info "=== Phase 1: Cleaning up MCP deployment chaos ==="
    
    # Stop orphaned containers
    log_info "Stopping orphaned MCP containers..."
    docker ps -a | grep -E "tender_|optimistic_|hungry_|vigilant_" | awk '{print $1}' | while read container; do
        docker stop "$container" 2>/dev/null || true
        docker rm -f "$container" 2>/dev/null || true
        log_info "Removed container: $container"
    done
    
    # Kill zombie processes
    log_info "Killing zombie MCP processes..."
    pkill -9 -f "npm exec.*mcp" 2>/dev/null || true
    pkill -9 -f "mcp-language-se" 2>/dev/null || true
    pkill -9 -f "docker run.*mcp/" 2>/dev/null || true
    
    # Remove any failed MCP containers
    docker ps -a | grep "mcp/" | grep -E "Exited|Dead" | awk '{print $1}' | while read container; do
        docker rm -f "$container" 2>/dev/null || true
    done
    
    log_success "Cleanup complete"
}

# Function to ensure prerequisites
ensure_prerequisites() {
    log_info "=== Phase 2: Ensuring prerequisites ==="
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required"
        exit 1
    fi
    
    # Install Python dependencies
    log_info "Installing Python dependencies..."
    pip3 install docker pyyaml aiohttp --quiet
    
    # Ensure Docker network exists
    if ! docker network ls | grep -q "sutazai-network"; then
        log_info "Creating Docker network: sutazai-network"
        docker network create sutazai-network
    fi
    
    # Create necessary directories
    mkdir -p "${BASE_DIR}/config/ports"
    mkdir -p "${BASE_DIR}/logs/mcp"
    
    log_success "Prerequisites ready"
}

# Function to deploy MCPs
deploy_mcps() {
    log_info "=== Phase 3: Deploying MCP servers with orchestration ==="
    
    # Make orchestrator executable
    chmod +x "${ORCHESTRATOR}"
    
    # Run orchestrator
    log_info "Starting MCP orchestrator..."
    python3 "${ORCHESTRATOR}" --cleanup --deploy --sequential
    
    if [ $? -eq 0 ]; then
        log_success "MCP deployment successful"
    else
        log_error "MCP deployment failed"
        return 1
    fi
}

# Function to verify deployment
verify_deployment() {
    log_info "=== Phase 4: Verifying deployment ==="
    
    # Check container status
    log_info "Checking MCP container status..."
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep "sutazai-mcp-" || {
        log_warning "No MCP containers found"
        return 1
    }
    
    # Run health checks
    log_info "Running health checks..."
    python3 "${ORCHESTRATOR}" --health
    
    # Test multi-client access
    log_info "Testing multi-client access..."
    python3 "${ORCHESTRATOR}" --test
    
    log_success "Verification complete"
}

# Function to integrate with mesh
integrate_mesh() {
    log_info "=== Phase 5: Integrating with service mesh ==="
    
    # Restart backend to pick up new MCP configurations
    log_info "Restarting backend service..."
    docker restart sutazai-backend 2>/dev/null || {
        log_warning "Backend not running, skipping restart"
    }
    
    # Wait for backend to be ready
    sleep 5
    
    # Test mesh integration
    log_info "Testing mesh integration..."
    curl -s http://localhost:10010/api/v1/mcp/status | python3 -m json.tool || {
        log_warning "Mesh integration test failed"
        return 1
    }
    
    log_success "Mesh integration complete"
}

# Function to generate report
generate_report() {
    log_info "=== Phase 6: Generating deployment report ==="
    
    REPORT_FILE="${BASE_DIR}/logs/mcp/deployment_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "${REPORT_FILE}" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "deployment_status": "completed",
    "containers": $(docker ps --format '{"name":"{{.Names}}","status":"{{.Status}}","ports":"{{.Ports}}"}' | grep sutazai-mcp | jq -s .),
    "port_allocations": $(cat ${BASE_DIR}/config/ports/mcp_ports.json 2>/dev/null || echo '{}'),
    "health_status": $(python3 -c "
import sys
sys.path.append('${SCRIPT_DIR}')
from orchestrator import MCPOrchestrator
o = MCPOrchestrator()
import json
print(json.dumps(o.health_check_all()))
" 2>/dev/null || echo '{}')
}
EOF
    
    log_success "Report saved to: ${REPORT_FILE}"
}

# Main execution
main() {
    log_info "Starting MCP Orchestrated Deployment"
    log_info "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    
    # Execute phases
    cleanup_chaos
    ensure_prerequisites
    deploy_mcps
    verify_deployment
    integrate_mesh
    generate_report
    
    log_success "=== MCP Deployment Complete ==="
    log_info "Port allocations saved to: ${BASE_DIR}/config/ports/mcp_ports.json"
    log_info "Docker compose file: ${BASE_DIR}/docker/docker-compose.mcp-orchestrated.yml"
    log_info "Logs available at: ${BASE_DIR}/logs/mcp/"
    
    # Show summary
    echo ""
    log_info "Deployment Summary:"
    docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}" | grep sutazai-mcp || true
    
    echo ""
    log_success "Multi-client access enabled through mesh at http://localhost:10010/api/v1/mcp/"
}

# Run main function
main "$@"