#!/bin/bash
# Deploy MCP Network Infrastructure
# Comprehensive network setup with proper isolation and load balancing

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.mcp-network.yml"
LOG_FILE="/tmp/mcp-network-deploy.log"

# Logging function
log() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
    
    # Check if sutazai-network exists
    if ! docker network ls | grep -q sutazai-network; then
        log_error "sutazai-network not found. Please ensure the main infrastructure is running."
        exit 1
    fi
    
    # Check if ports are available
    for port in 11090 11099 11100 11101 11102 11103 11104 11105; do
        if ss -tulpn | grep -q ":$port "; then
            log_warn "Port $port is already in use. This may cause conflicts."
        fi
    done
    
    log_info "Prerequisites check completed."
}

# Clean up any existing MCP network resources
cleanup_existing() {
    log_info "Cleaning up existing MCP network resources..."
    
    # Stop and remove any existing MCP containers
    for container in sutazai-mcp-consul sutazai-mcp-haproxy sutazai-mcp-monitor \
                    sutazai-mcp-postgres sutazai-mcp-files sutazai-mcp-http \
                    sutazai-mcp-ddg sutazai-mcp-github sutazai-mcp-memory; do
        if docker ps -a --format '{{.Names}}' | grep -q "^$container$"; then
            log_info "Removing existing container: $container"
            docker rm -f "$container" 2>/dev/null || true
        fi
    done
    
    # Remove MCP internal network if it exists
    if docker network ls --format '{{.Name}}' | grep -q "^mcp-internal$"; then
        log_info "Removing existing mcp-internal network"
        docker network rm mcp-internal 2>/dev/null || true
    fi
    
    log_info "Cleanup completed."
}

# Build base MCP image
build_base_image() {
    log_info "Building base MCP image..."
    
    cd "${PROJECT_ROOT}/docker/mcp-services/base"
    docker build -t sutazaiapp/mcp-base:latest . || {
        log_error "Failed to build base MCP image"
        exit 1
    }
    
    log_info "Base MCP image built successfully."
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "${PROJECT_ROOT}/docker/config/consul/mcp"
    mkdir -p "${PROJECT_ROOT}/docker/config/haproxy"
    mkdir -p "${PROJECT_ROOT}/scripts/network/monitor"
    mkdir -p "${PROJECT_ROOT}/docs/network"
    
    # Set proper permissions
    chmod 755 "${PROJECT_ROOT}/scripts/network/monitor"
    
    log_info "Directories created."
}

# Deploy network infrastructure
deploy_network() {
    log_info "Deploying MCP network infrastructure..."
    
    cd "$PROJECT_ROOT"
    
    # Use docker-compose or docker compose based on availability
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Deploy the network infrastructure
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" up -d || {
        log_error "Failed to deploy MCP network infrastructure"
        exit 1
    }
    
    log_info "Network infrastructure deployed."
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    local services=(
        "sutazai-mcp-consul:11090:/health"
        "sutazai-mcp-haproxy:11099:/stats"
        "sutazai-mcp-monitor:11091:/health"
    )
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service port path <<< "$service_info"
        
        log_info "Waiting for $service to be ready..."
        local retries=30
        while [ $retries -gt 0 ]; do
            if curl -sf "http://localhost:$port$path" >/dev/null 2>&1; then
                log_info "$service is ready"
                break
            fi
            sleep 2
            ((retries--))
        done
        
        if [ $retries -eq 0 ]; then
            log_warn "$service did not become ready in time"
        fi
    done
    
    log_info "Service readiness check completed."
}

# Register MCP services with Consul
register_services() {
    log_info "Registering MCP services with Consul..."
    
    # Wait a bit for Consul to be fully ready
    sleep 5
    
    # Register each MCP service
    local services=("postgres:11100" "files:11101" "http:11102" "ddg:11103" "github:11104" "memory:11105")
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service_info"
        
        cat << EOF | curl -X PUT "http://localhost:11090/v1/agent/service/register" -d @- || log_warn "Failed to register $name"
{
  "ID": "mcp-$name",
  "Name": "mcp-$name",
  "Port": $port,
  "Address": "mcp-$name",
  "Tags": ["mcp", "$name"],
  "Check": {
    "HTTP": "http://mcp-$name:$port/health",
    "Interval": "10s",
    "Timeout": "5s"
  },
  "Meta": {
    "version": "1.0.0",
    "mcp_service": "true"
  }
}
EOF
    done
    
    log_info "Service registration completed."
}

# Update port registry
update_port_registry() {
    log_info "Updating port registry..."
    
    local port_registry="${PROJECT_ROOT}/IMPORTANT/diagrams/PortRegistry.md"
    
    # Create backup
    cp "$port_registry" "${port_registry}.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
    
    # Add MCP services to port registry (append to existing content)
    cat >> "$port_registry" << 'EOF'

## MCP Services (11090-11199) - Added 2025-08-16

- 11090: MCP Consul UI (sutazai-mcp-consul)
- 11091: MCP Network Monitor (sutazai-mcp-monitor)  
- 11099: MCP HAProxy Stats (sutazai-mcp-haproxy)
- 11100: MCP PostgreSQL Service (sutazai-mcp-postgres)
- 11101: MCP Files Service (sutazai-mcp-files)
- 11102: MCP HTTP Service (sutazai-mcp-http)
- 11103: MCP DuckDuckGo Service (sutazai-mcp-ddg)
- 11104: MCP GitHub Service (sutazai-mcp-github)
- 11105: MCP Memory Service (sutazai-mcp-memory)

**Status**: All MCP services properly networked and load balanced
**Network**: sutazai-network + mcp-internal (isolated)
**Load Balancer**: HAProxy with health checks and failover
**Service Discovery**: Consul with automatic registration
EOF
    
    log_info "Port registry updated."
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    local failed_checks=0
    
    # Check container status
    local containers=(
        "sutazai-mcp-consul"
        "sutazai-mcp-haproxy" 
        "sutazai-mcp-monitor"
    )
    
    for container in "${containers[@]}"; do
        if ! docker ps --format '{{.Names}}' | grep -q "^$container$"; then
            log_error "Container $container is not running"
            ((failed_checks++))
        else
            log_info "✓ Container $container is running"
        fi
    done
    
    # Check network connectivity
    log_info "Checking network connectivity..."
    
    # Test Consul
    if curl -sf "http://localhost:11090/v1/status/leader" >/dev/null; then
        log_info "✓ Consul is accessible"
    else
        log_error "✗ Consul is not accessible"
        ((failed_checks++))
    fi
    
    # Test HAProxy stats
    if curl -sf "http://localhost:11099/stats" >/dev/null; then
        log_info "✓ HAProxy stats are accessible"
    else
        log_error "✗ HAProxy stats are not accessible"
        ((failed_checks++))
    fi
    
    # Test network monitor
    if curl -sf "http://localhost:11091/health" >/dev/null; then
        log_info "✓ Network monitor is accessible"
    else
        log_error "✗ Network monitor is not accessible"
        ((failed_checks++))
    fi
    
    # Check networks
    if docker network ls | grep -q mcp-internal; then
        log_info "✓ MCP internal network exists"
    else
        log_error "✗ MCP internal network missing"
        ((failed_checks++))
    fi
    
    if [ $failed_checks -eq 0 ]; then
        log_info "✅ All validation checks passed!"
        return 0
    else
        log_error "❌ $failed_checks validation checks failed"
        return 1
    fi
}

# Print deployment summary
print_summary() {
    log_info "Deployment Summary:"
    echo "=========================="
    echo "MCP Network Infrastructure deployed successfully!"
    echo ""
    echo "Access Points:"
    echo "- Consul UI:        http://localhost:11090"
    echo "- HAProxy Stats:    http://localhost:11099/stats"
    echo "- Network Monitor:  http://localhost:11091"
    echo ""
    echo "MCP Service Ports:"
    echo "- PostgreSQL:       http://localhost:11100"
    echo "- Files:            http://localhost:11101"
    echo "- HTTP:             http://localhost:11102"
    echo "- DuckDuckGo:       http://localhost:11103"
    echo "- GitHub:           http://localhost:11104"
    echo "- Memory:           http://localhost:11105"
    echo ""
    echo "Network Features:"
    echo "✓ Service Discovery (Consul)"
    echo "✓ Load Balancing (HAProxy)"
    echo "✓ Health Monitoring"
    echo "✓ Network Isolation"
    echo "✓ Multi-client Support"
    echo ""
    echo "Logs: $LOG_FILE"
    echo "=========================="
}

# Main deployment function
main() {
    log_info "Starting MCP network infrastructure deployment..."
    
    check_prerequisites
    cleanup_existing
    create_directories
    build_base_image
    deploy_network
    wait_for_services
    register_services
    update_port_registry
    
    if validate_deployment; then
        print_summary
        log_info "MCP network deployment completed successfully!"
        exit 0
    else
        log_error "MCP network deployment validation failed!"
        exit 1
    fi
}

# Run main function
main "$@"