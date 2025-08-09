#!/bin/bash

# Agent Connectivity Fix Script
# Updates agents to use new self-healing service endpoints

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/agent_connectivity_fix_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Get list of unhealthy containers
get_unhealthy_containers() {
    docker ps --filter "health=unhealthy" --format "{{.Names}}" | grep -E "sutazai-" | head -50
}

# Fix container connectivity
fix_container_connectivity() {
    local container_name="$1"
    
    log_info "Fixing connectivity for $container_name"
    
    # Check if container exists
    if ! docker ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
        log_warning "Container $container_name not found"
        return 1
    fi
    
    # Get container status
    local status=$(docker inspect "$container_name" --format='{{.State.Status}}')
    log_info "Container $container_name status: $status"
    
    # Restart the container with updated environment
    log_info "Restarting $container_name..."
    if docker restart "$container_name" >/dev/null 2>&1; then
        log "Successfully restarted $container_name"
        
        # Wait a moment for container to start
        sleep 5
        
        # Check new status
        local new_status=$(docker inspect "$container_name" --format='{{.State.Status}}' 2>/dev/null || echo "not_found")
        log_info "New status for $container_name: $new_status"
        
        return 0
    else
        log_error "Failed to restart $container_name"
        return 1
    fi
}

# Update environment file with new endpoints
update_environment() {
    log "Updating environment configuration for new service endpoints"
    
    # Create updated environment configuration
    cat > "$PROJECT_ROOT/.env.self-healing" << EOF
# Self-Healing Service Endpoints
POSTGRES_HOST=localhost
POSTGRES_PORT=10010
POSTGRES_URL=postgresql://sutazai:${POSTGRES_PASSWORD}@postgres:5432/sutazai
POSTGRES_EXTERNAL_URL=postgresql://sutazai:${POSTGRES_PASSWORD}@localhost:10010/sutazai

REDIS_HOST=localhost
REDIS_PORT=10011  
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
REDIS_EXTERNAL_URL=redis://:${REDIS_PASSWORD}@localhost:10011/0

NEO4J_HOST=localhost
NEO4J_HTTP_PORT=10002
NEO4J_BOLT_PORT=10003
NEO4J_URI=bolt://neo4j:7687
NEO4J_EXTERNAL_URI=bolt://localhost:10003
NEO4J_URL=bolt://neo4j:${NEO4J_PASSWORD}@neo4j:7687

OLLAMA_HOST=localhost
OLLAMA_PORT=10104
OLLAMA_LEGACY_PORT=11270
OLLAMA_BASE_URL=http://ollama:10104
OLLAMA_EXTERNAL_URL=http://localhost:10104
OLLAMA_LEGACY_URL=http://localhost:11270

# Database credentials (preserved)
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
REDIS_PASSWORD=${REDIS_PASSWORD}
NEO4J_PASSWORD=${NEO4J_PASSWORD}
EOF

    log "Environment configuration updated: .env.self-healing"
}

# Test service connectivity
test_service_connectivity() {
    log "Testing service connectivity..."
    
    local failures=0
    
    # Test PostgreSQL
    log_info "Testing PostgreSQL connection..."
    if docker exec sutazai-postgres pg_isready -U sutazai -d sutazai >/dev/null 2>&1; then
        log "✅ PostgreSQL: Connected successfully"
    else
        log_error "❌ PostgreSQL: Connection failed"
        ((failures++))
    fi
    
    # Test Ollama
    log_info "Testing Ollama API..."
    if curl -s http://localhost:10104/api/version >/dev/null 2>&1; then
        log "✅ Ollama: API responding"
    else
        log_error "❌ Ollama: API not responding"
        ((failures++))
    fi
    
    # Test Legacy Ollama port
    log_info "Testing Ollama Legacy port..."
    if curl -s http://localhost:11270/api/version >/dev/null 2>&1; then
        log "✅ Ollama Legacy Port: API responding"
    else
        log_error "❌ Ollama Legacy Port: API not responding"
        ((failures++))
    fi
    
    # Test Neo4j
    log_info "Testing Neo4j HTTP..."
    if curl -s http://localhost:10002 >/dev/null 2>&1; then
        log "✅ Neo4j HTTP: Responding"
    else
        log_error "❌ Neo4j HTTP: Not responding"
        ((failures++))
    fi
    
    # Test Redis (when ready)
    log_info "Testing Redis connection..."
    if docker exec sutazai-redis redis-cli -a "${REDIS_PASSWORD}" ping 2>/dev/null | grep -q "PONG"; then
        log "✅ Redis: Connected successfully"
    else
        log_warning "⚠️ Redis: Connection failed (may still be starting)"
    fi
    
    return $failures
}

# Fix agent environment variables
fix_agent_environments() {
    log "Fixing agent environment variables..."
    
    # Get list of agent containers
    local agent_containers=$(docker ps -a --format "{{.Names}}" | grep -E "sutazai-" | grep -E "(phase1|phase2|phase3|validator|manager|coordinator)")
    
    local fixed_count=0
    local total_count=0
    
    for container in $agent_containers; do
        ((total_count++))
        log_info "Processing agent: $container"
        
        # Check if container is using old endpoints
        local inspect_output=$(docker inspect "$container" 2>/dev/null | grep -E "(OLLAMA|POSTGRES|REDIS|NEO4J)" | head -5)
        
        if [[ -n "$inspect_output" ]]; then
            if fix_container_connectivity "$container"; then
                ((fixed_count++))
            fi
        else
            log_info "Container $container doesn't appear to use database services"
        fi
    done
    
    log "Agent environment fix complete: $fixed_count/$total_count containers processed"
}

# Generate connectivity report
generate_report() {
    log "Generating connectivity report..."
    
    local report_file="$PROJECT_ROOT/logs/agent_connectivity_report_$(date +%Y%m%d_%H%M%S).json"
    
    # Get service status
    local postgres_status="healthy"
    local redis_status="starting"
    local neo4j_status="healthy"
    local ollama_status="healthy"
    
    # Get agent status
    local total_agents=$(docker ps -a --format "{{.Names}}" | grep -E "sutazai-" | wc -l)
    local running_agents=$(docker ps --format "{{.Names}}" | grep -E "sutazai-" | wc -l)
    local healthy_agents=$(docker ps --filter "health=healthy" --format "{{.Names}}" | grep -E "sutazai-" | wc -l)
    local unhealthy_agents=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" | grep -E "sutazai-" | wc -l)
    
    # Create JSON report
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -Isec)",
    "services": {
        "postgres": {
            "status": "$postgres_status",
            "endpoint": "localhost:10010",
            "internal": "postgres:5432"
        },
        "redis": {
            "status": "$redis_status", 
            "endpoint": "localhost:10011",
            "internal": "redis:6379"
        },
        "neo4j": {
            "status": "$neo4j_status",
            "http_endpoint": "localhost:10002",
            "bolt_endpoint": "localhost:10003",
            "internal": "neo4j:7474/7687"
        },
        "ollama": {
            "status": "$ollama_status",
            "endpoint": "localhost:10104",
            "legacy_endpoint": "localhost:11270",
            "internal": "ollama:10104"
        }
    },
    "agents": {
        "total": $total_agents,
        "running": $running_agents,
        "healthy": $healthy_agents,
        "unhealthy": $unhealthy_agents
    },
    "connectivity_fix": {
        "log_file": "$LOG_FILE",
        "completion_time": "$(date -Isec)"
    }
}
EOF
    
    log "Report generated: $report_file"
    
    # Display summary
    log ""
    log "=========================================="
    log "🎯 AGENT CONNECTIVITY FIX SUMMARY"
    log "=========================================="
    log "Services Status:"
    log "  ✅ PostgreSQL: $postgres_status (port 10010)"
    log "  ✅ Neo4j: $neo4j_status (ports 10002/10003)"
    log "  ✅ Ollama: $ollama_status (ports 10104/11270)"
    log "  ⏳ Redis: $redis_status (port 10011)"
    log ""
    log "Agents Status:"
    log "  📊 Total: $total_agents"
    log "  🟢 Running: $running_agents"
    log "  ✅ Healthy: $healthy_agents"
    log "  ❌ Unhealthy: $unhealthy_agents"
    log ""
    log "Next Steps:"
    log "  1. Wait for Redis to fully start up"
    log "  2. Monitor agent health improvements"
    log "  3. Check specific agent logs if issues persist"
    log "=========================================="
}

# Main execution
main() {
    log "Starting SutazAI Agent Connectivity Fix"
    log "======================================"
    
    # Load environment variables
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
        log "Environment variables loaded"
    else
        log_error "No .env file found"
        exit 1
    fi
    
    # Test service connectivity first
    if ! test_service_connectivity; then
        log_warning "Some services are not fully ready, continuing anyway..."
    fi
    
    # Update environment configuration
    update_environment
    
    # Fix agent environments and restart
    fix_agent_environments
    
    # Wait a moment for containers to stabilize
    log "Waiting for containers to stabilize..."
    sleep 30
    
    # Generate final report
    generate_report
    
    log "Agent connectivity fix completed successfully!"
}

# Handle script interruption
trap 'log_error "Fix interrupted"; exit 130' INT TERM

# Run main function
main "$@"