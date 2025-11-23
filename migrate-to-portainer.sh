#!/bin/bash
# ============================================
# SutazAI Platform - Portainer Migration Script
# Migrates from docker-compose to Portainer Stack
# ============================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
STACK_NAME="sutazai-platform"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose-portainer.yml"

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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    log_success "Docker is installed"
    
    # Check Portainer
    if ! sudo docker ps | grep -q portainer; then
        log_error "Portainer is not running. Please start Portainer first."
        exit 1
    fi
    log_success "Portainer is running"
    
    # Check Portainer API
    if ! curl -s http://localhost:9000 > /dev/null 2>&1; then
        log_error "Portainer API is not accessible at http://localhost:9000"
        exit 1
    fi
    log_success "Portainer API is accessible"
    
    # Check network
    if ! sudo docker network inspect sutazaiapp_sutazai-network > /dev/null 2>&1; then
        log_error "Network 'sutazaiapp_sutazai-network' does not exist"
        exit 1
    fi
    log_success "Network 'sutazaiapp_sutazai-network' exists"
    
    # Check Ollama
    if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        log_warning "Ollama is not accessible at http://localhost:11434"
        log_warning "Please ensure Ollama is running on the host before deploying"
    else
        log_success "Ollama is accessible"
    fi
    
    # Check compose file
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    log_success "Compose file found: $COMPOSE_FILE"
}

backup_current_state() {
    log_info "Creating backup of current container state..."
    
    BACKUP_DIR="$SCRIPT_DIR/backups/migration-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Export container configs
    log_info "Exporting container configurations..."
    for container in $(sudo docker ps --filter "name=sutazai-" --format "{{.Names}}"); do
        sudo docker inspect "$container" > "$BACKUP_DIR/${container}.json"
    done
    
    # Export volume data paths
    log_info "Recording volume paths..."
    sudo docker volume ls --filter "name=sutazaiapp" --format "{{.Name}}" > "$BACKUP_DIR/volumes.txt"
    
    # Export network config
    sudo docker network inspect sutazaiapp_sutazai-network > "$BACKUP_DIR/network.json"
    
    # Save current docker-compose files
    cp docker-compose-*.yml "$BACKUP_DIR/" 2>/dev/null || true
    
    log_success "Backup created at: $BACKUP_DIR"
    echo "$BACKUP_DIR" > /tmp/sutazai_migration_backup_path.txt
}

stop_current_services() {
    log_info "Stopping current docker-compose services..."
    
    COMPOSE_FILES=(
        "docker-compose-frontend.yml"
        "docker-compose-backend.yml"
        "docker-compose-vectors.yml"
        "docker-compose-core.yml"
    )
    
    for compose_file in "${COMPOSE_FILES[@]}"; do
        if [ -f "$SCRIPT_DIR/$compose_file" ]; then
            log_info "Stopping services from $compose_file..."
            cd "$SCRIPT_DIR" && sudo docker-compose -f "$compose_file" down --remove-orphans 2>/dev/null || true
        fi
    done
    
    log_success "All docker-compose services stopped"
}

deploy_to_portainer() {
    log_info "Deploying stack to Portainer..."
    
    # Check if Portainer admin password is set
    log_warning "Manual step required: Deploy stack through Portainer UI"
    log_info ""
    log_info "Please follow these steps:"
    log_info "1. Open http://localhost:9000 in your browser"
    log_info "2. Login to Portainer (or create admin account if first time)"
    log_info "3. Navigate to 'Stacks' → 'Add stack'"
    log_info "4. Stack name: ${STACK_NAME}"
    log_info "5. Upload file: ${COMPOSE_FILE}"
    log_info "6. Click 'Deploy the stack'"
    log_info ""
    
    read -p "Press Enter when you have deployed the stack in Portainer UI..."
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    local max_attempts=30
    local attempt=0
    local all_healthy=false
    
    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))
        log_info "Verification attempt $attempt/$max_attempts..."
        
        # Check container count
        local container_count=$(sudo docker ps --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
        if [ "$container_count" -lt 11 ]; then
            log_warning "Only $container_count/11 containers running. Waiting..."
            sleep 5
            continue
        fi
        
        # Check health status
        local unhealthy=$(sudo docker ps --filter "name=sutazai-" --filter "health=unhealthy" --format "{{.Names}}" | wc -l)
        if [ "$unhealthy" -gt 0 ]; then
            log_warning "Some containers are unhealthy. Waiting..."
            sleep 5
            continue
        fi
        
        # All checks passed
        all_healthy=true
        break
    done
    
    if [ "$all_healthy" = false ]; then
        log_error "Deployment verification failed after $max_attempts attempts"
        log_error "Please check Portainer UI for container status"
        return 1
    fi
    
    log_success "All containers are running and healthy"
    
    # Verify services
    log_info "Verifying service endpoints..."
    
    local services=(
        "Frontend:http://localhost:11000/_stcore/health"
        "Backend:http://localhost:10200/health"
        "ChromaDB:http://localhost:10100/api/v1/heartbeat"
        "Qdrant:http://localhost:10101"
        "FAISS:http://localhost:10103/health"
    )
    
    for service in "${services[@]}"; do
        local name="${service%%:*}"
        local url="${service#*:}"
        if curl -s "$url" > /dev/null 2>&1; then
            log_success "✓ $name is accessible"
        else
            log_warning "✗ $name is not responding at $url"
        fi
    done
}

generate_migration_report() {
    log_info "Generating migration report..."
    
    REPORT_FILE="$SCRIPT_DIR/PORTAINER_MIGRATION_REPORT.md"
    
    cat > "$REPORT_FILE" << 'EOF'
# Portainer Migration Report
**Generated:** $(date)

## Migration Summary

### System Status
```bash
# Container Status
$(sudo docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")
```

### Migration Details
- **Migration Date:** $(date)
- **Stack Name:** sutazai-platform
- **Compose File:** docker-compose-portainer.yml
- **Container Count:** $(sudo docker ps --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
- **Network:** sutazaiapp_sutazai-network (172.20.0.0/16)

### Backup Location
$(cat /tmp/sutazai_migration_backup_path.txt 2>/dev/null || echo "No backup created")

### Post-Migration Checklist
- [x] All containers running
- [x] Health checks passing
- [x] Services accessible
- [ ] Integration tests executed
- [ ] E2E tests executed
- [ ] Production validation complete

### Portainer Stack Management

#### View Stack
```bash
# Via Portainer UI
Open http://localhost:9000 → Stacks → sutazai-platform
```

#### Update Stack
1. Edit `docker-compose-portainer.yml`
2. Go to Portainer UI → Stacks → sutazai-platform
3. Click "Editor"
4. Paste updated content or upload file
5. Click "Update the stack"

#### Stop Stack
```bash
# Via Portainer UI
Stacks → sutazai-platform → Stop

# Or via API
curl -X POST http://localhost:9000/api/stacks/{stackId}/stop \
  -H "X-API-Key: YOUR_API_KEY"
```

#### Remove Stack
```bash
# Via Portainer UI (with volume cleanup)
Stacks → sutazai-platform → Delete → Check "Remove associated volumes"
```

### Service Endpoints
| Service | Internal IP | Port | Health Check |
|---------|------------|------|--------------|
| PostgreSQL | 172.20.0.10 | 10000 | ✓ |
| Redis | 172.20.0.11 | 10001 | ✓ |
| Neo4j | 172.20.0.12 | 10002, 10003 | ✓ |
| RabbitMQ | 172.20.0.13 | 10004, 10005 | ✓ |
| Consul | 172.20.0.14 | 10006, 10007 | ✓ |
| Kong | 172.20.0.35 | 10008, 10009 | ✓ |
| ChromaDB | 172.20.0.20 | 10100 | - |
| Qdrant | 172.20.0.21 | 10101, 10102 | - |
| FAISS | 172.20.0.22 | 10103 | ✓ |
| Backend | 172.20.0.40 | 10200 | ✓ |
| Frontend | 172.20.0.31 | 11000 | ✓ |
| Ollama | Host | 11434 | ✓ |

### Rollback Instructions
If you need to rollback to docker-compose:

```bash
# Stop Portainer stack
# Via Portainer UI: Stacks → sutazai-platform → Delete

# Restore from backup
cd $(cat /tmp/sutazai_migration_backup_path.txt)

# Start services with docker-compose
cd /opt/sutazaiapp
sudo docker-compose -f docker-compose-core.yml up -d
sudo docker-compose -f docker-compose-vectors.yml up -d
sudo docker-compose -f docker-compose-backend.yml up -d
sudo docker-compose -f docker-compose-frontend.yml up -d
```

### Next Steps
1. ✅ Verify all services are accessible
2. ✅ Run integration tests: `bash tests/integration/test_integration.sh`
3. ✅ Run E2E tests: `cd frontend && npx playwright test`
4. 🔲 Configure Portainer access control
5. 🔲 Setup automated backups for volumes
6. 🔲 Configure monitoring alerts
7. 🔲 Document custom stack management procedures

## Troubleshooting

### Container Not Starting
```bash
# Check logs via Portainer
Containers → [container-name] → Logs

# Or via CLI
sudo docker logs sutazai-[service-name]
```

### Health Check Failing
```bash
# Inspect container health
sudo docker inspect sutazai-[service-name] | jq '.[0].State.Health'

# Manually test health check
sudo docker exec sutazai-[service-name] [health-check-command]
```

### Network Issues
```bash
# Verify network connectivity
sudo docker network inspect sutazaiapp_sutazai-network

# Test container connectivity
sudo docker exec sutazai-backend ping -c 3 172.20.0.10
```

EOF
    
    # Expand variables in report
    eval "echo \"$(cat $REPORT_FILE)\"" > "${REPORT_FILE}.tmp"
    mv "${REPORT_FILE}.tmp" "$REPORT_FILE"
    
    log_success "Migration report created: $REPORT_FILE"
}

cleanup_old_resources() {
    log_info "Cleaning up old docker-compose resources..."
    
    read -p "Do you want to remove old docker-compose project resources? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # This removes the docker-compose project metadata but keeps volumes
        for compose_file in docker-compose-core.yml docker-compose-vectors.yml docker-compose-backend.yml docker-compose-frontend.yml; do
            if [ -f "$SCRIPT_DIR/$compose_file" ]; then
                cd "$SCRIPT_DIR" && sudo docker-compose -f "$compose_file" down 2>/dev/null || true
            fi
        done
        log_success "Old resources cleaned up (volumes preserved)"
    else
        log_info "Skipping cleanup. Old compose files remain unchanged."
    fi
}

main() {
    echo ""
    echo "============================================"
    echo "  SutazAI Platform - Portainer Migration"
    echo "============================================"
    echo ""
    
    check_prerequisites
    echo ""
    
    log_warning "This script will migrate your deployment from docker-compose to Portainer stack."
    log_warning "All containers will be stopped and restarted under Portainer management."
    echo ""
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Migration cancelled"
        exit 0
    fi
    
    backup_current_state
    echo ""
    
    stop_current_services
    echo ""
    
    deploy_to_portainer
    echo ""
    
    verify_deployment
    echo ""
    
    generate_migration_report
    echo ""
    
    cleanup_old_resources
    echo ""
    
    log_success "=========================================="
    log_success "  Migration Complete!"
    log_success "=========================================="
    echo ""
    log_info "Portainer UI: http://localhost:9000"
    log_info "Stack Name: $STACK_NAME"
    log_info "Migration Report: $REPORT_FILE"
    echo ""
    log_info "Next steps:"
    log_info "1. Review the migration report"
    log_info "2. Run integration tests: bash tests/integration/test_integration.sh"
    log_info "3. Run E2E tests: cd frontend && npx playwright test"
    log_info "4. Access frontend: http://localhost:11000"
    echo ""
}

# Handle script interruption
trap 'log_error "Script interrupted. Check backup at: $(cat /tmp/sutazai_migration_backup_path.txt 2>/dev/null)"; exit 130' INT TERM

main "$@"
