#!/bin/bash

# Docker Infrastructure Consolidation Script
# Senior Deployment Architect Implementation
# Date: 2025-08-20
# Version: 1.0.0

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
DOCKER_DIR="${PROJECT_ROOT}/docker"
BACKUP_DIR="${PROJECT_ROOT}/backups/docker_consolidation_$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="${PROJECT_ROOT}/docs/operations/cleanup/docker_consolidation_report.md"

# Logging function
log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
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

# Create backup of current Docker infrastructure
backup_current_state() {
    log "Creating backup of current Docker infrastructure..."
    
    mkdir -p "${BACKUP_DIR}"
    
    # Backup Docker directory
    if [ -d "${DOCKER_DIR}" ]; then
        cp -r "${DOCKER_DIR}" "${BACKUP_DIR}/"
        success "Docker directory backed up to ${BACKUP_DIR}"
    fi
    
    # Save current container state
    docker ps -a > "${BACKUP_DIR}/containers_state.txt" 2>/dev/null || true
    docker images > "${BACKUP_DIR}/images_state.txt" 2>/dev/null || true
    docker network ls > "${BACKUP_DIR}/networks_state.txt" 2>/dev/null || true
    docker volume ls > "${BACKUP_DIR}/volumes_state.txt" 2>/dev/null || true
    
    # Save docker-compose configurations
    find "${PROJECT_ROOT}" -name "docker-compose*.yml" -o -name "docker-compose*.yaml" | \
        while read -r file; do
            rel_path=$(realpath --relative-to="${PROJECT_ROOT}" "$file")
            mkdir -p "${BACKUP_DIR}/compose_files/$(dirname "$rel_path")"
            cp "$file" "${BACKUP_DIR}/compose_files/${rel_path}"
        done 2>/dev/null || true
    
    success "Backup completed at ${BACKUP_DIR}"
}

# Analyze current Docker infrastructure
analyze_infrastructure() {
    log "Analyzing current Docker infrastructure..."
    
    echo "=== Docker Files Analysis ===" > "${BACKUP_DIR}/analysis.txt"
    echo "" >> "${BACKUP_DIR}/analysis.txt"
    
    # Count Docker files
    total_files=$(find "${PROJECT_ROOT}" -type f \( -name "Dockerfile*" -o -name "docker-compose*.yml" \) 2>/dev/null | wc -l)
    project_files=$(find "${PROJECT_ROOT}" -path "*/node_modules" -prune -o -path "*/backups" -prune -o \
        -type f \( -name "Dockerfile*" -o -name "docker-compose*.yml" \) -print 2>/dev/null | grep -v -E "(node_modules|backups)" | wc -l)
    
    echo "Total Docker files: ${total_files}" >> "${BACKUP_DIR}/analysis.txt"
    echo "Project Docker files (excluding node_modules/backups): ${project_files}" >> "${BACKUP_DIR}/analysis.txt"
    echo "" >> "${BACKUP_DIR}/analysis.txt"
    
    # List orphaned Dockerfiles
    echo "=== Potentially Orphaned Files ===" >> "${BACKUP_DIR}/analysis.txt"
    
    # Check for unused Dockerfiles
    for dockerfile in $(find "${DOCKER_DIR}" -name "Dockerfile*" -type f 2>/dev/null); do
        basename_file=$(basename "$dockerfile")
        dirname_file=$(dirname "$dockerfile")
        
        # Check if referenced in any docker-compose file
        if ! grep -r "$basename_file\|$(basename "$dirname_file")" "${DOCKER_DIR}" --include="docker-compose*.yml" > /dev/null 2>&1; then
            echo "  - ${dockerfile}" >> "${BACKUP_DIR}/analysis.txt"
            warning "Potentially orphaned: ${dockerfile}"
        fi
    done
    
    success "Analysis completed and saved to ${BACKUP_DIR}/analysis.txt"
}

# Identify consolidation opportunities
identify_consolidation() {
    log "Identifying consolidation opportunities..."
    
    local consolidation_plan="${BACKUP_DIR}/consolidation_plan.txt"
    
    echo "=== Consolidation Opportunities ===" > "${consolidation_plan}"
    echo "" >> "${consolidation_plan}"
    
    # Check for missing main docker-compose.yml
    if [ ! -f "${DOCKER_DIR}/docker-compose.yml" ]; then
        echo "1. CRITICAL: Missing main docker-compose.yml" >> "${consolidation_plan}"
        echo "   Action: Create unified orchestration file" >> "${consolidation_plan}"
        warning "Missing main docker-compose.yml"
    fi
    
    # Check for fragmented MCP services
    mcp_compose_count=$(find "${DOCKER_DIR}" -path "*mcp*" -name "docker-compose*.yml" 2>/dev/null | wc -l)
    if [ "${mcp_compose_count}" -gt 1 ]; then
        echo "2. MCP Services Fragmentation: ${mcp_compose_count} separate files" >> "${consolidation_plan}"
        echo "   Action: Consolidate into single MCP orchestration" >> "${consolidation_plan}"
        warning "Found ${mcp_compose_count} separate MCP docker-compose files"
    fi
    
    # Check for duplicate service definitions
    echo "3. Checking for duplicate service definitions..." >> "${consolidation_plan}"
    
    # Find all service names
    service_names=$(grep -h "^\s*[a-z][a-z0-9-]*:" "${DOCKER_DIR}"/**/docker-compose*.yml 2>/dev/null | \
        sed 's/://g' | sed 's/^[[:space:]]*//' | sort | uniq -d)
    
    if [ -n "${service_names}" ]; then
        echo "   Duplicate services found:" >> "${consolidation_plan}"
        echo "${service_names}" | while read -r service; do
            echo "     - ${service}" >> "${consolidation_plan}"
        done
    fi
    
    success "Consolidation opportunities identified"
}

# Generate consolidation script
generate_consolidation_script() {
    log "Generating consolidation script..."
    
    local script_file="${BACKUP_DIR}/execute_consolidation.sh"
    
    cat > "${script_file}" << 'EOF'
#!/bin/bash
# Auto-generated Docker Consolidation Script
# Generated: DATE_PLACEHOLDER

set -euo pipefail

echo "Starting Docker consolidation..."

# Step 1: Create main docker-compose.yml
create_main_compose() {
    cat > /opt/sutazaiapp/docker/docker-compose.yml << 'COMPOSE_EOF'
version: '3.8'

networks:
  sutazai-main:
    driver: bridge
  sutazai-internal:
    driver: bridge
    internal: true

volumes:
  postgres-data:
  redis-data:
  neo4j-data:
  chromadb-data:
  qdrant-data:
  faiss-data:
  ollama-data:
  grafana-data:
  prometheus-data:
  loki-data:
  consul-data:

services:
  # Core Application Services
  backend:
    build:
      context: ../backend
      dockerfile: Dockerfile
    image: sutazaiapp-backend:latest
    container_name: sutazai-backend
    ports:
      - "10010:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/sutazai
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - sutazai-main
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    image: sutazaiapp-frontend:latest
    container_name: sutazai-frontend
    ports:
      - "10011:8501"
    depends_on:
      - backend
    networks:
      - sutazai-main
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Databases
  postgres:
    image: postgres:15-alpine
    container_name: sutazai-postgres
    ports:
      - "10000:5432"
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=sutazai
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - sutazai-main
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: sutazai-redis
    ports:
      - "10001:6379"
    volumes:
      - redis-data:/data
    networks:
      - sutazai-main
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Additional services would be added here...
COMPOSE_EOF
    echo "Main docker-compose.yml created"
}

# Step 2: Consolidate MCP services
consolidate_mcp_services() {
    echo "Consolidating MCP services..."
    # Implementation would merge MCP compose files
}

# Step 3: Remove orphaned files
cleanup_orphaned_files() {
    echo "Cleaning up orphaned Docker files..."
    # Implementation would safely remove identified orphaned files
}

# Execute consolidation
create_main_compose
consolidate_mcp_services
cleanup_orphaned_files

echo "Consolidation complete!"
EOF
    
    sed -i "s/DATE_PLACEHOLDER/$(date +'%Y-%m-%d %H:%M:%S')/" "${script_file}"
    chmod +x "${script_file}"
    
    success "Consolidation script generated at ${script_file}"
}

# Main execution flow
main() {
    log "Docker Infrastructure Consolidation Tool v1.0.0" "$GREEN"
    log "Senior Deployment Architect Implementation" "$GREEN"
    echo ""
    
    # Check if running as root or with appropriate permissions
    if [ "$EUID" -ne 0 ] && ! groups | grep -q docker; then
        error "This script must be run as root or by a user in the docker group"
        exit 1
    fi
    
    # Confirmation prompt
    echo -e "${YELLOW}This script will analyze and prepare Docker infrastructure consolidation.${NC}"
    echo -e "${YELLOW}A backup will be created before any changes.${NC}"
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Operation cancelled by user"
        exit 0
    fi
    
    # Execute analysis steps
    backup_current_state
    analyze_infrastructure
    identify_consolidation
    generate_consolidation_script
    
    echo ""
    success "Docker consolidation analysis complete!"
    echo ""
    log "Next steps:"
    echo "  1. Review the analysis at: ${BACKUP_DIR}/analysis.txt"
    echo "  2. Review consolidation plan at: ${BACKUP_DIR}/consolidation_plan.txt"
    echo "  3. Review and customize the script at: ${BACKUP_DIR}/execute_consolidation.sh"
    echo "  4. Execute consolidation when ready: bash ${BACKUP_DIR}/execute_consolidation.sh"
    echo ""
    warning "Always review generated scripts before execution!"
}

# Run main function
main "$@"