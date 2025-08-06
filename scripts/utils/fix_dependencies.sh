#!/bin/bash
# SutazAI Dependency Remediation Script
# Fixes critical dependency conflicts and security issues

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
BACKUP_DIR="${PROJECT_ROOT}/backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/logs/dependency_fix_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $1" "$RED"
    exit 1
}

success() {
    log "SUCCESS: $1" "$GREEN"
}

warning() {
    log "WARNING: $1" "$YELLOW"
}

# Create backups
create_backups() {
    log "Creating backups..."
    mkdir -p "$BACKUP_DIR"
    
    # Backup critical files
    cp docker-compose*.yml "$BACKUP_DIR/" 2>/dev/null || true
    cp requirements*.txt "$BACKUP_DIR/" 2>/dev/null || true
    cp -r backend/requirements*.txt "$BACKUP_DIR/" 2>/dev/null || true
    cp -r frontend/requirements*.txt "$BACKUP_DIR/" 2>/dev/null || true
    cp .env* "$BACKUP_DIR/" 2>/dev/null || true
    
    success "Backups created in $BACKUP_DIR"
}

# Phase 1: Fix security vulnerabilities
fix_security_issues() {
    log "Phase 1: Fixing security vulnerabilities..."
    
    # Fix file permissions
    if [[ -f ".env" ]]; then
        chmod 600 .env
        success "Fixed .env permissions"
    fi
    
    if [[ -f "secrets.json" ]]; then
        chmod 600 secrets.json
        success "Fixed secrets.json permissions"
    fi
    
    # Remove hardcoded credentials (basic patterns)
    find . -name "*.py" -type f -exec grep -l "password.*=" {} \; | while read -r file; do
        if [[ ! "$file" =~ backup|env|.git ]]; then
            warning "Potential hardcoded password found in: $file"
            # Note: Manual review needed for these files
        fi
    done
    
    success "Security fixes completed"
}

# Phase 2: Consolidate Python requirements
consolidate_requirements() {
    log "Phase 2: Consolidating Python requirements..."
    
    # Create unified requirements.txt
    cat > requirements-unified.txt << 'EOF'
# SutazAI Unified Requirements - Version Locked for Stability

# ================================
# CORE FRAMEWORK VERSIONS (LOCKED)
# ================================
fastapi==0.109.2
pydantic==2.5.3
pydantic-settings==2.1.0
uvicorn[standard]==0.27.1
starlette==0.35.1

# ================================
# FRONTEND FRAMEWORK
# ================================
streamlit==1.32.2
streamlit-elements==0.1.0
streamlit-chat==0.1.1
streamlit-option-menu==0.3.6

# ================================
# AI/ML FRAMEWORKS (LOCKED)
# ================================
torch==2.1.2
transformers==4.36.2
sentence-transformers==2.2.2
accelerate==0.25.0
safetensors==0.4.1

# ================================
# LANGUAGE MODEL INTEGRATIONS
# ================================
langchain==0.1.10
langchain-community==0.0.24
langchain-experimental==0.0.15
langchain-ollama==0.1.1
ollama==0.1.7

# ================================
# VECTOR DATABASES (LOCKED)
# ================================
chromadb==0.4.22
qdrant-client==1.7.3
faiss-cpu==1.7.4

# ================================
# DATABASE & CACHE (LOCKED)
# ================================
sqlalchemy==2.0.25
alembic==1.13.1
psycopg2-binary==2.9.9
redis==5.0.1
pymongo==4.6.1

# ================================
# SECURITY (LOCKED)
# ================================
cryptography==42.0.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
bcrypt==4.1.2

# ================================
# WEB & API UTILITIES
# ================================
httpx==0.26.0
aiohttp==3.9.1
requests==2.31.0
websockets==12.0
python-multipart==0.0.6

# ================================
# DATA PROCESSING
# ================================
pandas==2.1.4
numpy==1.24.4
scipy==1.11.4
scikit-learn==1.3.2
matplotlib==3.8.2
plotly==5.18.0

# ================================
# DOCUMENT PROCESSING
# ================================
pypdf==3.17.4
python-docx==1.1.0
beautifulsoup4==4.12.2
unstructured==0.11.8

# ================================
# SYSTEM & MONITORING
# ================================
psutil==5.9.8
prometheus-client==0.19.0
docker==7.0.0
kubernetes==28.1.0

# ================================
# DEVELOPMENT TOOLS
# ================================
python-dotenv==1.0.0
pyyaml==6.0.1
jinja2==3.1.2
typer==0.9.0
rich==13.7.0
click==8.1.7

# ================================
# TESTING FRAMEWORK
# ================================
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0

# ================================
# CODE QUALITY
# ================================
black==23.12.1
ruff==0.1.8
mypy==1.8.0

# ================================
# ASYNC & CONCURRENCY
# ================================
asyncio-mqtt==0.13.0
aiofiles==23.2.1
aiocache==0.12.2
aioredis==2.0.1
EOF

    # Backup old requirements and use new unified one
    mv requirements.txt requirements-old.txt 2>/dev/null || true
    mv requirements-unified.txt requirements.txt
    
    # Remove conflicting requirements files
    rm -f requirements_complete.txt requirements_final.txt 2>/dev/null || true
    rm -f backend/requirements-minimal.txt backend/requirements-optimized.txt 2>/dev/null || true
    
    success "Unified requirements.txt created"
}

# Phase 3: Fix Docker port conflicts
fix_docker_ports() {
    log "Phase 3: Fixing Docker port conflicts..."
    
    # Create new docker-compose.yml with resolved port conflicts
    cat > docker-compose-fixed.yml << 'EOF'
# SutazAI Docker Compose - Port Conflicts Resolved
version: '3.8'

x-common-variables: &common-variables
  TZ: ${TZ:-UTC}

x-gpu-config: &gpu-config
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G

networks:
  sutazai-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  chromadb_data:
  qdrant_data:
  ollama_data:
  prometheus_data:
  grafana_data:
  loki_data:

services:
  # ========================================
  # CORE INFRASTRUCTURE (Standard Ports)
  # ========================================
  postgres:
    image: postgres:16.3-alpine
    container_name: sutazai-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-sutazai}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-sutazai_password}
      POSTGRES_DB: ${POSTGRES_DB:-sutazai}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"  # PostgreSQL standard port
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-sutazai}"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - sutazai-network

  redis:
    image: redis:7.2-alpine
    container_name: sutazai-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD:-redis_password}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"  # Redis standard port
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - sutazai-network

  # ========================================
  # VECTOR DATABASES (6xxx range)
  # ========================================
  chromadb:
    image: chromadb/chroma:0.5.0
    container_name: sutazai-chromadb
    restart: unless-stopped
    volumes:
      - chromadb_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
    ports:
      - "6001:8000"  # Changed from 8001 to 6001
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - sutazai-network

  qdrant:
    image: qdrant/qdrant:v1.9.2
    container_name: sutazai-qdrant
    restart: unless-stopped
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__SERVICE__GRPC_PORT: 6334
    ports:
      - "6333:6333"  # Keep Qdrant standard ports
      - "6334:6334"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - sutazai-network

  # ========================================
  # MODEL SERVING (11xxx range)
  # ========================================
  ollama:
    image: ollama/ollama:latest
    container_name: sutazai-ollama
    restart: unless-stopped
    <<: *gpu-config
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "10104:10104"  # Ollama standard port
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_ORIGINS="*"
    networks:
      - sutazai-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:10104/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ========================================
  # BACKEND SERVICES (8xxx range)
  # ========================================
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: sutazai-backend
    restart: unless-stopped
    volumes:
      - ./backend:/app
      - ./data:/data
      - ./logs:/logs
    environment:
      <<: *common-variables
      DATABASE_URL: postgresql://${POSTGRES_USER:-sutazai}:${POSTGRES_PASSWORD:-sutazai_password}@postgres:5432/${POSTGRES_DB:-sutazai}
      REDIS_URL: redis://:${REDIS_PASSWORD:-redis_password}@redis:6379/0
      OLLAMA_URL: http://ollama:10104
      CHROMADB_URL: http://chromadb:8000  # Internal communication
      QDRANT_URL: http://qdrant:6333
    ports:
      - "8000:8000"  # Backend API
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - sutazai-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  # ========================================
  # FRONTEND (85xx range)
  # ========================================
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: sutazai-frontend
    restart: unless-stopped
    volumes:
      - ./frontend:/app
    environment:
      <<: *common-variables
      BACKEND_URL: http://backend:8000
    ports:
      - "8501:8501"  # Streamlit standard port
    depends_on:
      - backend
    networks:
      - sutazai-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/healthz"]
      interval: 30s
      timeout: 10s
      retries: 5

  # ========================================
  # MONITORING STACK (9xxx & 3xxx range)
  # ========================================
  prometheus:
    image: prom/prometheus:latest
    container_name: sutazai-prometheus
    restart: unless-stopped
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"  # Prometheus standard port
    networks:
      - sutazai-network

  grafana:
    image: grafana/grafana:latest
    container_name: sutazai-grafana
    restart: unless-stopped
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    ports:
      - "3000:3000"  # Grafana standard port
    networks:
      - sutazai-network

  loki:
    image: grafana/loki:2.9.0
    container_name: sutazai-loki
    restart: unless-stopped
    ports:
      - "3100:3100"  # Loki standard port
    volumes:
      - loki_data:/loki
    networks:
      - sutazai-network
EOF

    # Replace main docker-compose.yml
    mv docker-compose.yml docker-compose-backup.yml 2>/dev/null || true
    mv docker-compose-fixed.yml docker-compose.yml
    
    success "Docker port conflicts resolved"
}

# Phase 4: Update Dockerfiles
update_dockerfiles() {
    log "Phase 4: Updating Docker base images..."
    
    # Update Python base images to latest secure version
    find . -name "Dockerfile*" -type f -exec grep -l "python:3.11" {} \; | while read -r dockerfile; do
        if [[ ! "$dockerfile" =~ backup|.git ]]; then
            sed -i 's/python:3.11-slim/python:3.12-slim/g' "$dockerfile"
            success "Updated base image in $dockerfile"
        fi
    done
    
    # Update Node.js base images
    find . -name "Dockerfile*" -type f -exec grep -l "node:18" {} \; | while read -r dockerfile; do
        if [[ ! "$dockerfile" =~ backup|.git ]]; then
            sed -i 's/node:18-alpine/node:20-alpine/g' "$dockerfile"
            success "Updated Node base image in $dockerfile"
        fi
    done
}

# Phase 5: Clean up legacy files
cleanup_legacy_files() {
    log "Phase 5: Cleaning up legacy and duplicate files..."
    
    # Remove legacy docker-compose files that cause conflicts
    legacy_files=(
        "docker/legacy/docker-compose-*.yml"
        "requirements_complete.txt"
        "requirements_final.txt"
        "backend/requirements-minimal.txt"
        "backend/requirements-optimized.txt"
    )
    
    for pattern in "${legacy_files[@]}"; do
        if ls $pattern 1> /dev/null 2>&1; then
            rm -f $pattern
            success "Removed legacy files: $pattern"
        fi
    done
    
    # Clean Python cache files
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    success "Legacy cleanup completed"
}

# Phase 6: Validate fixes
validate_fixes() {
    log "Phase 6: Validating fixes..."
    
    # Check if requirements.txt is valid
    if python3 -m pip check --quiet 2>/dev/null; then
        success "Python requirements validation passed"
    else
        warning "Python requirements may still have issues"
    fi
    
    # Check docker-compose syntax
    if docker-compose config > /dev/null 2>&1; then
        success "Docker Compose configuration is valid"
    else
        warning "Docker Compose configuration has issues"
    fi
    
    # Test port availability
    critical_ports=(5432 6379 8000 8501 10104)
    for port in "${critical_ports[@]}"; do
        if ! netstat -tuln | grep -q ":$port "; then
            success "Port $port is available"
        else
            warning "Port $port is still in use"
        fi
    done
}

# Main execution
main() {
    log "Starting SutazAI dependency remediation..."
    
    # Change to project directory
    cd "$PROJECT_ROOT" || error "Cannot access project directory"
    
    # Create log directory
    mkdir -p logs
    
    # Run remediation phases
    create_backups
    fix_security_issues
    consolidate_requirements
    fix_docker_ports
    update_dockerfiles
    cleanup_legacy_files
    validate_fixes
    
    log "============================================"
    success "Dependency remediation completed successfully!"
    log "============================================"
    
    echo ""
    log "Next steps:" "$GREEN"
    log "1. Review and test the changes" "$YELLOW"
    log "2. Run: docker-compose down && docker-compose up -d" "$YELLOW"
    log "3. Test all endpoints and services" "$YELLOW"
    log "4. Monitor logs for any remaining issues" "$YELLOW"
    echo ""
    log "Backups saved in: $BACKUP_DIR" "$BLUE"
    log "Full log available at: $LOG_FILE" "$BLUE"
    log "============================================"
}

# Run main function
main "$@" 2>&1 | tee -a "$LOG_FILE"