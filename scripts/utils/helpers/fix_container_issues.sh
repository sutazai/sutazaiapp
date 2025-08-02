#!/bin/bash
# Fix Container Issues Script - SutazAI
# Fixes Loki, N8N, backend, and frontend container issues

source "$(dirname "$0")/common.sh"

set -euo pipefail

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
LOG_FILE="logs/container_fixes_$(date +%Y%m%d_%H%M%S).log"

# Setup logging
mkdir -p "$(dirname "$LOG_FILE")"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

log_info "🔧 Starting container fixes for SutazAI system..."

# Fix Loki configuration and restart
fix_loki_container() {
    log_info "Fixing Loki container issues..."
    
    # Stop Loki if running
    docker stop sutazai-loki 2>/dev/null || true
    docker rm sutazai-loki 2>/dev/null || true
    
    # Ensure config directory exists
    mkdir -p "$PROJECT_ROOT/config/loki"
    mkdir -p "$PROJECT_ROOT/data/loki"
    
    # Set proper permissions
    chmod -R 777 "$PROJECT_ROOT/data/loki" 2>/dev/null || true
    
    # Start Loki with fixed configuration
    docker-compose up -d loki
    
    # Wait for Loki to be healthy
    log_info "Waiting for Loki to start..."
    for i in {1..30}; do
        if docker ps | grep -q "sutazai-loki.*Up"; then
            log_success "Loki container started successfully"
            return 0
        fi
        sleep 2
    done
    
    log_warning "Loki may still be starting - check logs if issues persist"
}

# Fix N8N environment and restart
fix_n8n_container() {
    log_info "Fixing N8N container issues..."
    
    # Stop N8N if running  
    docker stop sutazai-n8n 2>/dev/null || true
    docker rm sutazai-n8n 2>/dev/null || true
    
    # Ensure data directory exists
    mkdir -p "$PROJECT_ROOT/data/n8n"
    chmod -R 777 "$PROJECT_ROOT/data/n8n" 2>/dev/null || true
    
    # Start N8N
    docker-compose up -d n8n
    
    # Wait for N8N to be ready
    log_info "Waiting for N8N to start..."
    for i in {1..30}; do
        if docker ps | grep -q "sutazai-n8n.*Up"; then
            log_success "N8N container started successfully"
            return 0
        fi
        sleep 2
    done
    
    log_warning "N8N may still be starting - check logs if issues persist"
}

# Fix backend container
fix_backend_container() {
    log_info "Fixing backend container issues..."
    
    # Check if working_main.py exists
    if [[ ! -f "$PROJECT_ROOT/backend/app/working_main.py" ]]; then
        log_error "backend/app/working_main.py not found - container cannot start"
        return 1
    fi
    
    # Build and start backend
    cd "$PROJECT_ROOT"
    docker-compose build backend
    docker-compose up -d backend
    
    # Wait for backend to be healthy
    log_info "Waiting for backend to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            log_success "Backend-automation container started successfully"
            return 0
        fi
        sleep 3
    done
    
    log_warning "Backend-automation may still be starting - check logs"
}

# Fix frontend container
fix_frontend_container() {
    log_info "Fixing frontend container issues..."
    
    # Check if app_agi_enhanced.py exists, if not use app.py
    if [[ ! -f "$PROJECT_ROOT/frontend/app_agi_enhanced.py" ]]; then
        if [[ -f "$PROJECT_ROOT/frontend/app.py" ]]; then
            log_info "Using app.py instead of app_agi_enhanced.py"
            # Update Dockerfile to use app.py
            sed -i 's/app_agi_enhanced.py/app.py/g' "$PROJECT_ROOT/frontend/Dockerfile" 2>/dev/null || true
        else
            log_error "No frontend application file found"
            return 1
        fi
    fi
    
    # Build and start frontend
    cd "$PROJECT_ROOT"
    docker-compose build frontend
    docker-compose up -d frontend
    
    # Wait for frontend to be ready
    log_info "Waiting for frontend to be available..."
    for i in {1..60}; do
        if curl -s http://localhost:8501 >/dev/null 2>&1; then
            log_success "Frontend-automation container started successfully"
            return 0
        fi
        sleep 3
    done
    
    log_warning "Frontend-automation may still be starting - check logs"
}

# Fix Qdrant health check issues
fix_qdrant_health() {
    log_info "Fixing Qdrant health check issues..."
    
    # Restart Qdrant
    docker restart sutazai-qdrant
    
    # Wait for Qdrant to be healthy
    for i in {1..30}; do
        if curl -s http://localhost:6333/cluster >/dev/null 2>&1; then
            log_success "Qdrant is now healthy"
            return 0
        fi
        sleep 2
    done
    
    log_warning "Qdrant may still be starting"
}

# Fix FAISS service issues
fix_faiss_service() {
    log_info "Fixing FAISS service issues..."
    
    # Restart FAISS
    docker restart sutazai-faiss 2>/dev/null || {
        log_info "Rebuilding FAISS service..."
        cd "$PROJECT_ROOT"
        docker-compose build faiss
        docker-compose up -d faiss
    }
    
    log_success "FAISS service restart attempted"
}

# Main execution
main() {
    log_info "🚀 Starting comprehensive container fixes..."
    
    # Fix all container issues
    fix_loki_container
    fix_n8n_container
    fix_backend_container  
    fix_frontend_container
    fix_qdrant_health
    fix_faiss_service
    
    # Final health check
    log_info "🔍 Running final health checks..."
    sleep 10
    
    # Check container statuses
    echo "\n📊 Container Status Summary:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai
    
    log_success "🎉 Container fixes completed! Check individual services for any remaining issues."
    log_info "📄 Access services:"
    log_info "   - Frontend: http://localhost:8501"
    log_info "   - Backend: http://localhost:8000/docs"
    log_info "   - N8N: http://localhost:5678"
    log_info "   - Grafana: http://localhost:3000"
}

# Run main function
main "$@"