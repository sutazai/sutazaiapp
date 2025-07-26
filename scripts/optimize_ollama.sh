#!/bin/bash

# Ollama Performance Optimization Script for SutazAI
# This script optimizes Ollama for better performance and reduces timeouts

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   warning "Running as root user"
fi

log "Starting Ollama optimization for SutazAI..."

# 1. Optimize Ollama environment variables
optimize_ollama_env() {
    log "Optimizing Ollama environment variables..."
    
    # Create optimized environment file
    cat > /opt/sutazaiapp/.env.ollama << EOF
# Ollama Performance Settings
OLLAMA_NUM_PARALLEL=4
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_KEEP_ALIVE=5m
OLLAMA_CPU_THREADS=10
OLLAMA_GPU_LAYERS=99
OLLAMA_MODELS=/root/.ollama/models
OLLAMA_HOST=0.0.0.0:11434

# Memory settings
OLLAMA_MAX_MEMORY=8192
OLLAMA_FLASH_ATTENTION=1

# Timeout settings
OLLAMA_TIMEOUT=300
OLLAMA_IDLE_TIMEOUT=600

# Logging
OLLAMA_DEBUG=false
OLLAMA_LOG_LEVEL=info
EOF
    
    success "Environment variables optimized"
}

# 2. Preload essential models
preload_models() {
    log "Preloading essential models..."
    
    ESSENTIAL_MODELS=(
        "codellama:7b"
        "llama3.2:1b"
        "nomic-embed-text"
    )
    
    for model in "${ESSENTIAL_MODELS[@]}"; do
        log "Checking model: $model"
        if docker exec sutazai-ollama ollama list | grep -q "$model"; then
            success "Model $model already loaded"
        else
            warning "Model $model not found, pulling..."
            docker exec sutazai-ollama ollama pull "$model" || error "Failed to pull $model"
        fi
    done
}

# 3. Configure model-specific optimizations
configure_model_optimizations() {
    log "Configuring model-specific optimizations..."
    
    # Create model configuration
    docker exec sutazai-ollama bash -c 'mkdir -p /root/.ollama/configs'
    
    # Codellama optimization
    docker exec sutazai-ollama bash -c 'cat > /root/.ollama/configs/codellama.json << EOF
{
    "num_ctx": 4096,
    "num_batch": 512,
    "num_gqa": 8,
    "rope_frequency_base": 10000,
    "rope_frequency_scale": 1.0,
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1
}
EOF'
    
    success "Model configurations created"
}

# 4. Optimize container resources
optimize_container_resources() {
    log "Optimizing container resources..."
    
    # Update container CPU and memory limits
    docker update sutazai-ollama \
        --cpus="4" \
        --memory="8g" \
        --memory-swap="12g" \
        || warning "Could not update container resources"
    
    success "Container resources optimized"
}

# 5. Setup model caching
setup_model_caching() {
    log "Setting up model caching..."
    
    # Create cache directories
    docker exec sutazai-ollama bash -c '
        mkdir -p /root/.ollama/cache
        mkdir -p /root/.ollama/tmp
        chmod 755 /root/.ollama/cache
        chmod 755 /root/.ollama/tmp
    '
    
    success "Model caching configured"
}

# 6. Create health check script
create_health_check() {
    log "Creating Ollama health check script..."
    
    cat > /opt/sutazaiapp/scripts/ollama_health_check.sh << 'EOF'
#!/bin/bash
# Ollama Health Check

check_ollama_health() {
    # Check if Ollama is responding
    if curl -f -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is healthy"
        
        # Check loaded models
        MODELS=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null | wc -l)
        echo "Loaded models: $MODELS"
        
        # Check response time
        RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:11434/api/tags)
        echo "API response time: ${RESPONSE_TIME}s"
        
        # Memory usage
        MEMORY=$(docker stats sutazai-ollama --no-stream --format "{{.MemUsage}}" 2>/dev/null)
        echo "Memory usage: $MEMORY"
        
        return 0
    else
        echo "Ollama is not responding"
        return 1
    fi
}

# Run health check
check_ollama_health
EOF
    
    chmod +x /opt/sutazaiapp/scripts/ollama_health_check.sh
    success "Health check script created"
}

# 7. Optimize model loading
optimize_model_loading() {
    log "Optimizing model loading..."
    
    # Create model preload script
    cat > /opt/sutazaiapp/scripts/preload_models.sh << 'EOF'
#!/bin/bash
# Preload models on startup

MODELS=("codellama:7b" "llama3.2:1b" "nomic-embed-text")

for model in "${MODELS[@]}"; do
    echo "Preloading $model..."
    curl -X POST http://localhost:11434/api/generate \
        -d "{\"model\": \"$model\", \"prompt\": \"test\", \"stream\": false}" \
        > /dev/null 2>&1 &
done

wait
echo "All models preloaded"
EOF
    
    chmod +x /opt/sutazaiapp/scripts/preload_models.sh
    success "Model preloading script created"
}

# 8. Configure automatic cleanup
setup_auto_cleanup() {
    log "Setting up automatic cleanup..."
    
    # Create cleanup script
    cat > /opt/sutazaiapp/scripts/ollama_cleanup.sh << 'EOF'
#!/bin/bash
# Clean up unused models and cache

# Remove old cache files (older than 7 days)
docker exec sutazai-ollama find /root/.ollama/cache -type f -mtime +7 -delete 2>/dev/null

# Clean temporary files
docker exec sutazai-ollama find /root/.ollama/tmp -type f -mtime +1 -delete 2>/dev/null

# Report disk usage
DISK_USAGE=$(docker exec sutazai-ollama du -sh /root/.ollama/ 2>/dev/null | cut -f1)
echo "Ollama disk usage: $DISK_USAGE"
EOF
    
    chmod +x /opt/sutazaiapp/scripts/ollama_cleanup.sh
    
    # Add to crontab (daily at 3 AM)
    (crontab -l 2>/dev/null; echo "0 3 * * * /opt/sutazaiapp/scripts/ollama_cleanup.sh > /opt/sutazaiapp/logs/ollama_cleanup.log 2>&1") | crontab -
    
    success "Auto cleanup configured"
}

# 9. Apply optimizations
apply_optimizations() {
    log "Applying all optimizations..."
    
    # Restart Ollama with new settings
    log "Restarting Ollama container..."
    docker-compose -f /opt/sutazaiapp/docker-compose-consolidated.yml restart ollama
    
    # Wait for Ollama to be ready
    log "Waiting for Ollama to be ready..."
    sleep 10
    
    # Run health check
    if /opt/sutazaiapp/scripts/ollama_health_check.sh; then
        success "Ollama is healthy after optimization"
    else
        error "Ollama health check failed"
    fi
}

# Main execution
main() {
    optimize_ollama_env
    preload_models
    configure_model_optimizations
    optimize_container_resources
    setup_model_caching
    create_health_check
    optimize_model_loading
    setup_auto_cleanup
    apply_optimizations
    
    echo ""
    success "Ollama optimization complete!"
    echo ""
    log "Optimization summary:"
    echo "  - Environment variables optimized"
    echo "  - Essential models preloaded"
    echo "  - Model-specific configs created"
    echo "  - Container resources optimized"
    echo "  - Caching enabled"
    echo "  - Health check script created"
    echo "  - Auto cleanup scheduled"
    echo ""
    log "Run './scripts/ollama_health_check.sh' to check Ollama status"
    log "Run './scripts/monitor_dashboard.sh' to view system dashboard"
}

# Run main function
main "$@"