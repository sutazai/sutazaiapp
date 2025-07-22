#!/bin/bash
#
# SutazAI AI Agents Deployment - Fixed Version
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
AGENTS_DIR="${PROJECT_ROOT}/agents"
LOG_FILE="${PROJECT_ROOT}/logs/agents_deployment_fixed_$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p "${PROJECT_ROOT}/logs"

# Logging functions
log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    log "$1" "$GREEN"
}

warn() {
    log "$1" "$YELLOW"
}

error() {
    log "$1" "$RED"
}

info() {
    log "$1" "$CYAN"
}

# Check if container already exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^$1$"
}

# Deploy agents using available images
deploy_image_based_agents() {
    log "Deploying image-based agents..." "$BLUE"
    
    # TabbyML already started successfully
    if container_exists "sutazai-tabbyml"; then
        log "TabbyML already deployed" "$GREEN"
    fi
    
    # Skip LocalAGI as the image doesn't exist, use alternative
    log "Skipping LocalAGI (image not available), will use alternative" "$YELLOW"
    
    # Start LangFlow
    if ! container_exists "sutazai-langflow"; then
        log "Starting LangFlow..." "$YELLOW"
        docker run -d \
            --name sutazai-langflow \
            --network sutazai-network \
            -p 8090:7860 \
            -v langflow_data:/app/langflow \
            -e LANGFLOW_DATABASE_URL=postgresql://sutazai:sutazai123@sutazai-postgres:5432/sutazai_main \
            langflowai/langflow:latest || warn "Failed to start LangFlow"
    else
        log "LangFlow already exists" "$YELLOW"
    fi
    
    # Start FlowiseAI
    if ! container_exists "sutazai-flowise"; then
        log "Starting FlowiseAI..." "$YELLOW"
        docker run -d \
            --name sutazai-flowise \
            --network sutazai-network \
            -p 8099:3000 \
            -v flowise_data:/root/.flowise \
            flowiseai/flowise:latest || warn "Failed to start FlowiseAI"
    else
        log "FlowiseAI already exists" "$YELLOW"
    fi
    
    # Start Dify
    if ! container_exists "sutazai-dify"; then
        log "Starting Dify..." "$YELLOW"
        docker run -d \
            --name sutazai-dify \
            --network sutazai-network \
            -p 8083:3000 \
            -v dify_data:/app/api/storage \
            -e DB_HOST=sutazai-postgres \
            -e DB_PORT=5432 \
            -e DB_USERNAME=sutazai \
            -e DB_PASSWORD=sutazai123 \
            -e DB_DATABASE=sutazai_main \
            -e REDIS_HOST=sutazai-redis \
            -e REDIS_PORT=6379 \
            langgenius/dify:latest || warn "Failed to start Dify"
    else
        log "Dify already exists" "$YELLOW"
    fi
    
    # Start n8n as workflow automation
    if ! container_exists "sutazai-n8n"; then
        log "Starting n8n..." "$YELLOW"
        docker run -d \
            --name sutazai-n8n \
            --network sutazai-network \
            -p 8085:5678 \
            -v n8n_data:/home/node/.n8n \
            -e N8N_BASIC_AUTH_ACTIVE=false \
            -e DB_TYPE=postgresdb \
            -e DB_POSTGRESDB_HOST=sutazai-postgres \
            -e DB_POSTGRESDB_PORT=5432 \
            -e DB_POSTGRESDB_DATABASE=sutazai_main \
            -e DB_POSTGRESDB_USER=sutazai \
            -e DB_POSTGRESDB_PASSWORD=sutazai123 \
            n8nio/n8n:latest || warn "Failed to start n8n"
    else
        log "n8n already exists" "$YELLOW"
    fi
    
    # Start Open WebUI (alternative to some missing agents)
    if ! container_exists "sutazai-openwebui"; then
        log "Starting Open WebUI..." "$YELLOW"
        docker run -d \
            --name sutazai-openwebui \
            --network sutazai-network \
            -p 8086:8080 \
            -v openwebui_data:/app/backend/data \
            -e OLLAMA_API_BASE_URL=http://sutazai-ollama:11434/api \
            ghcr.io/open-webui/open-webui:main || warn "Failed to start Open WebUI"
    else
        log "Open WebUI already exists" "$YELLOW"
    fi
    
    success "Image-based agents deployment completed"
}

# Build and deploy custom agents
deploy_custom_agents() {
    log "Building and deploying custom agents..." "$BLUE"
    
    local agents=(
        "autogpt:8080"
        "browser-use:8084"
        "documind:8092"
        "finrobot:8093"
        "gpt-engineer:8094"
        "aider:8095"
        "crewai:8096"
        "llamaindex:8098"
        "jax:8089"
        "realtime-stt:8101"
        "shellgpt:8102"
        "pentestgpt:8100"
    )
    
    for agent_port in "${agents[@]}"; do
        agent="${agent_port%:*}"
        port="${agent_port#*:}"
        
        if container_exists "sutazai-${agent}"; then
            log "${agent} already exists, skipping..." "$YELLOW"
            continue
        fi
        
        if [[ -f "${AGENTS_DIR}/${agent}/Dockerfile" ]]; then
            log "Building ${agent}..." "$YELLOW"
            
            # Build with timeout
            timeout 300 docker build -t "sutazai-${agent}" "${AGENTS_DIR}/${agent}/" || {
                warn "Failed to build ${agent} (timeout or error)"
                continue
            }
            
            log "Starting ${agent}..." "$YELLOW"
            docker run -d \
                --name "sutazai-${agent}" \
                --network sutazai-network \
                -p "${port}:8080" \
                -v "${agent}_data:/app/data" \
                "sutazai-${agent}" || warn "Failed to start ${agent}"
            
            success "${agent} deployed on port ${port}"
        else
            warn "Dockerfile not found for ${agent}, skipping"
        fi
    done
}

# Quick deploy for critical agents
quick_deploy_critical() {
    log "Quick deploying critical agents..." "$BLUE"
    
    # Deploy Semgrep for code analysis
    if ! container_exists "sutazai-semgrep"; then
        docker run -d \
            --name sutazai-semgrep \
            --network sutazai-network \
            -p 8091:8080 \
            -v semgrep_data:/src \
            returntocorp/semgrep:latest || warn "Failed to start Semgrep"
    fi
    
    # Deploy Skyvern for browser automation
    if ! container_exists "sutazai-skyvern"; then
        docker run -d \
            --name sutazai-skyvern \
            --network sutazai-network \
            -p 8097:8080 \
            -v skyvern_data:/app/data \
            ghcr.io/skyvern-ai/skyvern:latest || warn "Failed to start Skyvern"
    fi
}

# Verify agent deployments
verify_agents() {
    log "Verifying agent deployments..." "$BLUE"
    
    log "Checking running containers..." "$CYAN"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai || true
    
    log "\nSummary of deployed agents:" "$CYAN"
    local total_agents=$(docker ps -a --format '{{.Names}}' | grep -c sutazai || echo "0")
    local running_agents=$(docker ps --format '{{.Names}}' | grep -c sutazai || echo "0")
    
    log "Total SutazAI containers: $total_agents" "$CYAN"
    log "Running containers: $running_agents" "$CYAN"
    
    # Show agent access URLs
    log "\nAgent Access URLs:" "$PURPLE"
    log "- AutoGPT: http://localhost:8080" "$CYAN"
    log "- TabbyML: http://localhost:8082" "$CYAN"
    log "- Dify: http://localhost:8083" "$CYAN"
    log "- Browser-Use: http://localhost:8084" "$CYAN"
    log "- n8n: http://localhost:8085" "$CYAN"
    log "- Open WebUI: http://localhost:8086" "$CYAN"
    log "- LangFlow: http://localhost:8090" "$CYAN"
    log "- FlowiseAI: http://localhost:8099" "$CYAN"
    log "- And more..." "$CYAN"
}

# Main deployment function
main() {
    log "Starting SutazAI AI Agents Deployment (Fixed Version)..." "$PURPLE"
    
    # Deploy agents
    deploy_image_based_agents
    sleep 5
    quick_deploy_critical
    sleep 5
    deploy_custom_agents
    
    # Wait for services to start
    log "Waiting for agents to initialize..." "$YELLOW"
    sleep 10
    
    # Verify deployment
    verify_agents
    
    success "SutazAI AI Agents Deployment completed!"
    log "View logs at: ${LOG_FILE}" "$CYAN"
}

# Run main function
main "$@"