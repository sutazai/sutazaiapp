#!/bin/bash
# Test SutazAI setup in isolated Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${1}"
}

success() {
    log "${GREEN}✅ ${1}${NC}"
}

info() {
    log "${BLUE}ℹ️  ${1}${NC}"
}

warn() {
    log "${YELLOW}⚠️  ${1}${NC}"
}

error() {
    log "${RED}❌ ${1}${NC}"
}

# Header
log "${BLUE}"
log "=============================================="
log "🧪 SutazAI Setup Container Test"
log "=============================================="
log "${NC}"

# Create required directories
info "Creating required directories..."
mkdir -p logs data backups secrets configs

# Build and start the setup container
info "Building setup container..."
docker-compose -f docker-compose.setup.yml build

info "Starting setup container..."
docker-compose -f docker-compose.setup.yml up -d

# Monitor setup progress
info "Monitoring setup progress..."
docker-compose -f docker-compose.setup.yml logs -f sutazai-setup &
LOGS_PID=$!

# Wait for setup to complete (check for specific completion message)
TIMEOUT=1800  # 30 minutes timeout
ELAPSED=0
INTERVAL=10

while [ $ELAPSED -lt $TIMEOUT ]; do
    if docker-compose -f docker-compose.setup.yml logs sutazai-setup | grep -q "Setup completed successfully"; then
        success "Setup completed successfully!"
        break
    fi
    
    if docker-compose -f docker-compose.setup.yml logs sutazai-setup | grep -q "ERROR:"; then
        error "Setup failed with errors. Check logs for details."
        break
    fi
    
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    
    # Show progress
    info "Setup in progress... (${ELAPSED}/${TIMEOUT}s)"
done

# Kill the log monitoring
kill $LOGS_PID 2>/dev/null || true

if [ $ELAPSED -ge $TIMEOUT ]; then
    warn "Setup timed out after ${TIMEOUT} seconds"
fi

# Test the deployed system
info "Testing deployed system..."

# Function to test endpoint
test_endpoint() {
    local url=$1
    local name=$2
    local timeout=${3:-10}
    
    if timeout $timeout curl -s -f "$url" > /dev/null 2>&1; then
        success "$name is responding"
        return 0
    else
        error "$name is not responding"
        return 1
    fi
}

# Wait a bit for services to fully start
sleep 30

# Test endpoints
test_endpoint "http://localhost:18501" "Frontend (Streamlit)" 15
test_endpoint "http://localhost:18000/health" "Backend API" 15
test_endpoint "http://localhost:13000/api/health" "Grafana" 15
test_endpoint "http://localhost:19090/-/healthy" "Prometheus" 15
test_endpoint "http://localhost:21434/api/tags" "Ollama" 30

# Test backend API endpoints
info "Testing backend API endpoints..."
if curl -s -f "http://localhost:18000/api/v1/agents" > /dev/null 2>&1; then
    success "Agents endpoint is working"
else
    error "Agents endpoint is not working"
fi

if curl -s -f "http://localhost:18000/api/v1/system/metrics" > /dev/null 2>&1; then
    success "Metrics endpoint is working"
else
    error "Metrics endpoint is not working"
fi

# Test model availability
info "Testing AI models..."
if curl -s "http://localhost:21434/api/tags" | jq -r '.models[].name' | grep -q "deepseek-r1:8b"; then
    success "DeepSeek R1 model is available"
else
    warn "DeepSeek R1 model is not available"
fi

# Show container status
info "Container status:"
docker-compose -f docker-compose.setup.yml ps

# Show resource usage
info "System resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Generate test report
log "${GREEN}"
log "=============================================="
log "🎉 SutazAI Container Test Complete!"
log "=============================================="
log "${NC}"

log "${BLUE}Access Points:${NC}"
log "🌐 Frontend UI:      http://localhost:18501"
log "🔌 Backend API:      http://localhost:18000"
log "📚 API Docs:         http://localhost:18000/docs"
log "📊 Monitoring:       http://localhost:13000"
log "📈 Prometheus:       http://localhost:19090"
log ""

log "${BLUE}Container Management:${NC}"
log "📋 View logs:        docker-compose -f docker-compose.setup.yml logs"
log "⏹️  Stop container:   docker-compose -f docker-compose.setup.yml down"
log "🔍 Enter container:   docker exec -it sutazai-setup-container bash"
log ""

log "${YELLOW}Note: Container will remain running for manual testing.${NC}"
log "${YELLOW}Use 'docker-compose -f docker-compose.setup.yml down' to stop.${NC}"