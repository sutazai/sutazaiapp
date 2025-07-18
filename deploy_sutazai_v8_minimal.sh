#!/bin/bash
# SutazAI v8 Minimal Deployment Script
# Core services deployment for demonstration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

echo -e "${BLUE}"
echo "=================================================================="
echo "🚀 SUTAZAI V8 MINIMAL DEPLOYMENT"
echo "=================================================================="
echo "Version: 2.0.0"
echo "Core Services: Backend, Frontend, Vector DBs, FAISS, Awesome Code AI"
echo "=================================================================="
echo -e "${NC}"

log "Starting SutazAI v8 minimal deployment..."

# Check prerequisites
log "Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    error "Docker is not installed"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    error "Docker Compose is not available"
    exit 1
fi

log "✅ Prerequisites check passed"

# Create directories
log "Creating required directories..."
mkdir -p data/{models,documents,uploads,backups,logs}
mkdir -p docker/{faiss,awesome-code-ai,enhanced-model-manager,backend,streamlit}

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    log "Creating .env file..."
    cat > .env << EOF
POSTGRES_PASSWORD=sutazai_secure_password_$(openssl rand -hex 8)
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
EOF
fi

# Stop existing services
log "Stopping any existing services..."
docker compose -f docker-compose-minimal.yml down --remove-orphans 2>/dev/null || true

# Clean up
log "Cleaning up Docker environment..."
docker system prune -f || true

# Start core services
log "Starting core SutazAI v8 services..."
docker compose -f docker-compose-minimal.yml up --build -d

# Wait for services
log "Waiting for services to start..."
sleep 30

# Check service health
log "Checking service health..."
services=("postgres" "redis" "qdrant" "chromadb" "ollama")

for service in "${services[@]}"; do
    if docker compose -f docker-compose-minimal.yml ps "$service" | grep -q "Up"; then
        log "✅ $service is running"
    else
        warn "⚠️ $service may not be running properly"
    fi
done

# Print results
echo -e "${GREEN}"
echo "=================================================================="
echo "🎉 SUTAZAI V8 MINIMAL DEPLOYMENT COMPLETED"
echo "=================================================================="
echo -e "${NC}"

running_services=$(docker compose -f docker-compose-minimal.yml ps --services --filter status=running | wc -l)
total_services=$(docker compose -f docker-compose-minimal.yml ps --services | wc -l)

echo "📊 Services Status: $running_services/$total_services running"
echo
echo "🌐 Access Points:"
echo "   • Backend API: http://localhost:8000"
echo "   • Frontend UI: http://localhost:8501"
echo "   • ChromaDB: http://localhost:8001"
echo "   • Qdrant: http://localhost:6333"
echo "   • Ollama: http://localhost:11434"
echo "   • FAISS: http://localhost:8096"
echo "   • Awesome Code AI: http://localhost:8097"
echo "   • Enhanced Model Manager: http://localhost:8098"
echo
echo "📋 Next Steps:"
echo "   1. Check service logs: docker compose -f docker-compose-minimal.yml logs -f"
echo "   2. Test API: curl http://localhost:8000/health"
echo "   3. Access documentation: http://localhost:8000/docs"
echo
echo "✅ SutazAI v8 core system is ready!"
echo "=================================================================="