#!/bin/bash
# Start SutazAI with optional features based on environment variables
# Supports command-line overrides and health checks

set -e

# Colors for output

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    print_info "Environment variables loaded from .env"
fi

# Parse command line arguments
ENABLE_FSDP=${ENABLE_FSDP:-false}
ENABLE_TABBY=${ENABLE_TABBY:-false}

while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-fsdp)
            ENABLE_FSDP=true
            shift
            ;;
        --enable-tabby)
            ENABLE_TABBY=true
            shift
            ;;
        --minimal)
            ENABLE_FSDP=false
            ENABLE_TABBY=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --enable-fsdp    Enable FSDP distributed training"
            echo "  --enable-tabby   Enable TabbyML code completion"
            echo "  --minimal        Start with minimal features only"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_info "Starting SutazAI..."
echo "  FSDP:    $ENABLE_FSDP"
echo "  TabbyML: $ENABLE_TABBY"

# Build profiles string
PROFILES=""

if [ "$ENABLE_FSDP" = "true" ]; then
    echo "Enabling FSDP service..."
    PROFILES="$PROFILES --profile fsdp"
fi

if [ "$ENABLE_TABBY" = "true" ]; then
    echo "Enabling TabbyML service..."
    PROFILES="$PROFILES --profile tabby"
fi

# Ensure network exists
docker network create sutazai-network 2>/dev/null || true

# Start core services (always run)
echo "Starting core services..."
docker-compose up -d \
    postgres \
    redis \
    neo4j \
    ollama \
    backend \
    frontend \
    prometheus \
    grafana \
    loki

# Start optional services if enabled
if [ -n "$PROFILES" ]; then
    echo "Starting optional services with profiles:$PROFILES"
    docker-compose $PROFILES up -d
fi

# Show status
print_info "Services started. Checking status..."
sleep 5

# Function to check service health
check_service() {
    local service=$1
    local port=$2
    local endpoint=${3:-/health}
    
    if curl -s -f "http://localhost:$port$endpoint" > /dev/null 2>&1; then
        echo -e "  $service: ${GREEN}✓${NC}"
        return 0
    else
        echo -e "  $service: ${YELLOW}starting...${NC}"
        return 1
    fi
}

print_info "Service Status:"
check_service "Backend API" 10010 /health
check_service "Frontend" 10011 /
check_service "Ollama" 10104 /api/tags
check_service "Prometheus" 10200 /-/healthy
check_service "Grafana" 10201 /api/health

if [ "$ENABLE_FSDP" = "true" ]; then
    check_service "FSDP Training" 8596 /health
fi

if [ "$ENABLE_TABBY" = "true" ]; then
    check_service "TabbyML" 10303 /health
fi

echo ""
print_info "SutazAI is starting up..."
echo "  Backend API: http://localhost:10010"
echo "  Frontend UI: http://localhost:10011"
echo "  Grafana: http://localhost:10201 (admin/admin)"

if [ "$ENABLE_FSDP" = "true" ]; then
    echo "  FSDP Training: http://localhost:8596"
fi

if [ "$ENABLE_TABBY" = "true" ]; then
    echo "  TabbyML: http://localhost:10303"
fi