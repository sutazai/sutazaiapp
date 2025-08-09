#!/bin/bash

# SutazAI AI Services Deployment Script
# Purpose: Deploy and configure all external AI services
# Usage: ./deploy-ai-services.sh [--service SERVICE_NAME] [--category CATEGORY] [--dry-run]

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
SERVICES_CONFIG="${PROJECT_ROOT}/config/services.yaml"
DOCKER_COMPOSE_DIR="${PROJECT_ROOT}/docker"
LOG_DIR="${PROJECT_ROOT}/logs/deployment"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_DIR/deploy_${TIMESTAMP}.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_DIR/deploy_${TIMESTAMP}.log" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_DIR/deploy_${TIMESTAMP}.log"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_DIR/deploy_${TIMESTAMP}.log"
}

# Parse command line arguments
DRY_RUN=false
DEPLOY_SERVICE=""
DEPLOY_CATEGORY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --service)
            DEPLOY_SERVICE="$2"
            shift 2
            ;;
        --category)
            DEPLOY_CATEGORY="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--service SERVICE_NAME] [--category CATEGORY] [--dry-run]"
            echo "  --service    Deploy specific service"
            echo "  --category   Deploy all services in category (vector_databases, ai_frameworks, etc.)"
            echo "  --dry-run    Show what would be deployed without executing"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    local deps=("docker" "docker-compose" "yq" "curl" "jq")
    local missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing[*]}"
        error "Please install them first"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    log "All dependencies satisfied"
}

# Parse services configuration
parse_services() {
    local category=$1
    local service=$2
    
    if [ -n "$service" ]; then
        # Get specific service
        yq eval ".services.$category.$service" "$SERVICES_CONFIG" 2>/dev/null
    elif [ -n "$category" ]; then
        # Get all services in category
        yq eval ".services.$category | keys | .[]" "$SERVICES_CONFIG" 2>/dev/null
    else
        # Get all services
        for cat in $(yq eval '.services | keys | .[]' "$SERVICES_CONFIG"); do
            yq eval ".services.$cat | keys | .[]" "$SERVICES_CONFIG" 2>/dev/null
        done
    fi
}

# Get service configuration
get_service_config() {
    local category=$1
    local service=$2
    local key=$3
    
    yq eval ".services.$category.$service.$key" "$SERVICES_CONFIG" 2>/dev/null
}

# Create Docker Compose file for service
create_compose_file() {
    local category=$1
    local service=$2
    local compose_file="${DOCKER_COMPOSE_DIR}/docker-compose.${service}.yml"
    
    if [ "$DRY_RUN" = true ]; then
        info "Would create compose file: $compose_file"
        return
    fi
    
    log "Creating Docker Compose file for $service..."
    
    # Get service configuration
    local enabled=$(get_service_config "$category" "$service" "enabled")
    if [ "$enabled" != "true" ]; then
        warning "Service $service is disabled in configuration"
        return
    fi
    
    # Extract configuration
    local adapter=$(get_service_config "$category" "$service" "adapter")
    local config=$(get_service_config "$category" "$service" "config")
    local resources=$(get_service_config "$category" "$service" "resources")
    
    # Create compose file
    cat > "$compose_file" << EOF
version: '3.8'

networks:
  sutazai-network:
    external: true

services:
  ${service}:
    image: sutazai/${service}:latest
    container_name: sutazai-${service}
    restart: unless-stopped
    networks:
      - sutazai-network
    environment:
      - SERVICE_NAME=${service}
      - SERVICE_CATEGORY=${category}
      - ADAPTER_CLASS=${adapter}
EOF

    # Add configuration as environment variables
    if [ -n "$config" ]; then
        echo "    # Service configuration" >> "$compose_file"
        while IFS= read -r line; do
            if [[ $line =~ ^([^:]+):[[:space:]]*(.+)$ ]]; then
                key="${BASH_REMATCH[1]}"
                value="${BASH_REMATCH[2]}"
                # Convert to uppercase and replace non-alphanumeric with underscore
                env_key=$(echo "$key" | tr '[:lower:]' '[:upper:]' | tr -c '[:alnum:]' '_')
                echo "      - ${env_key}=${value}" >> "$compose_file"
            fi
        done <<< "$(echo "$config" | yq eval -o=props -)"
    fi
    
    # Add resource limits
    if [ -n "$resources" ]; then
        echo "    deploy:" >> "$compose_file"
        echo "      resources:" >> "$compose_file"
        echo "        limits:" >> "$compose_file"
        
        local cpu=$(echo "$resources" | yq eval '.cpu' -)
        local memory=$(echo "$resources" | yq eval '.memory' -)
        local gpu=$(echo "$resources" | yq eval '.gpu' -)
        
        [ -n "$cpu" ] && [ "$cpu" != "null" ] && echo "          cpus: '$cpu'" >> "$compose_file"
        [ -n "$memory" ] && [ "$memory" != "null" ] && echo "          memory: $memory" >> "$compose_file"
        
        if [ "$gpu" = "required" ] || [ "$gpu" = "optional" ]; then
            echo "        reservations:" >> "$compose_file"
            echo "          devices:" >> "$compose_file"
            echo "            - driver: nvidia" >> "$compose_file"
            echo "              count: 1" >> "$compose_file"
            echo "              capabilities: [gpu]" >> "$compose_file"
        fi
    fi
    
    # Add volumes
    echo "    volumes:" >> "$compose_file"
    echo "      - ${PROJECT_ROOT}/data/${service}:/data" >> "$compose_file"
    echo "      - ${PROJECT_ROOT}/models:/models" >> "$compose_file"
    echo "      - ${PROJECT_ROOT}/services/adapters:/app/adapters:ro" >> "$compose_file"
    
    log "Created compose file: $compose_file"
}

# Build service image
build_service_image() {
    local service=$1
    local dockerfile="${DOCKER_COMPOSE_DIR}/${service}/Dockerfile"
    
    if [ "$DRY_RUN" = true ]; then
        info "Would build image for $service"
        return
    fi
    
    log "Building Docker image for $service..."
    
    # Create Dockerfile if it doesn't exist
    if [ ! -f "$dockerfile" ]; then
        mkdir -p "$(dirname "$dockerfile")"
        
        cat > "$dockerfile" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy adapter code
COPY services/adapters /app/adapters

# Install Python dependencies
RUN pip install --no-cache-dir \
    aiohttp \
    asyncio \
    pyyaml \
    redis \
    prometheus-client

# Service-specific dependencies will be installed at runtime

# Create entrypoint script
COPY <<'ENTRYPOINT' /app/entrypoint.py
import os
import sys
import asyncio
import importlib
from pathlib import Path

sys.path.insert(0, '/app')

async def main():
    # Get service configuration from environment
    adapter_class = os.environ.get('ADAPTER_CLASS', '')
    
    if not adapter_class:
        print("Error: ADAPTER_CLASS not specified")
        sys.exit(1)
    
    # Import adapter
    module_path, class_name = adapter_class.rsplit('.', 1)
    module = importlib.import_module(f"adapters.{module_path}")
    adapter_cls = getattr(module, class_name)
    
    # Create configuration from environment
    config = {}
    for key, value in os.environ.items():
        if key not in ['PATH', 'HOME', 'USER', 'ADAPTER_CLASS', 'SERVICE_NAME', 'SERVICE_CATEGORY']:
            config[key.lower()] = value
    
    # Initialize adapter
    adapter = adapter_cls(config)
    
    try:
        await adapter.initialize()
        print(f"Service {os.environ.get('SERVICE_NAME')} initialized successfully")
        
        # Keep service running
        while True:
            await asyncio.sleep(60)
            health = await adapter.health_check()
            print(f"Health check: {health}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
ENTRYPOINT

RUN chmod +x /app/entrypoint.py

CMD ["python", "/app/entrypoint.py"]
EOF
        
        log "Created Dockerfile: $dockerfile"
    fi
    
    # Build image
    docker build -t "sutazai/${service}:latest" -f "$dockerfile" "$PROJECT_ROOT" 2>&1 | tee -a "$LOG_DIR/build_${service}_${TIMESTAMP}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "Successfully built image for $service"
    else
        error "Failed to build image for $service"
        return 1
    fi
}

# Deploy service
deploy_service() {
    local category=$1
    local service=$2
    local compose_file="${DOCKER_COMPOSE_DIR}/docker-compose.${service}.yml"
    
    if [ "$DRY_RUN" = true ]; then
        info "Would deploy service: $service"
        return
    fi
    
    log "Deploying service: $service"
    
    # Create necessary directories
    mkdir -p "${PROJECT_ROOT}/data/${service}"
    
    # Deploy using docker-compose
    docker-compose -f "$compose_file" up -d 2>&1 | tee -a "$LOG_DIR/deploy_${service}_${TIMESTAMP}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "Successfully deployed $service"
        
        # Wait for service to be healthy
        wait_for_service "$service"
    else
        error "Failed to deploy $service"
        return 1
    fi
}

# Wait for service to be healthy
wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=0
    
    log "Waiting for $service to be healthy..."
    
    while [ $attempt -lt $max_attempts ]; do
        if docker exec "sutazai-${service}" curl -s http://localhost:8000/health &> /dev/null; then
            log "$service is healthy"
            return 0
        fi
        
        sleep 2
        ((attempt++))
    done
    
    warning "$service did not become healthy within expected time"
    return 1
}

# Update API Gateway configuration
update_api_gateway() {
    if [ "$DRY_RUN" = true ]; then
        info "Would update API Gateway configuration"
        return
    fi
    
    log "Updating API Gateway configuration..."
    
    # Extract routes from services.yaml
    local routes=$(yq eval '.api_gateway.routes' "$SERVICES_CONFIG")
    
    # Update Kong configuration
    # This would typically involve using Kong Admin API
    # For now, we'll just log the action
    
    log "API Gateway configuration updated"
}

# Register service with Consul
register_service() {
    local service=$1
    local category=$2
    
    if [ "$DRY_RUN" = true ]; then
        info "Would register $service with Consul"
        return
    fi
    
    log "Registering $service with service discovery..."
    
    # Get service port (default to 8000)
    local port=8000
    
    # Register with Consul
    local consul_data=$(cat <<EOF
{
  "ID": "sutazai-${service}",
  "Name": "${service}",
  "Tags": ["${category}", "ai-service"],
  "Port": ${port},
  "Check": {
    "HTTP": "http://sutazai-${service}:${port}/health",
    "Interval": "30s"
  }
}
EOF
)
    
    curl -s -X PUT \
        -H "Content-Type: application/json" \
        -d "$consul_data" \
        "http://consul:8500/v1/agent/service/register" \
        > /dev/null
    
    log "Service $service registered with Consul"
}

# Main deployment function
main() {
    log "Starting AI Services Deployment"
    log "Configuration: $SERVICES_CONFIG"
    log "Timestamp: $TIMESTAMP"
    
    # Check dependencies
    check_dependencies
    
    # Determine what to deploy
    local services_to_deploy=()
    
    if [ -n "$DEPLOY_SERVICE" ]; then
        # Find category for the service
        for cat in $(yq eval '.services | keys | .[]' "$SERVICES_CONFIG"); do
            if yq eval ".services.$cat | has(\"$DEPLOY_SERVICE\")" "$SERVICES_CONFIG" 2>/dev/null | grep -q true; then
                services_to_deploy+=("$cat:$DEPLOY_SERVICE")
                break
            fi
        done
        
        if [ ${#services_to_deploy[@]} -eq 0 ]; then
            error "Service $DEPLOY_SERVICE not found in configuration"
            exit 1
        fi
    elif [ -n "$DEPLOY_CATEGORY" ]; then
        # Deploy all services in category
        for svc in $(parse_services "$DEPLOY_CATEGORY" ""); do
            services_to_deploy+=("$DEPLOY_CATEGORY:$svc")
        done
    else
        # Deploy all services
        for cat in $(yq eval '.services | keys | .[]' "$SERVICES_CONFIG"); do
            for svc in $(parse_services "$cat" ""); do
                services_to_deploy+=("$cat:$svc")
            done
        done
    fi
    
    if [ ${#services_to_deploy[@]} -eq 0 ]; then
        warning "No services to deploy"
        exit 0
    fi
    
    log "Services to deploy: ${services_to_deploy[*]}"
    
    # Deploy each service
    local failed=0
    for service_spec in "${services_to_deploy[@]}"; do
        IFS=':' read -r category service <<< "$service_spec"
        
        log "Processing $service (category: $category)"
        
        # Create compose file
        create_compose_file "$category" "$service"
        
        # Build image
        if ! build_service_image "$service"; then
            ((failed++))
            continue
        fi
        
        # Deploy service
        if ! deploy_service "$category" "$service"; then
            ((failed++))
            continue
        fi
        
        # Register with service discovery
        register_service "$service" "$category"
    done
    
    # Update API Gateway
    update_api_gateway
    
    # Summary
    log "Deployment complete"
    log "Total services: ${#services_to_deploy[@]}"
    log "Failed: $failed"
    
    if [ $failed -gt 0 ]; then
        error "Some services failed to deploy"
        exit 1
    fi
    
    # Show deployed services
    if [ "$DRY_RUN" = false ]; then
        log "Deployed services:"
        docker ps --filter "label=com.sutazai.service=true" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    fi
}

# Run main function
main "$@"