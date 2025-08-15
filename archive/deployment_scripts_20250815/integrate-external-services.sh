#!/bin/bash
# Purpose: Integrate external services into SutazAI ecosystem
# Usage: ./integrate-external-services.sh [--discover] [--deploy] [--monitor]
# Requires: Docker, docker-compose, python3

set -euo pipefail

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
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INTEGRATION_CONFIG="${PROJECT_ROOT}/external_services_integration.json"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.external-integration.yml"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    print_success "All prerequisites met"
}

# Function to discover external services
discover_services() {
    print_status "Discovering external services..."
    
    cd "$PROJECT_ROOT"
    
    # Install required Python packages
    pip3 install -q docker psutil requests pyyaml prometheus-client
    
    # Run discovery script
    python3 scripts/external-service-discovery.py --output json
    
    if [ -f "$INTEGRATION_CONFIG" ]; then
        print_success "Service discovery completed"
        
        # Display discovered services
        echo -e "\n${BLUE}Discovered Services:${NC}"
        python3 -m json.tool "$INTEGRATION_CONFIG" | grep -E '"name"|"port"|"image"' | head -20
    else
        print_warning "No services discovered"
    fi
}

# Function to create configuration files
create_configs() {
    print_status "Creating configuration files..."
    
    # Create config directories
    mkdir -p "${PROJECT_ROOT}/configs"/{prometheus,loki,kong,envoy,grafana/provisioning/{dashboards,datasources}}
    
    # Create Prometheus config
    cat > "${PROJECT_ROOT}/configs/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sutazai-adapters'
    static_configs:
      - targets:
          - 'sutazai-postgres-adapter:9090'
          - 'sutazai-redis-adapter:9090'
          - 'sutazai-api-gateway:8001'
    metrics_path: '/metrics'

  - job_name: 'external-services'
    file_sd_configs:
      - files:
          - '/etc/prometheus/targets/*.yml'
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
      - source_labels: [__meta_service_name]
        target_label: service
EOF

    # Create Loki config
    cat > "${PROJECT_ROOT}/configs/loki/loki-config.yaml" << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://localhost:9093
EOF

    # Create Kong config
    cat > "${PROJECT_ROOT}/configs/kong/kong.yml" << 'EOF'
_format_version: "3.0"

services:
  - name: postgres-service
    url: http://sutazai-postgres-adapter:8080
    routes:
      - name: postgres-route
        paths:
          - /postgres
        strip_path: true

  - name: redis-service
    url: http://sutazai-redis-adapter:8080
    routes:
      - name: redis-route
        paths:
          - /redis
        strip_path: true

plugins:
  - name: prometheus
    config:
      per_consumer: false

  - name: request-transformer
    config:
      add:
        headers:
          - X-SutazAI-Gateway:v1.0

  - name: rate-limiting
    config:
      minute: 100
      policy: local
EOF

    # Create Envoy config
    cat > "${PROJECT_ROOT}/configs/envoy/envoy.yaml" << 'EOF'
admin:
  address:
    socket_address:
      address: 0.0.0.0
      port_value: 9901

static_resources:
  listeners:
    - name: listener_0
      address:
        socket_address:
          address: 0.0.0.0
          port_value: 10000
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: ingress_http
                access_log:
                  - name: envoy.access_loggers.stdout
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: local_service
                      domains: ["*"]
                      routes:
                        - match:
                            prefix: "/"
                          route:
                            cluster: service_cluster
                http_filters:
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router

  clusters:
    - name: service_cluster
      connect_timeout: 30s
      type: LOGICAL_DNS
      lb_policy: ROUND_ROBIN
      load_assignment:
        cluster_name: service_cluster
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: sutazai-api-gateway
                      port_value: 8000
EOF

    # Create Grafana datasources
    cat > "${PROJECT_ROOT}/configs/grafana/provisioning/datasources/datasources.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://sutazai-metrics-aggregator:9090
    isDefault: true

  - name: Loki
    type: loki
    access: proxy
    url: http://sutazai-log-collector:3100
EOF

    print_success "Configuration files created"
}

# Function to build adapter images
build_adapters() {
    print_status "Building adapter images..."
    
    cd "$PROJECT_ROOT"
    
    # Build base adapter image
    docker build -t sutazaiapp/service-adapter:latest -f docker/adapters/Dockerfile.service-adapter docker/adapters/
    
    # Build specific adapters if their directories exist
    if [ -d "docker/adapters/postgres" ]; then
        docker build -t sutazaiapp/postgres-adapter:latest -f docker/adapters/postgres/Dockerfile docker/adapters/postgres/
    fi
    
    print_success "Adapter images built"
}

# Function to deploy integration services
deploy_services() {
    print_status "Deploying integration services..."
    
    cd "$PROJECT_ROOT"
    
    # Start the integration stack
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    print_status "Waiting for services to become healthy..."
    sleep 10
    
    # Check service health
    services=("sutazai-service-discovery" "sutazai-api-gateway" "sutazai-metrics-aggregator")
    all_healthy=true
    
    for service in "${services[@]}"; do
        if docker ps --filter "name=$service" --filter "health=healthy" | grep -q "$service"; then
            print_success "$service is healthy"
        else
            print_warning "$service is not healthy yet"
            all_healthy=false
        fi
    done
    
    if $all_healthy; then
        print_success "All integration services deployed successfully"
    else
        print_warning "Some services are still starting up. Check 'docker ps' for status"
    fi
}

# Function to setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring dashboards..."
    
    # Wait for Grafana to be ready
    until curl -s http://localhost:10050/api/health > /dev/null; do
        echo -n "."
        sleep 2
    done
    echo ""
    
    # Create API key for dashboard import
    GRAFANA_API_KEY=$(curl -s -X POST http://admin:admin@localhost:10050/api/auth/keys \
        -H "Content-Type: application/json" \
        -d '{"name":"sutazai-integration","role":"Admin"}' | jq -r .key)
    
    if [ -n "$GRAFANA_API_KEY" ]; then
        print_success "Grafana API key created"
        
        # Import dashboards would go here
        print_status "Dashboard import functionality to be implemented"
    else
        print_warning "Failed to create Grafana API key"
    fi
}

# Function to display integration status
show_status() {
    print_status "Integration Status:"
    echo ""
    
    echo -e "${BLUE}Service URLs:${NC}"
    echo "  • Service Discovery: http://localhost:10000"
    echo "  • API Gateway: http://localhost:10001"
    echo "  • Metrics (Prometheus): http://localhost:10010"
    echo "  • Logs (Loki): http://localhost:10020"
    echo "  • Service Mesh Admin: http://localhost:10031"
    echo "  • Config Service (Consul): http://localhost:10040"
    echo "  • Dashboard (Grafana): http://localhost:10050 (admin/admin)"
    echo ""
    
    echo -e "${BLUE}Adapter Endpoints:${NC}"
    echo "  • PostgreSQL Adapter: http://localhost:10100"
    echo "  • Redis Adapter: http://localhost:10110"
    echo ""
    
    echo -e "${BLUE}Running Containers:${NC}"
    docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up integration services..."
    
    cd "$PROJECT_ROOT"
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    print_success "Integration services stopped"
}

# Main function
main() {
    case "${1:-}" in
        --discover)
            check_prerequisites
            discover_services
            ;;
        --deploy)
            check_prerequisites
            create_configs
            build_adapters
            deploy_services
            show_status
            ;;
        --monitor)
            check_prerequisites
            setup_monitoring
            ;;
        --status)
            show_status
            ;;
        --cleanup)
            cleanup
            ;;
        --all)
            check_prerequisites
            discover_services
            create_configs
            build_adapters
            deploy_services
            setup_monitoring
            show_status
            ;;
        *)
            echo "Usage: $0 [--discover|--deploy|--monitor|--status|--cleanup|--all]"
            echo ""
            echo "Options:"
            echo "  --discover  Discover external services"
            echo "  --deploy    Deploy integration infrastructure"
            echo "  --monitor   Setup monitoring dashboards"
            echo "  --status    Show integration status"
            echo "  --cleanup   Stop and remove integration services"
            echo "  --all       Run all steps"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"