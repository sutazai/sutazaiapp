#!/bin/bash
# SutazAI Auto-scaling Deployment Script
# Deploys auto-scaling infrastructure for Kubernetes, Docker Swarm, or Docker Compose

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
AUTOSCALING_DIR="$PROJECT_ROOT/deployment/autoscaling"

# Default values
PLATFORM="${1:-compose}"
ENVIRONMENT="${2:-local}"
NAMESPACE="${3:-sutazai}"

# Validate platform
if [[ ! "$PLATFORM" =~ ^(kubernetes|k8s|swarm|compose)$ ]]; then
    log_error "Invalid platform: $PLATFORM"
    echo "Usage: $0 [kubernetes|swarm|compose] [production|staging|local] [namespace]"
    exit 1
fi

# Normalize platform name
[[ "$PLATFORM" == "k8s" ]] && PLATFORM="kubernetes"

log_info "Deploying auto-scaling for platform: $PLATFORM, environment: $ENVIRONMENT"

# Function to check prerequisites
check_prerequisites() {
    case "$PLATFORM" in
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl not found. Please install kubectl first."
                exit 1
            fi
            
            # Check if cluster is accessible
            if ! kubectl cluster-info &> /dev/null; then
                log_error "Cannot connect to Kubernetes cluster"
                exit 1
            fi
            
            # Check if metrics-server is installed
            if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
                log_warning "metrics-server not found. Installing..."
                kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
            fi
            ;;
            
        swarm)
            if ! command -v docker &> /dev/null; then
                log_error "Docker not found. Please install Docker first."
                exit 1
            fi
            
            # Check if Swarm is initialized
            if ! docker info 2>/dev/null | grep -q "Swarm: active"; then
                log_error "Docker Swarm not initialized. Run 'docker swarm init' first."
                exit 1
            fi
            ;;
            
        compose)
            if ! command -v docker-compose &> /dev/null; then
                log_error "docker-compose not found. Please install docker-compose first."
                exit 1
            fi
            ;;
    esac
}

# Function to deploy Kubernetes auto-scaling
deploy_kubernetes() {
    log_info "Deploying Kubernetes auto-scaling components..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy HPA configurations
    log_info "Deploying Horizontal Pod Autoscalers..."
    kubectl apply -f "$AUTOSCALING_DIR/hpa-enhanced.yaml" -n "$NAMESPACE"
    
    # Deploy VPA configurations
    log_info "Deploying Vertical Pod Autoscalers..."
    
    # Check if VPA is installed
    if ! kubectl get crd verticalpodautoscalers.autoscaling.k8s.io &> /dev/null; then
        log_warning "VPA CRDs not found. Installing VPA..."
        git clone https://github.com/kubernetes/autoscaler.git /tmp/autoscaler || true
        cd /tmp/autoscaler/vertical-pod-autoscaler
        ./hack/vpa-up.sh
        cd -
    fi
    
    kubectl apply -f "$AUTOSCALING_DIR/vpa-config.yaml" -n "$NAMESPACE"
    
    # Deploy load balancing
    log_info "Deploying load balancing configurations..."
    kubectl apply -f "$AUTOSCALING_DIR/load-balancing/" -n "$NAMESPACE"
    
    # Deploy monitoring components
    log_info "Deploying AI metrics exporter..."
    kubectl apply -f "$AUTOSCALING_DIR/monitoring/ai-metrics-exporter.yaml" -n "$NAMESPACE"
    
    # Deploy core services if they don't exist
    if [[ -f "$AUTOSCALING_DIR/kubernetes/core-services.yaml" ]]; then
        kubectl apply -f "$AUTOSCALING_DIR/kubernetes/core-services.yaml" -n "$NAMESPACE"
    fi
    
    log_success "Kubernetes auto-scaling deployed successfully!"
    
    # Show status
    echo -e "\n${BLUE}HPA Status:${NC}"
    kubectl get hpa -n "$NAMESPACE"
    
    echo -e "\n${BLUE}VPA Status:${NC}"
    kubectl get vpa -n "$NAMESPACE"
}

# Function to deploy Docker Swarm auto-scaling
deploy_swarm() {
    log_info "Deploying Docker Swarm auto-scaling components..."
    
    # Deploy stack
    cd "$AUTOSCALING_DIR/swarm"
    
    # Create necessary configs
    docker config create nginx_config nginx.conf 2>/dev/null || \
        docker config rm nginx_config && docker config create nginx_config nginx.conf
    
    docker config create upstream_config upstream.conf 2>/dev/null || \
        docker config rm upstream_config && docker config create upstream_config upstream.conf
    
    # Deploy the stack
    docker stack deploy -c docker-compose.swarm.yml sutazai
    
    # Start the autoscaler
    log_info "Starting Swarm autoscaler..."
    docker service create \
        --name sutazai-swarm-autoscaler \
        --mount type=bind,source=/var/run/docker.sock,destination=/var/run/docker.sock \
        --env PROMETHEUS_URL=http://prometheus:9090 \
        --env MIN_REPLICAS=1 \
        --env MAX_REPLICAS=10 \
        --network sutazaiapp_sutazai-network \
        --replicas 1 \
        python:3.11-slim sh -c "pip install docker aiohttp prometheus-client && python /app/swarm-autoscaler.py" \
        || docker service update sutazai-swarm-autoscaler
    
    # Copy autoscaler script
    docker cp swarm-autoscaler.py "$(docker ps -q -f name=sutazai-swarm-autoscaler | head -1)":/app/swarm-autoscaler.py
    
    log_success "Docker Swarm auto-scaling deployed successfully!"
    
    # Show status
    docker service ls | grep sutazai
}

# Function to deploy Docker Compose auto-scaling
deploy_compose() {
    log_info "Deploying Docker Compose auto-scaling simulation..."
    
    cd "$PROJECT_ROOT"
    
    # Create auto-scaling compose file
    cat > docker-compose.autoscaling.yml <<EOF
version: '3.8'

services:
  # Auto-scaling simulator for development
  autoscaler:
    image: python:3.11-slim
    container_name: sutazai-autoscaler
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./deployment/autoscaling/swarm/swarm-autoscaler.py:/app/autoscaler.py
    environment:
      - PROMETHEUS_URL=http://prometheus:9090
      - MIN_REPLICAS=1
      - MAX_REPLICAS=5
      - PLATFORM=compose
    command: >
      sh -c "pip install docker aiohttp prometheus-client &&
             python /app/autoscaler.py"
    networks:
      - sutazaiapp_sutazai-network
    restart: unless-stopped

  # Load balancer
  nginx:
    image: nginx:alpine
    container_name: sutazai-nginx
    ports:
      - "8080:80"
    volumes:
      - ./deployment/autoscaling/swarm/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./deployment/autoscaling/swarm/upstream.conf:/etc/nginx/conf.d/upstream.conf:ro
    depends_on:
      - backend
    networks:
      - sutazaiapp_sutazai-network
    restart: unless-stopped

networks:
  sutazaiapp_sutazai-network:
    external: true
EOF
    
    # Start auto-scaling components
    docker-compose -f docker-compose.yml -f docker-compose.autoscaling.yml up -d
    
    log_success "Docker Compose auto-scaling simulation deployed!"
    log_warning "Note: Docker Compose doesn't support true auto-scaling. This is a simulation for development."
    
    # Show status
    docker-compose ps | grep -E "(autoscaler|nginx)"
}

# Function to verify deployment
verify_deployment() {
    log_info "Verifying auto-scaling deployment..."
    
    case "$PLATFORM" in
        kubernetes)
            # Check HPA
            if kubectl get hpa -n "$NAMESPACE" | grep -q sutazai; then
                log_success "HPA configured correctly"
            else
                log_warning "HPA not found"
            fi
            
            # Check VPA
            if kubectl get vpa -n "$NAMESPACE" | grep -q sutazai; then
                log_success "VPA configured correctly"
            else
                log_warning "VPA not found"
            fi
            
            # Check metrics
            if kubectl top nodes &> /dev/null; then
                log_success "Metrics server working"
            else
                log_warning "Metrics server not responding"
            fi
            ;;
            
        swarm)
            # Check autoscaler service
            if docker service ls | grep -q sutazai-swarm-autoscaler; then
                log_success "Swarm autoscaler running"
            else
                log_error "Swarm autoscaler not found"
            fi
            ;;
            
        compose)
            # Check autoscaler container
            if docker ps | grep -q sutazai-autoscaler; then
                log_success "Compose autoscaler running"
            else
                log_error "Compose autoscaler not found"
            fi
            ;;
    esac
}

# Function to show usage examples
show_usage() {
    echo -e "\n${BLUE}Auto-scaling Usage Examples:${NC}"
    
    case "$PLATFORM" in
        kubernetes)
            cat <<EOF

# Watch HPA status
kubectl get hpa -n $NAMESPACE -w

# Generate load for testing
kubectl run -i --tty load-generator --rm --image=busybox --restart=Never -- /bin/sh

# Inside the pod:
while true; do wget -q -O- http://sutazai-backend:8000/api/health; done

# Check scaling events
kubectl describe hpa sutazai-backend-hpa -n $NAMESPACE
EOF
            ;;
            
        swarm)
            cat <<EOF

# Watch service scaling
watch docker service ls

# Generate load for testing
docker run --rm -it --network sutazaiapp_sutazai-network alpine sh -c \\
  "apk add --no-cache curl && while true; do curl -s http://sutazai-backend:8000/api/health; done"

# Check autoscaler logs
docker service logs sutazai-swarm-autoscaler -f
EOF
            ;;
            
        compose)
            cat <<EOF

# Watch container status
watch docker ps

# Generate load for testing
docker run --rm -it --network sutazaiapp_sutazai-network alpine sh -c \\
  "apk add --no-cache curl && while true; do curl -s http://backend:8000/api/health; done"

# Check autoscaler logs
docker logs sutazai-autoscaler -f
EOF
            ;;
    esac
}

# Main execution
main() {
    log_info "Starting SutazAI auto-scaling deployment..."
    
    # Check prerequisites
    check_prerequisites
    
    # Deploy based on platform
    case "$PLATFORM" in
        kubernetes)
            deploy_kubernetes
            ;;
        swarm)
            deploy_swarm
            ;;
        compose)
            deploy_compose
            ;;
    esac
    
    # Verify deployment
    sleep 5
    verify_deployment
    
    # Show usage examples
    show_usage
    
    log_success "Auto-scaling deployment completed!"
}

# Run main function
main