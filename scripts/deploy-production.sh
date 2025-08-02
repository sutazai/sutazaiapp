#!/bin/bash
set -euo pipefail

# Production deployment script for SutazAI

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CLUSTER_NAME="sutazai-prod"
NAMESPACE="sutazai"
REGION="us-west-2"
ENVIRONMENT="production"

# Function to print colored output
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check required tools
    for tool in kubectl aws helm terraform docker; do
        if ! command -v $tool &> /dev/null; then
            error "$tool is not installed"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured"
        exit 1
    fi
    
    # Check kubectl context
    if ! kubectl config current-context | grep -q $CLUSTER_NAME; then
        warning "Current kubectl context does not match production cluster"
        log "Updating kubeconfig..."
        aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION
    fi
    
    log "Prerequisites check passed"
}

# Build and push Docker images
build_images() {
    log "Building Docker images..."
    
    # Get commit hash for tagging
    COMMIT_HASH=$(git rev-parse --short HEAD)
    VERSION="${VERSION:-$COMMIT_HASH}"
    
    # Build backend
    log "Building backend image..."
    docker build -f docker/production/backend.Dockerfile -t sutazai/backend:$VERSION .
    docker tag sutazai/backend:$VERSION sutazai/backend:latest
    
    # Build frontend
    log "Building frontend image..."
    docker build -f docker/production/frontend.Dockerfile -t sutazai/frontend:$VERSION .
    docker tag sutazai/frontend:$VERSION sutazai/frontend:latest
    
    # Push to registry
    log "Pushing images to registry..."
    docker push sutazai/backend:$VERSION
    docker push sutazai/backend:latest
    docker push sutazai/frontend:$VERSION
    docker push sutazai/frontend:latest
    
    log "Images built and pushed successfully"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log "Deploying infrastructure with Terraform..."
    
    cd terraform/environments/prod
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -out=tfplan
    
    # Apply with approval
    read -p "Do you want to apply the Terraform plan? (yes/no): " -n 3 -r
    echo
    if [[ $REPLY =~ ^yes$ ]]; then
        terraform apply tfplan
    else
        error "Terraform deployment cancelled"
        exit 1
    fi
    
    cd -
    log "Infrastructure deployed successfully"
}

# Deploy Kubernetes resources
deploy_kubernetes() {
    log "Deploying Kubernetes resources..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy with Kustomize
    kubectl apply -k k8s/overlays/prod
    
    # Wait for deployments
    log "Waiting for deployments to be ready..."
    kubectl -n $NAMESPACE wait --for=condition=available --timeout=600s deployment --all
    
    log "Kubernetes resources deployed successfully"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Create migration job
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration-$(date +%s)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: migrate
        image: sutazai/backend:$VERSION
        command: ["alembic", "upgrade", "head"]
        envFrom:
        - secretRef:
            name: backend-secret
        - configMapRef:
            name: backend-config
EOF
    
    # Wait for migration to complete
    kubectl -n $NAMESPACE wait --for=condition=complete --timeout=300s job/db-migration-$(date +%s)
    
    log "Database migrations completed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus stack
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values monitoring/helm/prometheus-values.yaml \
        --wait
    
    # Deploy Loki
    helm upgrade --install loki grafana/loki-stack \
        --namespace monitoring \
        --values monitoring/helm/loki-values.yaml \
        --wait
    
    log "Monitoring stack deployed"
}

# Run smoke tests
run_smoke_tests() {
    log "Running smoke tests..."
    
    # Get service endpoints
    API_ENDPOINT=$(kubectl -n $NAMESPACE get service backend -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    # Health check
    if curl -f "http://$API_ENDPOINT/health" > /dev/null 2>&1; then
        log "API health check passed"
    else
        error "API health check failed"
        exit 1
    fi
    
    # Basic API test
    if curl -f "http://$API_ENDPOINT/api/v1/status" > /dev/null 2>&1; then
        log "API status check passed"
    else
        error "API status check failed"
        exit 1
    fi
    
    log "Smoke tests passed"
}

# Main deployment flow
main() {
    log "Starting production deployment for SutazAI"
    
    # Confirmation
    warning "You are about to deploy to PRODUCTION"
    read -p "Are you sure you want to continue? (yes/no): " -n 3 -r
    echo
    if [[ ! $REPLY =~ ^yes$ ]]; then
        error "Deployment cancelled"
        exit 1
    fi
    
    # Run deployment steps
    check_prerequisites
    build_images
    deploy_infrastructure
    deploy_kubernetes
    run_migrations
    deploy_monitoring
    run_smoke_tests
    
    log "Production deployment completed successfully!"
    log "Access the application at: https://sutazai.ai"
}

# Run main function
main "$@"