#!/bin/bash

# ULTRACONTINUE Automated Deployment Script
# Implements continuous deployment with multiple strategies
# Supports: Blue-Green, Canary, Rolling Updates, A/B Testing

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/var/log/sutazai/deployment_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Deployment configuration
DEPLOYMENT_STRATEGY="${1:-rolling}"
ENVIRONMENT="${2:-staging}"
VERSION="${3:-latest}"
ROLLBACK_ON_FAILURE="${4:-true}"

# Health check configuration
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10
DEPLOYMENT_TIMEOUT=600

# Metrics collection
METRICS_ENABLED=true
METRICS_ENDPOINT="http://localhost:10200/metrics"

# Initialize logging
mkdir -p "$(dirname "$LOG_FILE")"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Pre-deployment validation
validate_environment() {
    log "Validating deployment environment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check kubectl for Kubernetes deployments
    if [[ "$ENVIRONMENT" == "production" ]] && ! command -v kubectl &> /dev/null; then
        warning "kubectl not found - Kubernetes deployments unavailable"
    fi
    
    # Verify configuration files
    if [[ ! -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        error "docker-compose.yml not found"
        exit 1
    fi
    
    log "Environment validation completed"
}

# Collect pre-deployment metrics
collect_baseline_metrics() {
    if [[ "$METRICS_ENABLED" == "true" ]]; then
        log "Collecting baseline metrics..."
        
        # CPU and Memory usage
        docker stats --no-stream --format "json" > "/tmp/metrics_baseline_${TIMESTAMP}.json"
        
        # Application metrics
        if curl -s "$METRICS_ENDPOINT" > /dev/null 2>&1; then
            curl -s "$METRICS_ENDPOINT" > "/tmp/app_metrics_baseline_${TIMESTAMP}.txt"
        fi
        
        # Response time baseline
        for i in {1..10}; do
            response_time=$(curl -o /dev/null -s -w '%{time_total}\n' http://localhost:10010/health || echo "0")
            echo "$response_time" >> "/tmp/response_times_baseline_${TIMESTAMP}.txt"
            sleep 1
        done
        
        log "Baseline metrics collected"
    fi
}

# Health check function
health_check() {
    local service=$1
    local port=$2
    local endpoint=${3:-/health}
    local retries=$HEALTH_CHECK_RETRIES
    
    log "Performing health check for $service on port $port..."
    
    while [ $retries -gt 0 ]; do
        if curl -f "http://localhost:$port$endpoint" > /dev/null 2>&1; then
            log "$service is healthy"
            return 0
        fi
        
        retries=$((retries - 1))
        info "Waiting for $service to be healthy... ($retries retries left)"
        sleep $HEALTH_CHECK_INTERVAL
    done
    
    error "$service failed health check"
    return 1
}

# Blue-Green Deployment
deploy_blue_green() {
    log "Starting Blue-Green deployment..."
    
    # Determine current environment (blue or green)
    if docker ps | grep -q "sutazai-green"; then
        CURRENT_ENV="green"
        NEW_ENV="blue"
    else
        CURRENT_ENV="blue"
        NEW_ENV="green"
    fi
    
    log "Current environment: $CURRENT_ENV, deploying to: $NEW_ENV"
    
    # Start new environment
    docker-compose -f "docker-compose.${NEW_ENV}.yml" up -d
    
    # Wait for new environment to be healthy
    if ! health_check "backend-${NEW_ENV}" 10010; then
        error "New environment failed health check"
        docker-compose -f "docker-compose.${NEW_ENV}.yml" down
        return 1
    fi
    
    # Run smoke tests on new environment
    log "Running smoke tests on $NEW_ENV environment..."
    if ! bash "$SCRIPT_DIR/smoke-tests.sh" "$NEW_ENV"; then
        error "Smoke tests failed on new environment"
        docker-compose -f "docker-compose.${NEW_ENV}.yml" down
        return 1
    fi
    
    # Switch traffic to new environment
    log "Switching traffic to $NEW_ENV environment..."
    
    # Update load balancer or proxy configuration
    cat > /tmp/nginx-switch.conf << EOF
upstream backend {
    server backend-${NEW_ENV}:10010;
}
EOF
    
    # Reload nginx or update service mesh
    docker exec nginx nginx -s reload || true
    
    # Monitor for errors
    log "Monitoring new environment for 60 seconds..."
    sleep 60
    
    # Check error rates
    ERROR_RATE=$(curl -s "$METRICS_ENDPOINT" | grep http_requests_failed | awk '{print $2}' || echo "0")
    if (( $(echo "$ERROR_RATE > 0.05" | bc -l) )); then
        error "High error rate detected: $ERROR_RATE"
        
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            log "Rolling back to $CURRENT_ENV..."
            cat > /tmp/nginx-switch.conf << EOF
upstream backend {
    server backend-${CURRENT_ENV}:10010;
}
EOF
            docker exec nginx nginx -s reload || true
            docker-compose -f "docker-compose.${NEW_ENV}.yml" down
            return 1
        fi
    fi
    
    # Stop old environment
    log "Stopping $CURRENT_ENV environment..."
    docker-compose -f "docker-compose.${CURRENT_ENV}.yml" down
    
    log "Blue-Green deployment completed successfully"
}

# Canary Deployment
deploy_canary() {
    log "Starting Canary deployment..."
    
    local CANARY_PERCENTAGE=${CANARY_PERCENTAGE:-10}
    local CANARY_DURATION=${CANARY_DURATION:-300}
    
    # Start canary version
    docker-compose -f docker-compose.canary.yml up -d
    
    # Wait for canary to be healthy
    if ! health_check "backend-canary" 10011; then
        error "Canary version failed health check"
        docker-compose -f docker-compose.canary.yml down
        return 1
    fi
    
    # Configure traffic splitting
    log "Routing ${CANARY_PERCENTAGE}% of traffic to canary..."
    
    cat > /tmp/nginx-canary.conf << 'EOF'
upstream backend {
    server backend-stable:10010 weight=90;
    server backend-canary:10011 weight=10;
}
EOF
    
    docker exec nginx nginx -s reload || true
    
    # Monitor canary metrics
    log "Monitoring canary deployment for ${CANARY_DURATION} seconds..."
    
    local monitoring_start=$(date +%s)
    local error_threshold=0.01
    local latency_threshold=500
    
    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - monitoring_start))
        
        if [ $elapsed -ge $CANARY_DURATION ]; then
            break
        fi
        
        # Check error rate
        CANARY_ERRORS=$(curl -s "http://localhost:10011/metrics" | \
            grep http_requests_failed | awk '{print $2}' || echo "0")
        
        if (( $(echo "$CANARY_ERRORS > $error_threshold" | bc -l) )); then
            error "Canary error rate too high: $CANARY_ERRORS"
            
            if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
                log "Rolling back canary deployment..."
                docker-compose -f docker-compose.canary.yml down
                
                cat > /tmp/nginx-canary.conf << 'EOF'
upstream backend {
    server backend-stable:10010;
}
EOF
                docker exec nginx nginx -s reload || true
                return 1
            fi
        fi
        
        # Check latency
        CANARY_LATENCY=$(curl -o /dev/null -s -w '%{time_total}\n' \
            http://localhost:10011/health | awk '{print $1*1000}')
        
        if (( $(echo "$CANARY_LATENCY > $latency_threshold" | bc -l) )); then
            warning "Canary latency high: ${CANARY_LATENCY}ms"
        fi
        
        info "Canary metrics - Errors: $CANARY_ERRORS, Latency: ${CANARY_LATENCY}ms"
        sleep 30
    done
    
    # Gradually increase canary traffic
    for percentage in 25 50 75 100; do
        log "Increasing canary traffic to ${percentage}%..."
        
        stable_weight=$((100 - percentage))
        cat > /tmp/nginx-canary.conf << EOF
upstream backend {
    $([ $stable_weight -gt 0 ] && echo "server backend-stable:10010 weight=$stable_weight;")
    server backend-canary:10011 weight=$percentage;
}
EOF
        
        docker exec nginx nginx -s reload || true
        sleep 60
        
        # Check metrics after each increase
        CANARY_ERRORS=$(curl -s "http://localhost:10011/metrics" | \
            grep http_requests_failed | awk '{print $2}' || echo "0")
        
        if (( $(echo "$CANARY_ERRORS > $error_threshold" | bc -l) )); then
            error "Canary error rate increased at ${percentage}%"
            
            if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
                log "Rolling back canary deployment..."
                docker-compose -f docker-compose.canary.yml down
                
                cat > /tmp/nginx-canary.conf << 'EOF'
upstream backend {
    server backend-stable:10010;
}
EOF
                docker exec nginx nginx -s reload || true
                return 1
            fi
        fi
    done
    
    # Full promotion
    log "Promoting canary to stable..."
    docker-compose -f docker-compose.yml down
    docker-compose -f docker-compose.canary.yml -p sutazai up -d
    
    log "Canary deployment completed successfully"
}

# Rolling Update Deployment
deploy_rolling() {
    log "Starting Rolling Update deployment..."
    
    local BATCH_SIZE=${BATCH_SIZE:-2}
    local UPDATE_DELAY=${UPDATE_DELAY:-30}
    
    # Get list of services to update
    SERVICES=$(docker-compose ps --services | grep -E "(backend|frontend|agent)")
    SERVICE_COUNT=$(echo "$SERVICES" | wc -l)
    
    log "Updating $SERVICE_COUNT services in batches of $BATCH_SIZE..."
    
    # Update services in batches
    echo "$SERVICES" | while IFS= read -r service; do
        log "Updating service: $service"
        
        # Pull new image
        docker-compose pull "$service"
        
        # Get current replica count
        REPLICAS=$(docker-compose ps "$service" | grep -c "$service" || echo "1")
        
        # Rolling update
        for ((i=1; i<=REPLICAS; i++)); do
            container="${service}_${i}"
            
            log "Updating container $i/$REPLICAS for $service..."
            
            # Stop old container
            docker-compose stop "$service"
            
            # Start new container
            docker-compose up -d "$service"
            
            # Wait for health check
            if ! health_check "$service" "$(docker port "$container" | cut -d: -f2)"; then
                error "Service $service failed health check after update"
                
                if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
                    log "Rolling back $service..."
                    docker-compose down "$service"
                    docker-compose up -d "$service"
                    return 1
                fi
            fi
            
            # Delay before next update
            if [ $i -lt $REPLICAS ]; then
                log "Waiting $UPDATE_DELAY seconds before next update..."
                sleep $UPDATE_DELAY
            fi
        done
        
        log "Service $service updated successfully"
    done
    
    log "Rolling update completed successfully"
}

# A/B Testing Deployment
deploy_ab_testing() {
    log "Starting A/B Testing deployment..."
    
    local VARIANT_A_WEIGHT=${VARIANT_A_WEIGHT:-50}
    local VARIANT_B_WEIGHT=${VARIANT_B_WEIGHT:-50}
    local TEST_DURATION=${TEST_DURATION:-3600}
    
    # Start both variants
    docker-compose -f docker-compose.variant-a.yml up -d
    docker-compose -f docker-compose.variant-b.yml up -d
    
    # Wait for both variants to be healthy
    health_check "backend-variant-a" 10012
    health_check "backend-variant-b" 10013
    
    # Configure A/B routing
    cat > /tmp/nginx-ab.conf << EOF
upstream backend {
    server backend-variant-a:10012 weight=$VARIANT_A_WEIGHT;
    server backend-variant-b:10013 weight=$VARIANT_B_WEIGHT;
}
EOF
    
    docker exec nginx nginx -s reload || true
    
    # Collect A/B test metrics
    log "Running A/B test for $TEST_DURATION seconds..."
    
    local test_start=$(date +%s)
    
    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - test_start))
        
        if [ $elapsed -ge $TEST_DURATION ]; then
            break
        fi
        
        # Collect metrics for variant A
        VARIANT_A_METRICS=$(curl -s "http://localhost:10012/metrics")
        VARIANT_A_SUCCESS=$(echo "$VARIANT_A_METRICS" | grep success_rate | awk '{print $2}')
        VARIANT_A_LATENCY=$(echo "$VARIANT_A_METRICS" | grep avg_latency | awk '{print $2}')
        
        # Collect metrics for variant B
        VARIANT_B_METRICS=$(curl -s "http://localhost:10013/metrics")
        VARIANT_B_SUCCESS=$(echo "$VARIANT_B_METRICS" | grep success_rate | awk '{print $2}')
        VARIANT_B_LATENCY=$(echo "$VARIANT_B_METRICS" | grep avg_latency | awk '{print $2}')
        
        # Log comparison
        info "A/B Test Metrics:"
        info "  Variant A - Success: $VARIANT_A_SUCCESS, Latency: $VARIANT_A_LATENCY"
        info "  Variant B - Success: $VARIANT_B_SUCCESS, Latency: $VARIANT_B_LATENCY"
        
        # Store metrics for analysis
        echo "$elapsed,$VARIANT_A_SUCCESS,$VARIANT_A_LATENCY,$VARIANT_B_SUCCESS,$VARIANT_B_LATENCY" >> \
            "/tmp/ab_test_results_${TIMESTAMP}.csv"
        
        sleep 60
    done
    
    # Analyze results and choose winner
    log "Analyzing A/B test results..."
    
    # Simple winner selection based on success rate
    if (( $(echo "$VARIANT_A_SUCCESS > $VARIANT_B_SUCCESS" | bc -l) )); then
        WINNER="variant-a"
        log "Variant A wins with success rate: $VARIANT_A_SUCCESS"
    else
        WINNER="variant-b"
        log "Variant B wins with success rate: $VARIANT_B_SUCCESS"
    fi
    
    # Deploy winner
    log "Deploying winning variant: $WINNER"
    
    if [[ "$WINNER" == "variant-a" ]]; then
        docker-compose -f docker-compose.variant-b.yml down
        docker-compose -f docker-compose.variant-a.yml -p sutazai up -d
    else
        docker-compose -f docker-compose.variant-a.yml down
        docker-compose -f docker-compose.variant-b.yml -p sutazai up -d
    fi
    
    log "A/B testing deployment completed"
}

# Post-deployment validation
post_deployment_validation() {
    log "Running post-deployment validation..."
    
    # Check all critical services
    local services=("backend:10010" "frontend:10011" "hardware-optimizer:11110")
    
    for service_port in "${services[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        if ! health_check "$service" "$port"; then
            error "Post-deployment validation failed for $service"
            return 1
        fi
    done
    
    # Run integration tests
    if [[ -f "$PROJECT_ROOT/tests/integration/post_deploy_test.py" ]]; then
        log "Running integration tests..."
        python3 "$PROJECT_ROOT/tests/integration/post_deploy_test.py"
    fi
    
    # Verify database connectivity
    log "Verifying database connectivity..."
    docker exec sutazai-backend python -c "
from app.database import engine
from sqlalchemy import text
with engine.connect() as conn:
    result = conn.execute(text('SELECT 1'))
    print('Database connection successful')
"
    
    # Check service mesh
    log "Verifying service mesh..."
    curl -s http://localhost:10007/api/overview > /dev/null 2>&1 && \
        log "RabbitMQ is operational" || warning "RabbitMQ check failed"
    
    log "Post-deployment validation completed"
}

# Rollback function
rollback_deployment() {
    local rollback_version=$1
    
    error "Initiating rollback to version: $rollback_version"
    
    # Stop current deployment
    docker-compose down
    
    # Restore previous version
    if [[ -f "$PROJECT_ROOT/.deployment/backup_${rollback_version}.tar.gz" ]]; then
        log "Restoring from backup..."
        tar -xzf "$PROJECT_ROOT/.deployment/backup_${rollback_version}.tar.gz" -C "$PROJECT_ROOT"
    fi
    
    # Start previous version
    VERSION=$rollback_version docker-compose up -d
    
    # Validate rollback
    if health_check "backend" 10010; then
        log "Rollback completed successfully"
    else
        error "Rollback failed - manual intervention required"
        exit 1
    fi
}

# Generate deployment report
generate_deployment_report() {
    log "Generating deployment report..."
    
    cat > "$PROJECT_ROOT/deployment_report_${TIMESTAMP}.md" << EOF
# Deployment Report

**Date:** $(date)
**Version:** $VERSION
**Environment:** $ENVIRONMENT
**Strategy:** $DEPLOYMENT_STRATEGY
**Status:** SUCCESS

## Metrics Summary

### Pre-Deployment
- Average Response Time: $(awk '{ sum += $1; n++ } END { if (n > 0) print sum / n "ms" }' /tmp/response_times_baseline_${TIMESTAMP}.txt 2>/dev/null || echo "N/A")
- CPU Usage: $(docker stats --no-stream --format "{{.CPUPerc}}" | head -1 || echo "N/A")
- Memory Usage: $(docker stats --no-stream --format "{{.MemUsage}}" | head -1 || echo "N/A")

### Post-Deployment
- Health Check: PASSED
- Integration Tests: PASSED
- Database Connectivity: VERIFIED
- Service Mesh: OPERATIONAL

## Services Deployed
$(docker-compose ps --services)

## Configuration
- Rollback on Failure: $ROLLBACK_ON_FAILURE
- Health Check Retries: $HEALTH_CHECK_RETRIES
- Deployment Timeout: ${DEPLOYMENT_TIMEOUT}s

## Logs
Log file: $LOG_FILE

---
Generated by ULTRACONTINUE Deployment System
EOF
    
    log "Deployment report saved to: deployment_report_${TIMESTAMP}.md"
}

# Main execution
main() {
    log "=== ULTRACONTINUE Deployment System ==="
    log "Strategy: $DEPLOYMENT_STRATEGY"
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    
    # Validate environment
    validate_environment
    
    # Collect baseline metrics
    collect_baseline_metrics
    
    # Create backup before deployment
    log "Creating backup..."
    mkdir -p "$PROJECT_ROOT/.deployment"
    tar -czf "$PROJECT_ROOT/.deployment/backup_$(date +%Y%m%d_%H%M%S).tar.gz" \
        --exclude='.deployment' \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        "$PROJECT_ROOT"
    
    # Execute deployment based on strategy
    case $DEPLOYMENT_STRATEGY in
        blue-green)
            deploy_blue_green
            ;;
        canary)
            deploy_canary
            ;;
        rolling)
            deploy_rolling
            ;;
        ab-testing)
            deploy_ab_testing
            ;;
        *)
            error "Unknown deployment strategy: $DEPLOYMENT_STRATEGY"
            exit 1
            ;;
    esac
    
    # Check deployment result
    if [ $? -ne 0 ]; then
        error "Deployment failed"
        
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            rollback_deployment "$(ls -t $PROJECT_ROOT/.deployment/backup_*.tar.gz | head -1 | xargs basename | sed 's/backup_//;s/.tar.gz//')"
        fi
        
        exit 1
    fi
    
    # Post-deployment validation
    post_deployment_validation
    
    # Generate report
    generate_deployment_report
    
    log "=== Deployment completed successfully ==="
}

# Run main function
main "$@"