#!/bin/bash
set -euo pipefail

# Configuration
BASE_DIR=$(pwd)
LOG_DIR="/var/log/sutazai"
DEPLOYMENT_LOG="${LOG_DIR}/full_deployment_$(date +%Y%m%d_%H%M%S).log"
MAX_RETRIES=3
RETRY_DELAY=30

# Initialize logging
setup_logging() {
    mkdir -p "$LOG_DIR"
    exec > >(tee -a "$DEPLOYMENT_LOG") 2>&1
    echo "📝 Full deployment log initialized at $DEPLOYMENT_LOG"
}

# Error handling
handle_error() {
    local exit_code=$?
    local error_message=$1
    
    echo "❌ Deployment failed: $error_message" | tee -a $DEPLOYMENT_LOG
    echo "🔄 Initiating rollback..." | tee -a $DEPLOYMENT_LOG
    comprehensive_rollback
    echo "📝 Check the full deployment log at: $DEPLOYMENT_LOG"
    exit $exit_code
}

# Comprehensive rollback
comprehensive_rollback() {
    echo "🔄 Starting comprehensive rollback..." | tee -a $DEPLOYMENT_LOG
    
    # Stop all services
    docker-compose -f docker-compose.yml down --remove-orphans || true
    docker-compose -f docker-compose-super.yml down --remove-orphans || true
    docker-compose -f docker-compose-ai.yml down --remove-orphans || true
    
    # Clean up resources
    docker system prune -f
    docker volume prune -f
    docker network prune -f
    
    echo "✅ Rollback completed" | tee -a $DEPLOYMENT_LOG
}

# Run script with retries
run_with_retries() {
    local script=$1
    local retries=$MAX_RETRIES
    local attempt=0
    local success=0
    
    while [ $attempt -lt $retries ]; do
        echo "🔧 Attempt $((attempt + 1)) of $retries: Running $script" | tee -a $DEPLOYMENT_LOG
        if ./$script; then
            success=1
            break
        fi
        attempt=$((attempt + 1))
        sleep $RETRY_DELAY
    done
    
    if [ $success -eq 0 ]; then
        handle_error "Failed to execute $script after $retries attempts"
    fi
}

# Main deployment function
deploy() {
    echo "🚀 Starting automated deployment..." | tee -a $DEPLOYMENT_LOG
    
    # Phase 1: Pre-deployment checks
    echo "🔍 Running pre-deployment checks..." | tee -a $DEPLOYMENT_LOG
    run_with_retries system_audit.sh
    run_with_retries hardware_health.sh
    run_with_retries resource_limits.sh
    
    # Phase 2: System configuration
    echo "⚙️ Configuring system..." | tee -a $DEPLOYMENT_LOG
    run_with_retries performance_tuning.sh
    run_with_retries log_rotation_check.sh
    run_with_retries service_dependency.sh
    
    # Phase 3: Container setup
    echo "🐳 Setting up container environment..." | tee -a $DEPLOYMENT_LOG
    run_with_retries container_virtualization.sh
    
    # Phase 4: Main deployment
    echo "🚀 Deploying services..." | tee -a $DEPLOYMENT_LOG
    run_with_retries deploy_all.sh
    
    # Phase 5: Post-deployment verification
    echo "🔍 Verifying deployment..." | tee -a $DEPLOYMENT_LOG
    run_with_retries log_analysis.sh
    run_with_retries log_integrity.sh
    
    echo "🎉 Deployment completed successfully!" | tee -a $DEPLOYMENT_LOG
}

# Execute deployment
setup_logging
deploy 