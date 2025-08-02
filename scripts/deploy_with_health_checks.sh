#!/bin/bash

echo "🚀 Deploying SutazAI with Health Checks"
echo "======================================"

# Function to stop and remove containers
cleanup_containers() {
    echo "🧹 Cleaning up existing containers..."
    
    # Get list of containers to restart
    CONTAINERS_TO_RESTART=(
        "sutazai-senior-ai-engineer"
        "sutazai-deployment-automation-master"
        "sutazai-infrastructure-devops-manager"
        "sutazai-ollama-integration-specialist"
        "sutazai-testing-qa-validator"
        "sutazai-ai-agent-creator"
        "sutazai-ai-agent-orchestrator"
        "sutazai-complex-problem-solver"
        "sutazai-financial-analysis-specialist"
        "sutazai-security-pentesting-specialist"
        "sutazai-kali-security-specialist"
        "sutazai-shell-automation-specialist"
        "sutazai-hardware-resource-optimizer"
        "sutazai-context-optimization-engineer"
        "sutazai-system-optimizer-reorganizer"
        "sutazai-system-architect"
        "sutazai-autonomous-system-controller"
    )
    
    # Stop containers
    for container in "${CONTAINERS_TO_RESTART[@]}"; do
        if docker ps -q -f name="$container" | grep -q .; then
            echo "Stopping $container..."
            docker stop "$container" || true
        fi
    done
    
    # Remove containers
    for container in "${CONTAINERS_TO_RESTART[@]}"; do
        if docker ps -aq -f name="$container" | grep -q .; then
            echo "Removing $container..."
            docker rm "$container" || true
        fi
    done
}

# Function to deploy agents with health checks
deploy_agents() {
    echo "🤖 Deploying AI agents with health checks..."
    
    # Deploy using agents-simple compose file
    if [ -f "docker-compose.agents-simple.yml" ]; then
        echo "Starting agent deployment..."
        docker-compose -f docker-compose.agents-simple.yml up -d
        
        if [ $? -eq 0 ]; then
            echo "✅ Agent containers deployed successfully!"
        else
            echo "❌ Error deploying agent containers"
            return 1
        fi
    else
        echo "❌ docker-compose.agents-simple.yml not found"
        return 1
    fi
}

# Function to wait for containers to be ready
wait_for_containers() {
    echo "⏳ Waiting for containers to start..."
    sleep 30
    
    echo "🔍 Checking container health status..."
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep sutazai | head -20
}

# Function to validate deployment
validate_deployment() {
    echo "🔬 Validating health checks..."
    
    if [ -f "scripts/validate_health_checks.py" ]; then
        python3 scripts/validate_health_checks.py
        return $?
    else
        echo "⚠️  Validation script not found, skipping detailed validation"
        return 0
    fi
}

# Main deployment flow
main() {
    echo "Starting deployment process..."
    echo ""
    
    # Step 1: Cleanup existing containers
    cleanup_containers
    echo ""
    
    # Step 2: Deploy agents with health checks
    deploy_agents
    if [ $? -ne 0 ]; then
        echo "❌ Deployment failed at agent deployment stage"
        exit 1
    fi
    echo ""
    
    # Step 3: Wait for containers
    wait_for_containers
    echo ""
    
    # Step 4: Validate deployment
    validate_deployment
    validation_result=$?
    echo ""
    
    if [ $validation_result -eq 0 ]; then
        echo "🎉 Deployment completed successfully!"
        echo "All containers are running with health checks configured."
    else
        echo "⚠️  Deployment completed with issues."
        echo "Some containers may need manual attention."
    fi
    
    echo ""
    echo "📊 Final status check:"
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(healthy|unhealthy|starting)" | head -10
    
    echo ""
    echo "🔗 Useful commands:"
    echo "  - Check all container status: docker ps"
    echo "  - View logs: docker-compose -f docker-compose.agents-simple.yml logs [service-name]"
    echo "  - Restart a service: docker-compose -f docker-compose.agents-simple.yml restart [service-name]"
    echo "  - Validate health: python3 scripts/validate_health_checks.py"
    
    return $validation_result
}

# Check if running as script
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi