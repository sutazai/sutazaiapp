#!/bin/bash
"""
Script to fix health checks by restarting containers that don't have them
"""

echo "🏥 SutazAI Health Check Fix Script"
echo "================================="

# Function to check if a container has health check
check_container_health() {
    local container_name=$1
    local status=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$container_name" | awk '{for(i=2;i<=NF;i++) printf "%s ", $i; print ""}')
    
    if [[ "$status" == *"healthy"* ]]; then
        echo "✅ $container_name: healthy"
        return 0
    elif [[ "$status" == *"unhealthy"* ]]; then
        echo "❌ $container_name: unhealthy"
        return 1
    else
        echo "⚠️  $container_name: no health check"
        return 2
    fi
}

# Function to restart a container
restart_container() {
    local container_name=$1
    echo "🔄 Restarting $container_name..."
    docker-compose restart $container_name
}

# List of agent containers that should have health checks
AGENT_CONTAINERS=(
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

# Monitoring services that should have health checks
MONITORING_CONTAINERS=(
    "sutazai-frontend"
    "sutazai-grafana"
    "sutazai-prometheus"
    "sutazai-n8n"
)

echo "Checking agent containers..."
echo "----------------------------"

RESTART_NEEDED=()

for container in "${AGENT_CONTAINERS[@]}"; do
    check_container_health "$container"
    result=$?
    if [ $result -eq 2 ]; then
        RESTART_NEEDED+=("$container")
    fi
done

echo ""
echo "Checking monitoring containers..."
echo "--------------------------------"

for container in "${MONITORING_CONTAINERS[@]}"; do
    check_container_health "$container"
    result=$?
    if [ $result -eq 2 ]; then
        RESTART_NEEDED+=("$container")
    fi
done

echo ""
echo "Summary:"
echo "--------"

if [ ${#RESTART_NEEDED[@]} -eq 0 ]; then
    echo "✅ All containers have health checks configured!"
else
    echo "⚠️  Found ${#RESTART_NEEDED[@]} containers without health checks:"
    for container in "${RESTART_NEEDED[@]}"; do
        echo "   - $container"
    done
    
    echo ""
    read -p "Do you want to restart these containers to apply health checks? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Restarting containers with docker-compose..."
        
        # Restart using the appropriate docker-compose file
        if [ -f "docker-compose.agents-simple.yml" ]; then
            echo "Using docker-compose.agents-simple.yml for agent containers..."
            for container in "${RESTART_NEEDED[@]}"; do
                service_name=$(echo $container | sed 's/sutazai-//')
                echo "Restarting service: $service_name"
                docker-compose -f docker-compose.agents-simple.yml restart "$service_name" 2>/dev/null || echo "Service $service_name not found in agents-simple.yml"
            done
        fi
        
        # Restart monitoring services from main compose file
        for container in "${RESTART_NEEDED[@]}"; do
            if [[ " ${MONITORING_CONTAINERS[@]} " =~ " ${container} " ]]; then
                service_name=$(echo $container | sed 's/sutazai-//')
                echo "Restarting monitoring service: $service_name"
                docker-compose restart "$service_name" 2>/dev/null || echo "Service $service_name not found in main docker-compose.yml"
            fi
        done
        
        echo ""
        echo "⏳ Waiting 30 seconds for containers to start..."
        sleep 30
        
        echo ""
        echo "🔍 Re-checking health status..."
        echo "-----------------------------"
        
        for container in "${RESTART_NEEDED[@]}"; do
            check_container_health "$container"
        done
    else
        echo "❌ Skipping container restart."
    fi
fi

echo ""
echo "🎯 Health check fix complete!"
echo ""
echo "To check all container health status, run:"
echo "docker ps --format 'table {{.Names}}\t{{.Status}}'"