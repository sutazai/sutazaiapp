#!/bin/bash

echo "🔧 Fixing Remaining Health Checks"
echo "================================="

# List of remaining containers that need health checks
AGENT_CONTAINERS=(
    "complex-problem-solver"
    "financial-analysis-specialist"
    "security-pentesting-specialist"
    "kali-security-specialist"
    "infrastructure-devops-manager"
    "system-optimizer-reorganizer"
    "autonomous-system-controller"
    "ai-agent-orchestrator"
    "ollama-integration-specialist"
    "system-architect"
    "shell-automation-specialist"
    "ai-agent-creator"
    "testing-qa-validator"
    "context-optimization-engineer"
    "hardware-resource-optimizer"
)

# Function to recreate a container with health checks
recreate_container() {
    local service_name=$1
    local container_name="sutazai-$service_name"
    
    echo "🔄 Recreating $container_name with health checks..."
    
    # Stop and remove the container
    docker-compose -f docker-compose.agents-simple.yml stop "$service_name" 2>/dev/null
    docker-compose -f docker-compose.agents-simple.yml rm -f "$service_name" 2>/dev/null
    
    # Recreate the container
    docker-compose -f docker-compose.agents-simple.yml up -d "$service_name"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully recreated $container_name"
        return 0
    else
        echo "❌ Failed to recreate $container_name"
        return 1
    fi
}

# Function to check health status
check_health_status() {
    local container_name=$1
    local status=$(docker ps --format "{{.Names}}\t{{.Status}}" | grep "$container_name" | awk '{for(i=2;i<=NF;i++) printf "%s ", $i; print ""}')
    
    if [[ "$status" == *"healthy"* ]]; then
        echo "✅ $container_name: healthy"
    elif [[ "$status" == *"starting"* ]]; then
        echo "🔄 $container_name: starting"
    elif [[ "$status" == *"unhealthy"* ]]; then
        echo "❌ $container_name: unhealthy"
    else
        echo "⚠️  $container_name: no health check"
    fi
}

echo "Starting to fix ${#AGENT_CONTAINERS[@]} agent containers..."
echo ""

# Recreate containers in batches to avoid overloading the system
BATCH_SIZE=3
TOTAL=${#AGENT_CONTAINERS[@]}
CURRENT=0

for service in "${AGENT_CONTAINERS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL] Processing $service..."
    
    recreate_container "$service"
    
    # Wait between batches
    if [ $((CURRENT % BATCH_SIZE)) -eq 0 ] && [ $CURRENT -lt $TOTAL ]; then
        echo ""
        echo "⏳ Waiting 30 seconds before next batch..."
        sleep 30
        echo ""
    fi
done

echo ""
echo "🏁 Finished recreating all agent containers!"
echo ""
echo "⏳ Waiting 2 minutes for all containers to start and become healthy..."
sleep 120

echo ""
echo "🔍 Checking final health status..."
echo "================================"

HEALTHY_COUNT=0
STARTING_COUNT=0
UNHEALTHY_COUNT=0
NO_CHECK_COUNT=0

for service in "${AGENT_CONTAINERS[@]}"; do
    container_name="sutazai-$service"
    status=$(docker ps --format "{{.Names}}\t{{.Status}}" | grep "$container_name" | awk '{for(i=2;i<=NF;i++) printf "%s ", $i; print ""}')
    
    if [[ "$status" == *"healthy"* ]]; then
        echo "✅ $container_name: healthy"
        HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
    elif [[ "$status" == *"starting"* ]]; then
        echo "🔄 $container_name: starting"
        STARTING_COUNT=$((STARTING_COUNT + 1))
    elif [[ "$status" == *"unhealthy"* ]]; then
        echo "❌ $container_name: unhealthy"
        UNHEALTHY_COUNT=$((UNHEALTHY_COUNT + 1))
    else
        echo "⚠️  $container_name: no health check"
        NO_CHECK_COUNT=$((NO_CHECK_COUNT + 1))
    fi
done

echo ""
echo "📊 Final Summary:"
echo "================"
echo "✅ Healthy:        $HEALTHY_COUNT"
echo "🔄 Starting:       $STARTING_COUNT"
echo "❌ Unhealthy:      $UNHEALTHY_COUNT"
echo "⚠️  No Healthcheck: $NO_CHECK_COUNT"
echo ""

# Calculate success rate
TOTAL_PROCESSED=${#AGENT_CONTAINERS[@]}
SUCCESS_COUNT=$((HEALTHY_COUNT + STARTING_COUNT))
SUCCESS_RATE=$((SUCCESS_COUNT * 100 / TOTAL_PROCESSED))

echo "🎯 Success Rate: $SUCCESS_RATE% ($SUCCESS_COUNT/$TOTAL_PROCESSED containers with health checks)"

if [ $SUCCESS_RATE -ge 90 ]; then
    echo "🎉 Excellent! Most containers now have working health checks!"
elif [ $SUCCESS_RATE -ge 70 ]; then
    echo "👍 Good progress! Most containers are working."
else
    echo "⚠️  Some containers may need manual attention."
fi

echo ""
echo "💡 Next steps:"
echo "- Run 'python3 scripts/validate_health_checks.py' for detailed validation"
echo "- Check individual container logs: docker logs [container-name]"
echo "- Monitor health status: docker ps"