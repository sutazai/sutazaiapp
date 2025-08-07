#!/bin/bash
# Fix health checks for running containers

echo "Fixing health checks for containers..."

# List of containers that need curl installed
CONTAINERS_NEED_CURL=(
    "sutazai-private-registry-manager-harbor"
    "sutazai-autonomous-system-controller"
    "sutazai-code-generation-improver"
    "sutazai-cognitive-architecture-designer"
    "sutazai-deep-learning-brain-architect"
    "sutazai-evolution-strategy-trainer"
    "sutazai-explainability-and-transparency-agent"
    "sutazai-secrets-vault-manager-vault"
    "sutazai-cognitive-load-monitor"
    "sutazai-compute-scheduler-and-optimizer"
    "sutazai-prompt-injection-guard"
    "sutazai-ethical-governor"
    "sutazai-bias-and-fairness-auditor"
    "sutazai-agent-creator"
    "sutazai-energy-consumption-optimize"
)

for container in "${CONTAINERS_NEED_CURL[@]}"; do
    echo "Fixing $container..."
    
    # Try to install curl in running container
    docker exec $container sh -c "apk add --no-cache curl 2>/dev/null || apt-get update && apt-get install -y curl 2>/dev/null || yum install -y curl 2>/dev/null" || {
        echo "  ⚠️  Could not install curl in $container"
    }
done

echo "✓ Health check fixes applied"
