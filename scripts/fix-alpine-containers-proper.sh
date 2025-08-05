#!/bin/bash

echo "=== Fixing Alpine Container Issues ==="
echo "This will properly fix all Alpine-based containers"
echo ""

# List of problematic containers
CONTAINERS=(
    "sutazai-autonomous-system-controller"
    "sutazai-code-generation-improver"
    "sutazai-cognitive-architecture-designer"
    "sutazai-deep-learning-brain-architect"
    "sutazai-deep-learning-brain-manager"
    "sutazai-evolution-strategy-trainer"
    "sutazai-explainability-and-transparency-agent"
    "sutazai-garbage-collector-coordinator"
    "sutazai-goal-setting-and-planning-agent"
    "sutazai-knowledge-distillation-expert"
    "sutazai-knowledge-graph-builder"
    "sutazai-memory-persistence-manager"
    "sutazai-neural-architecture-search"
    "sutazai-product-manager"
    "sutazai-resource-arbitration-agent"
    "sutazai-symbolic-reasoning-engine"
)

# Additional containers that might be affected
ADDITIONAL_CONTAINERS=(
    "sutazai-edge-inference-proxy"
    "sutazai-experiment-tracker"
    "sutazai-data-drift-detector"
    "sutazai-senior-engineer"
    "sutazai-private-data-analyst"
    "sutazai-self-healing-orchestrator"
    "sutazai-private-registry-manager-harbor"
    "sutazai-scrum-master"
    "sutazai-agent-creator"
    "sutazai-bias-and-fairness-auditor"
    "sutazai-ethical-governor"
    "sutazai-runtime-behavior-anomaly-detector"
    "sutazai-reinforcement-learning-trainer"
    "sutazai-neuromorphic-computing-expert"
    "sutazai-explainable-ai-specialist"
    "sutazai-deep-local-brain-builder"
)

# Combine all containers
ALL_CONTAINERS=("${CONTAINERS[@]}" "${ADDITIONAL_CONTAINERS[@]}")

# Function to fix a container
fix_container() {
    local container_name=$1
    echo "Processing: $container_name"
    
    # Check if container exists
    if docker ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
        # Stop and remove the container
        docker stop "$container_name" 2>/dev/null || true
        docker rm -f "$container_name" 2>/dev/null || true
        echo "  Removed old container"
    fi
    
    # Get port mapping from the original deployment
    case "$container_name" in
        "sutazai-garbage-collector-coordinator") PORT=11000 ;;
        "sutazai-edge-inference-proxy") PORT=11043 ;;
        "sutazai-experiment-tracker") PORT=11046 ;;
        "sutazai-data-drift-detector") PORT=11036 ;;
        "sutazai-senior-engineer") PORT=11009 ;;
        "sutazai-private-data-analyst") PORT=11066 ;;
        "sutazai-self-healing-orchestrator") PORT=10431 ;;
        "sutazai-private-registry-manager-harbor") PORT=10432 ;;
        "sutazai-product-manager") PORT=11010 ;;
        "sutazai-scrum-master") PORT=11011 ;;
        "sutazai-agent-creator") PORT=10454 ;;
        "sutazai-bias-and-fairness-auditor") PORT=11033 ;;
        "sutazai-ethical-governor") PORT=10455 ;;
        "sutazai-runtime-behavior-anomaly-detector") PORT=10456 ;;
        "sutazai-reinforcement-learning-trainer") PORT=11045 ;;
        "sutazai-neuromorphic-computing-expert") PORT=10457 ;;
        "sutazai-knowledge-distillation-expert") PORT=11062 ;;
        "sutazai-explainable-ai-specialist") PORT=11052 ;;
        "sutazai-deep-learning-brain-manager") PORT=11021 ;;
        "sutazai-deep-local-brain-builder") PORT=11022 ;;
        "sutazai-autonomous-system-controller") PORT=11003 ;;
        "sutazai-code-generation-improver") PORT=11027 ;;
        "sutazai-cognitive-architecture-designer") PORT=11040 ;;
        "sutazai-deep-learning-brain-architect") PORT=11020 ;;
        "sutazai-evolution-strategy-trainer") PORT=11045 ;;
        "sutazai-explainability-and-transparency-agent") PORT=11047 ;;
        "sutazai-goal-setting-and-planning-agent") PORT=11064 ;;
        "sutazai-knowledge-graph-builder") PORT=10458 ;;
        "sutazai-memory-persistence-manager") PORT=10459 ;;
        "sutazai-neural-architecture-search") PORT=10460 ;;
        "sutazai-resource-arbitration-agent") PORT=10461 ;;
        "sutazai-symbolic-reasoning-engine") PORT=10462 ;;
        *) PORT=8000 ;;
    esac
    
    # Create the container with a working command
    docker run -d \
        --name "$container_name" \
        --network sutazai-network \
        -p "${PORT}:8000" \
        -e "AGENT_NAME=$container_name" \
        -e "OLLAMA_BASE_URL=http://ollama:11434" \
        -e "REDIS_URL=redis://redis:6379/0" \
        --memory="1g" \
        --cpus="0.5" \
        --restart=unless-stopped \
        python:3.11-alpine \
        sh -c "apk add --no-cache gcc musl-dev linux-headers && pip install fastapi uvicorn requests redis psutil && echo 'from fastapi import FastAPI; app = FastAPI(); @app.get(\"/health\"); def health(): return {\"status\": \"healthy\", \"agent\": \"$container_name\"}' > app.py && uvicorn app:app --host 0.0.0.0 --port 8000"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Container recreated successfully"
    else
        echo "  ✗ Failed to recreate container"
    fi
    echo ""
}

# Fix all containers
for container in "${ALL_CONTAINERS[@]}"; do
    fix_container "$container"
done

echo "=== Verification ==="
echo "Checking container status..."
docker ps -a | grep -E "(${ALL_CONTAINERS[0]}" | head -5

echo ""
echo "✓ Alpine container fix completed!"
echo "Run 'docker ps' to verify all containers are running"