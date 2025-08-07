#!/bin/bash
# Add Prometheus metrics to all agent services

echo "=== Adding Prometheus Metrics to All Agents ==="

# List of agent directories
AGENTS=(
    "task_assignment_coordinator"
    "resource_arbitration_agent"
    "ai_agent_orchestrator"
    "hardware-resource-optimizer"
    "coordinator"
)

# Copy metrics module to each agent
for agent in "${AGENTS[@]}"; do
    agent_dir="/opt/sutazaiapp/agents/$agent"
    
    if [ -d "$agent_dir" ]; then
        echo "Processing $agent..."
        
        # Copy metrics module
        cp /opt/sutazaiapp/agents/core/metrics.py "$agent_dir/"
        
        # Add prometheus-client to requirements if not present
        if [ -f "$agent_dir/requirements.txt" ]; then
            if ! grep -q "prometheus-client" "$agent_dir/requirements.txt"; then
                echo "prometheus-client==0.19.0" >> "$agent_dir/requirements.txt"
            fi
        else
            # Create requirements.txt if it doesn't exist
            cat > "$agent_dir/requirements.txt" <<EOF
fastapi==0.104.1
uvicorn==0.24.0
prometheus-client==0.19.0
pydantic==2.5.0
EOF
        fi
        
        # Check if Dockerfile exists
        if [ -f "$agent_dir/Dockerfile" ]; then
            # Check if metrics copy is in Dockerfile
            if ! grep -q "metrics.py" "$agent_dir/Dockerfile"; then
                echo "  Note: Update Dockerfile to copy metrics.py"
            fi
        fi
        
        echo "  ✓ Metrics module added"
    else
        echo "  ⚠ Agent directory not found: $agent_dir"
    fi
done

echo ""
echo "=== Next Steps ==="
echo "1. Update each agent's app.py to import and initialize metrics"
echo "2. Add metrics tracking to key endpoints"
echo "3. Update Dockerfiles to include metrics.py"
echo "4. Rebuild and restart containers"
echo ""
echo "Example metrics initialization in app.py:"
echo ""
cat <<'EOF'
from agents.core.metrics import AgentMetrics, setup_metrics_endpoint

# In startup/lifespan function:
metrics = AgentMetrics("agent_name")
setup_metrics_endpoint(app, metrics)

# Track requests:
@metrics.track_request("method_name")
async def your_endpoint():
    # Your code here
EOF