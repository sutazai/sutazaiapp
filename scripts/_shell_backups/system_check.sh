#!/bin/bash
# SutazAI Full System Validation Script

set -eo pipefail

declare -A COMPONENTS=(
    ["Models"]="validate_models"
    ["Agents"]="validate_agents"
    ["Backend"]="validate_backend"
    ["WebUI"]="validate_ui"
    ["Networking"]="validate_network"
)

for component in "${!COMPONENTS[@]}"; do
    echo "🔍 Validating $component..."
    start_time=$(date +%s)
    
    if ! python3 -c "from validations import ${COMPONENTS[$component]}; ${COMPONENTS[$component]}()"; then
        echo "❌ $component validation failed!"
        exit 1
    fi
    
    duration=$(( $(date +%s) - start_time ))
    echo "✅ $component passed (${duration}s)"
done

echo "🚀 All systems operational"
echo "🌐 Access URL: https://$(hostname -I | awk '{print $1}')"

# AI Agent Subsystem Validation

components=(
    "Agent Services"
    "Hardware Config" 
    "Dependencies"
)

for component in "${components[@]}"; do
    echo "🔍 Checking $component..."
    if ! python3 -c "from agent_validations import validate_$component"; then
        echo "❌ $component failed!"
        exit 1
    fi
    echo "✅ $component passed"
done

echo "🤖 All AI agent systems operational" 