#!/bin/bash
# Test script for Docker Coordinator System

# Source the functions from deploy_complete_system.sh
source /opt/sutazaiapp/scripts/deploy_complete_system.sh

# Test the Docker Coordinator System
echo "🧠 Testing Super Intelligent Docker Coordinator System"
echo "=============================================="
echo ""

# First, show current Docker state
echo "📊 Current System State:"
echo "   → Docker installed: $(command -v docker >/dev/null 2>&1 && echo "Yes" || echo "No")"
echo "   → Docker running: $(docker version >/dev/null 2>&1 && echo "Yes" || echo "No")"
echo "   → Docker socket exists: $([ -S /var/run/docker.sock ] && echo "Yes" || echo "No")"
echo "   → Dockerd process: $(pgrep -x dockerd >/dev/null && echo "Running (PID: $(pgrep -x dockerd))" || echo "Not running")"
echo ""

# Now test our intelligent system
echo "🧠 Running Docker Coordinator Analysis..."
docker_state=$(analyze_docker_state)
echo "   → Detected State: $docker_state"
echo ""

# Test the full ensure_docker_running_perfectly function
echo "🚀 Testing ensure_docker_running_perfectly()..."
echo "=============================================="
if ensure_docker_running_perfectly; then
    echo ""
    echo "✅ SUCCESS: Docker Coordinator System worked perfectly!"
    echo ""
    echo "📊 Final State:"
    docker version --format 'Docker Version: {{.Server.Version}}'
    docker ps
else
    echo ""
    echo "❌ FAILED: Docker Coordinator System could not start Docker"
fi