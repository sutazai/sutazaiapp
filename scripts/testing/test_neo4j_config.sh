#!/bin/bash

# Neo4j Configuration Test and Recovery Script
# This script tests Neo4j startup with the new configuration

set -e


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

echo "🔧 Neo4j Configuration Test Script"
echo "=================================="

# Ensure network exists
echo "📡 Ensuring network exists..."
docker network create sutazai-network 2>/dev/null || echo "Network already exists"

# Start Neo4j with the updated configuration
echo "🚀 Starting Neo4j with updated configuration..."
cd /opt/sutazaiapp
docker-compose up -d neo4j

# Wait for Neo4j to initialize
echo "⏳ Waiting for Neo4j to start (this may take up to 60 seconds)..."
for i in {1..12}; do
    if docker logs sutazai-neo4j 2>&1 | grep -q "Started"; then
        echo "✅ Neo4j has started successfully!"
        break
    elif docker logs sutazai-neo4j 2>&1 | grep -q "ERROR\|FATAL"; then
        echo "❌ Neo4j startup failed. Checking logs..."
        docker logs sutazai-neo4j --tail 20
        exit 1
    fi
    echo "   ... still starting (attempt $i/12)"
    sleep 5
done

# Check if Neo4j is responding
echo "🔍 Testing Neo4j connectivity..."
for i in {1..6}; do
    if curl -s -f http://127.0.0.1:10002/ > /dev/null 2>&1; then
        echo "✅ Neo4j web interface is accessible at http://localhost:10002"
        break
    fi
    echo "   ... checking connectivity (attempt $i/6)"
    sleep 5
done

# Test Bolt connection
echo "🔗 Testing Bolt protocol connectivity..."
for i in {1..6}; do
    if nc -z 127.0.0.1 10003 2>/dev/null; then
        echo "✅ Neo4j Bolt protocol is accessible on port 10003"
        break
    fi
    echo "   ... checking Bolt connectivity (attempt $i/6)"
    sleep 5
done

# Show container status
echo "📊 Container Status:"
docker ps --filter "name=sutazai-neo4j" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Show recent logs (last 10 lines)
echo ""
echo "📋 Recent Neo4j Logs (last 10 lines):"
docker logs sutazai-neo4j --tail 10

echo ""
echo "🎉 Neo4j configuration test completed!"
echo "   Web UI: http://localhost:10002"
echo "   Bolt:   bolt://localhost:10003" 
echo "   User:   neo4j"
echo "   Pass:   \$NEO4J_PASSWORD (from .env file)"