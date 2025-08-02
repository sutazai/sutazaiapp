#!/bin/bash
# Quick script to check Docker health and restart if needed

echo "Checking Docker health..."

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Starting it..."
    
    # Kill any existing dockerd
    sudo pkill -9 dockerd 2>/dev/null
    sleep 2
    
    # Clean up common issues
    sudo rm -f /var/lib/docker/volumes/metadata.db
    sudo rm -rf /var/lib/docker/network
    sudo rm -f /var/run/docker.sock /var/run/docker.pid
    
    # Start Docker
    sudo dockerd > /tmp/docker_restart.log 2>&1 &
    
    # Wait for it to start
    sleep 10
    
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker started successfully!"
    else
        echo "❌ Docker failed to start. Check /tmp/docker_restart.log"
        tail -20 /tmp/docker_restart.log
    fi
else
    echo "✅ Docker is already running"
fi

docker version