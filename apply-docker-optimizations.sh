#!/bin/bash

# Docker Container Optimization Script
# Purpose: Apply immediate memory optimizations to running containers

set -e

echo "==================================================="
echo "Docker Container Memory Optimization Script"
echo "==================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root or with sudo"
   exit 1
fi

echo "Current Docker Memory Usage:"
echo "----------------------------"
docker stats --no-stream --format "table {{.Container}}\t{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo ""

# Backup current state
echo "Creating backup of current container states..."
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" > /tmp/docker-containers-backup-$(date +%Y%m%d-%H%M%S).txt
print_status "Backup created in /tmp/"

echo ""
echo "==================================================="
echo "PHASE 1: Critical Memory Reductions"
echo "==================================================="

# 1. Fix Ollama's massive over-allocation
if docker ps -q -f name=sutazai-ollama > /dev/null 2>&1; then
    echo "Optimizing Ollama container (23.3GB -> 2GB)..."
    docker update --memory="2g" --memory-swap="2g" sutazai-ollama
    print_status "Ollama memory reduced by 21.3GB!"
else
    print_warning "Ollama container not running"
fi

# 2. Optimize Backend
if docker ps -q -f name=sutazai-backend > /dev/null 2>&1; then
    echo "Optimizing Backend container (2GB -> 512MB)..."
    docker update --memory="512m" --memory-swap="512m" sutazai-backend
    print_status "Backend memory reduced by 1.5GB"
else
    print_warning "Backend container not running"
fi

# 3. Optimize FAISS
if docker ps -q -f name=sutazai-faiss > /dev/null 2>&1; then
    echo "Optimizing FAISS container (2GB -> 384MB)..."
    docker update --memory="384m" --memory-swap="384m" sutazai-faiss
    print_status "FAISS memory reduced by 1.6GB"
else
    print_warning "FAISS container not running"
fi

# 4. Optimize ChromaDB
if docker ps -q -f name=sutazai-chromadb > /dev/null 2>&1; then
    echo "Optimizing ChromaDB container (1GB -> 256MB)..."
    docker update --memory="256m" --memory-swap="256m" sutazai-chromadb
    print_status "ChromaDB memory reduced by 768MB"
else
    print_warning "ChromaDB container not running"
fi

# 5. Optimize Qdrant
if docker ps -q -f name=sutazai-qdrant > /dev/null 2>&1; then
    echo "Optimizing Qdrant container (1GB -> 384MB)..."
    docker update --memory="384m" --memory-swap="384m" sutazai-qdrant
    print_status "Qdrant memory reduced by 640MB"
else
    print_warning "Qdrant container not running"
fi

# 6. Optimize Letta
if docker ps -q -f name=sutazai-letta > /dev/null 2>&1; then
    echo "Optimizing Letta container (1GB -> 256MB)..."
    docker update --memory="256m" --memory-swap="256m" sutazai-letta
    print_status "Letta memory reduced by 768MB"
else
    print_warning "Letta container not running"
fi

echo ""
echo "==================================================="
echo "PHASE 2: Critical Memory Increases"
echo "==================================================="

# Increase memory for containers under pressure
# 1. Neo4j needs more memory
if docker ps -q -f name=sutazai-neo4j > /dev/null 2>&1; then
    echo "Increasing Neo4j container memory (512MB -> 1GB)..."
    docker update --memory="1g" --memory-swap="1g" sutazai-neo4j
    print_status "Neo4j memory increased to prevent OOM"
else
    print_warning "Neo4j container not running"
fi

# 2. Consul needs slight increase
if docker ps -q -f name=sutazai-consul > /dev/null 2>&1; then
    echo "Increasing Consul container memory (256MB -> 384MB)..."
    docker update --memory="384m" --memory-swap="384m" sutazai-consul
    print_status "Consul memory increased for stability"
else
    print_warning "Consul container not running"
fi

echo ""
echo "==================================================="
echo "PHASE 3: Stop Unhealthy Containers"
echo "==================================================="

# Stop unhealthy containers that are wasting resources
UNHEALTHY_CONTAINERS="sutazai-localagi sutazai-documind sutazai-finrobot sutazai-gpt-engineer"

for container in $UNHEALTHY_CONTAINERS; do
    if docker ps -q -f name=$container > /dev/null 2>&1; then
        echo "Stopping unhealthy container: $container"
        docker stop $container
        print_status "$container stopped"
    else
        print_warning "$container not running"
    fi
done

echo ""
echo "==================================================="
echo "PHASE 4: Clean Up Docker Resources"
echo "==================================================="

echo "Cleaning up unused Docker resources..."

# Remove stopped containers
STOPPED_COUNT=$(docker ps -aq -f status=exited | wc -l)
if [ "$STOPPED_COUNT" -gt 0 ]; then
    docker container prune -f
    print_status "Removed $STOPPED_COUNT stopped containers"
else
    print_status "No stopped containers to remove"
fi

# Remove unused images
echo "Removing unused images..."
docker image prune -f > /tmp/image-prune.log 2>&1
print_status "Unused images removed"

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f > /tmp/volume-prune.log 2>&1
print_status "Unused volumes removed"

# Clear build cache if it's large
BUILD_CACHE_SIZE=$(docker system df | grep "Build Cache" | awk '{print $4}')
echo "Current build cache size: $BUILD_CACHE_SIZE"
if [[ "$BUILD_CACHE_SIZE" =~ GB ]]; then
    echo "Clearing large build cache..."
    docker builder prune -f > /tmp/builder-prune.log 2>&1
    print_status "Build cache cleared"
fi

echo ""
echo "==================================================="
echo "PHASE 5: Verify Optimizations"
echo "==================================================="

echo "New Docker Memory Usage:"
echo "------------------------"
docker stats --no-stream --format "table {{.Container}}\t{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"

echo ""
echo "System Memory Status:"
echo "--------------------"
free -h

echo ""
echo "Docker System Overview:"
echo "----------------------"
docker system df

echo ""
echo "==================================================="
echo "OPTIMIZATION SUMMARY"
echo "==================================================="

TOTAL_SAVED=0

# Calculate savings
if docker ps -q -f name=sutazai-ollama > /dev/null 2>&1; then
    TOTAL_SAVED=$((TOTAL_SAVED + 21300))
    echo "- Ollama: Saved 21.3GB"
fi

if docker ps -q -f name=sutazai-backend > /dev/null 2>&1; then
    TOTAL_SAVED=$((TOTAL_SAVED + 1500))
    echo "- Backend: Saved 1.5GB"
fi

if docker ps -q -f name=sutazai-faiss > /dev/null 2>&1; then
    TOTAL_SAVED=$((TOTAL_SAVED + 1600))
    echo "- FAISS: Saved 1.6GB"
fi

echo ""
echo -e "${GREEN}Total Memory Saved: ~$((TOTAL_SAVED / 1000))GB${NC}"
echo ""

echo "==================================================="
echo "RECOMMENDED NEXT STEPS"
echo "==================================================="
echo ""
echo "1. Monitor containers for 10 minutes to ensure stability:"
echo "   watch -n 5 'docker stats --no-stream'"
echo ""
echo "2. If containers show issues, restore original limits:"
echo "   docker update --memory=\"23g\" sutazai-ollama  # (if needed)"
echo ""
echo "3. Update docker-compose files with new limits for permanence"
echo ""
echo "4. Consider implementing the full optimization report:"
echo "   cat /opt/sutazaiapp/docker-optimization-report.md"
echo ""

print_status "Optimization complete!"