#!/bin/bash
# Comprehensive Docker optimization script for SutazAI

set -e

echo "========================================="
echo "Docker Optimization for SutazAI"
echo "========================================="

# 1. Stop all containers
echo "Step 1: Stopping all containers..."
docker compose down --remove-orphans 2>/dev/null || true
docker stop $(docker ps -aq) 2>/dev/null || true

# 2. Clean up system
echo "Step 2: Cleaning Docker system..."
docker system prune -af --volumes
docker builder prune -af
docker network prune -f

# 3. Remove old images
echo "Step 3: Removing unused images..."
docker image prune -af

# 4. Clean volumes
echo "Step 4: Cleaning volumes..."
docker volume prune -f

# 5. Show disk usage
echo "Step 5: Current Docker disk usage:"
docker system df

# 6. Set Docker daemon optimizations
echo "Step 6: Applying Docker daemon optimizations..."
if [ ! -f /etc/docker/daemon.json.bak ]; then
    sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak 2>/dev/null || true
fi

# Create optimized daemon config
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "metrics-addr": "127.0.0.1:9323",
  "experimental": false,
  "features": {
    "buildkit": true
  },
  "builder": {
    "gc": {
      "enabled": true,
      "defaultKeepStorage": "10GB"
    }
  }
}
EOF

# 7. Restart Docker daemon
echo "Step 7: Restarting Docker daemon..."
sudo systemctl restart docker || sudo service docker restart

# 8. Wait for Docker to be ready
echo "Step 8: Waiting for Docker to be ready..."
sleep 5

# 9. Create network
echo "Step 9: Creating Docker network..."
docker network create sutazai-network 2>/dev/null || true

echo ""
echo "========================================="
echo "Optimization Complete!"
echo "========================================="
echo ""
echo "Docker disk usage after cleanup:"
docker system df
echo ""
echo "To start SutazAI:"
echo "  Minimal mode: ./scripts/start-minimal.sh"
echo "  With features: ./scripts/start-with-features.sh"
echo ""
echo "Tips for better performance:"
echo "  1. Use docker-compose.override.yml for resource limits"
echo "  2. Start only essential services"
echo "  3. Monitor with: docker stats"
echo "  4. Check logs with: docker compose logs -f [service]"