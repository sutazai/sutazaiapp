#!/bin/bash

echo "=== SutazAI System Optimization Script ==="
echo "Removing unnecessary services and optimizing for port 8501 only"

# Stop and remove unnecessary containers
echo "Stopping and removing non-essential containers..."
docker stop sutazai-open-webui || true
docker rm sutazai-open-webui || true
docker stop sutazai-enhanced-model-manager || true
docker rm sutazai-enhanced-model-manager || true

# Clear Docker logs to free up disk space
echo "Clearing Docker logs..."
truncate -s 0 /var/lib/docker/containers/*/*-json.log

# Clear system logs
echo "Clearing old system logs..."
journalctl --vacuum-time=1d
find /var/log -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true

# Optimize Ollama
echo "Optimizing Ollama resource usage..."
docker update sutazai-ollama \
  --cpus="1.0" \
  --memory="4g" \
  --memory-swap="4g" \
  --cpu-shares=512

# Kill excessive ollama runner processes
echo "Killing excessive ollama runner processes..."
pkill -f "ollama runner" || true

# Clean Docker system
echo "Cleaning Docker system..."
docker system prune -af --volumes

# Restart essential services with optimized settings
echo "Restarting Ollama with optimized settings..."
docker restart sutazai-ollama

# Clear cache
echo "Clearing system cache..."
sync && echo 3 > /proc/sys/vm/drop_caches

# Display new resource usage
echo "=== Current Resource Status ==="
free -h
df -h /
docker stats --no-stream

echo "=== Optimization Complete ==="
echo "Only http://192.168.131.128:8501/ and essential services are active"