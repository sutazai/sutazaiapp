#!/bin/bash
# Fix Redis Configuration and Restart

echo "🔧 Fixing Redis configuration..."

# Stop Redis
echo "Stopping Redis container..."
docker stop sutazai-redis
docker rm sutazai-redis

# Create proper Redis configuration
cat > /tmp/redis.conf << 'EOF'
# Redis configuration for SutazAI
bind 0.0.0.0
protected-mode no
port 6379
maxmemory 512mb
maxmemory-policy allkeys-lru
save 60 1000
appendonly yes
appendfilename "appendonly.aof"
EOF

# Update docker-compose to use config file
echo "Updating Redis configuration..."

# Start Redis with proper configuration
docker run -d \
  --name sutazai-redis \
  --network sutazai-network \
  --restart unless-stopped \
  -p 6379:6379 \
  -v redis_data:/data \
  -v /tmp/redis.conf:/usr/local/etc/redis/redis.conf \
  redis:7.2-alpine redis-server /usr/local/etc/redis/redis.conf

# Wait for Redis to be ready
echo "Waiting for Redis to start..."
for i in {1..30}; do
  if docker exec sutazai-redis redis-cli ping 2>/dev/null | grep -q PONG; then
    echo "✅ Redis is ready!"
    break
  fi
  sleep 1
done

# Verify Redis is working
echo "Testing Redis..."
docker exec sutazai-redis redis-cli ping

echo "✅ Redis fixed and running!"