#!/bin/bash
# Emergency Memory Fix Script for SutazAI
# Immediately stops OOM kills and optimizes memory usage

set -e

echo "🚨 SutazAI Emergency Memory Fix Script"
echo "====================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Stop all running containers immediately
echo -e "${YELLOW}Step 1: Stopping all containers to prevent OOM kills...${NC}"
docker-compose -f docker-compose.yml down --remove-orphans 2>/dev/null || true
docker-compose -f docker-compose-optimized.yml down --remove-orphans 2>/dev/null || true
docker stop $(docker ps -q) 2>/dev/null || true

# Clean up Docker resources
echo -e "${YELLOW}Step 2: Cleaning up Docker resources...${NC}"
docker container prune -f
docker image prune -a -f
docker volume prune -f
docker network prune -f

# Add swap space immediately
echo -e "${YELLOW}Step 3: Adding swap space (8GB)...${NC}"
if [ ! -f /swapfile ]; then
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo -e "${GREEN}✓ 8GB swap file created${NC}"
else
    sudo swapon /swapfile 2>/dev/null || true
    echo -e "${GREEN}✓ Swap file activated${NC}"
fi

# Optimize system settings
echo -e "${YELLOW}Step 4: Optimizing system memory settings...${NC}"
echo 1 | sudo tee /proc/sys/vm/overcommit_memory
echo 10 | sudo tee /proc/sys/vm/swappiness
echo -e "${GREEN}✓ System settings optimized${NC}"

# Create emergency deployment
echo -e "${YELLOW}Step 5: Creating emergency minimal deployment...${NC}"

cat > docker-compose-emergency.yml << 'EOF'
version: '3.8'

services:
  postgresql:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: sutazai
      POSTGRES_USER: sutazai
      POSTGRES_PASSWORD: sutazai2024
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
    command: >
      postgres
      -c shared_buffers=128MB
      -c work_mem=2MB
      -c maintenance_work_mem=32MB
      -c effective_cache_size=256MB
      -c max_connections=50

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M
    command: >
      redis-server
      --maxmemory 128mb
      --maxmemory-policy allkeys-lru

  ollama:
    image: ollama/ollama:latest
    ports:
      - "10104:10104"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      OLLAMA_HOST: 0.0.0.0
      OLLAMA_NUM_PARALLEL: 1
      OLLAMA_MAX_LOADED_MODELS: 1
      OLLAMA_KEEP_ALIVE: 2m
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  streamlit-frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      BACKEND_URL: http://localhost:8000
      STREAMLIT_SERVER_MAX_MESSAGE_SIZE: 50
      STREAMLIT_SERVER_MAX_UPLOAD_SIZE: 50
    volumes:
      - ./intelligent_chat_app_fixed.py:/app/app.py:ro
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
    depends_on:
      - ollama

volumes:
  postgres_data:
  redis_data:
  ollama_data:
EOF

# Start minimal services
echo -e "${YELLOW}Step 6: Starting minimal services...${NC}"
docker-compose -f docker-compose-emergency.yml up -d --build

# Wait and verify
echo -e "${YELLOW}Step 7: Waiting for services to start...${NC}"
sleep 30

# Load minimal model
echo -e "${YELLOW}Step 8: Loading minimal model...${NC}"
docker-compose -f docker-compose-emergency.yml exec ollama ollama pull tinyllama2.5:3b || true

echo -e "${GREEN}
✅ Emergency fix completed!

Current status:
- All containers stopped and cleaned up
- 8GB swap space added
- Memory settings optimized
- Minimal services started with strict memory limits

Access the application at: http://localhost:8501

To monitor memory usage:
  free -h
  docker stats

To deploy the full optimized system later:
  ./deploy-optimized.sh

${NC}"

# Show current memory status
echo -e "${YELLOW}Current memory status:${NC}"
free -h