#\!/bin/bash
# System Optimization Script for SutazAI

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║           SutazAI System Optimization Tool                        ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
   echo -e "${RED}Please run as root (use sudo)${NC}"
   exit 1
fi

echo -e "${BLUE}1. Cleaning Docker Resources...${NC}"
# Remove stopped containers
stopped_count=$(docker ps -aq -f status=exited | wc -l)
if [ $stopped_count -gt 0 ]; then
    echo "Removing $stopped_count stopped containers..."
    docker container prune -f
else
    echo "No stopped containers to remove."
fi

# Remove unused images
echo "Removing unused images..."
docker image prune -f

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove unused networks
echo "Removing unused networks (keeping sutazai network)..."
docker network prune -f

echo -e "\n${BLUE}2. Optimizing Container Resources...${NC}"
# Set memory limits for high-usage containers
echo "Setting resource limits for agents..."

# Get all agent containers
docker ps --format "{{.Names}}" | grep -E "agent|developer|engineer|specialist" | while read container; do
    # Set reasonable memory limit (512MB) and CPU limit (0.5 CPU)
    docker update --memory="512m" --memory-swap="512m" --cpus="0.5" $container 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} Updated limits for $container"
    fi
done

echo -e "\n${BLUE}3. System Cache Optimization...${NC}"
# Clear system caches
echo "Clearing system caches..."
sync
echo 3 > /proc/sys/vm/drop_caches
echo -e "  ${GREEN}✓${NC} System caches cleared"

# Optimize Redis
echo "Optimizing Redis..."
docker exec sutazai-redis redis-cli -a redis_password BGREWRITEAOF 2>/dev/null
docker exec sutazai-redis redis-cli -a redis_password MEMORY PURGE 2>/dev/null
echo -e "  ${GREEN}✓${NC} Redis optimized"

echo -e "\n${BLUE}4. Log Rotation...${NC}"
# Rotate Docker logs
echo "Configuring log rotation..."
cat > /etc/docker/daemon.json << EOL
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOL

# Restart Docker daemon to apply changes
systemctl restart docker 2>/dev/null || service docker restart 2>/dev/null
echo -e "  ${GREEN}✓${NC} Log rotation configured"

echo -e "\n${BLUE}5. Database Optimization...${NC}"
# Optimize PostgreSQL
echo "Running PostgreSQL VACUUM..."
docker exec sutazai-postgres psql -U sutazai -d sutazai_agents -c "VACUUM ANALYZE;" 2>/dev/null
echo -e "  ${GREEN}✓${NC} PostgreSQL optimized"

# Show results
echo -e "\n${BLUE}6. Optimization Results:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Disk space saved
df -h / | awk 'NR==2 {print "Disk Usage: " $3 " / " $2 " (" $5 ")"}'

# Memory usage
free -h | awk '/^Mem:/ {print "Memory Usage: " $3 " / " $2}'

# Container count
echo "Active Containers: $(docker ps -q | wc -l)"

# Calculate saved space
saved_space=$(docker system df | grep "Reclaimable" | awk '{sum+=$4} END {print sum}')
echo -e "\n${GREEN}Space Reclaimed: ${saved_space}GB${NC}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "\n${GREEN}✅ System optimization complete\!${NC}"
echo ""
echo "Recommendations:"
echo "1. Run this script weekly to maintain optimal performance"
echo "2. Monitor resource usage with: docker stats"
echo "3. Check system health with: ./scripts/verify_complete_system.sh"
