#!/bin/bash
#
# Network Security Assessment Script for SutazAI
# Analyzes network exposure and provides security recommendations
# Author: Claude Security Specialist
# Date: August 4, 2025
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

echo "==========================================="
echo "    SutazAI Network Security Assessment"
echo "==========================================="
echo

log "Analyzing network exposure..."

# Check exposed ports
EXPOSED_SERVICES=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep "0.0.0.0" | wc -l)
log "Total services with external exposure: $EXPOSED_SERVICES"

# Check for database services on external interfaces
log "Checking for exposed database services..."
DB_EXPOSED=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep "0.0.0.0" | grep -E "(3306|5432|6379|27017|9200|5984)" || true)

if [[ -n "$DB_EXPOSED" ]]; then
    warn "Database services exposed on external interfaces:"
    echo "$DB_EXPOSED"
else
    success "No critical database services directly exposed"
fi

# Check for administrative interfaces
log "Checking for exposed administrative interfaces..."
ADMIN_EXPOSED=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep "0.0.0.0" | grep -E "(8080|8081|9090|3000|4000)" || true)

if [[ -n "$ADMIN_EXPOSED" ]]; then
    warn "Administrative interfaces exposed:"
    echo "$ADMIN_EXPOSED"
fi

# Generate firewall rules recommendation
log "Generating firewall configuration recommendations..."

cat > /opt/sutazaiapp/firewall-rules.txt << 'EOF'
# SutazAI Recommended Firewall Rules
# Apply these rules to secure network access

# Allow SSH (adjust port as needed)
-A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
-A INPUT -p tcp --dport 80 -j ACCEPT
-A INPUT -p tcp --dport 443 -j ACCEPT

# Allow only specific SutazAI services (examples)
# Main dashboard/API
-A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
-A INPUT -p tcp --dport 8000 -s 172.16.0.0/12 -j ACCEPT
-A INPUT -p tcp --dport 8000 -s 192.168.0.0/16 -j ACCEPT

# Monitoring (Prometheus)
-A INPUT -p tcp --dport 10200 -s 10.0.0.0/8 -j ACCEPT
-A INPUT -p tcp --dport 10200 -s 172.16.0.0/12 -j ACCEPT
-A INPUT -p tcp --dport 10200 -s 192.168.0.0/16 -j ACCEPT

# Block all other external access to SutazAI ports
-A INPUT -p tcp --dport 10000:12000 -j DROP

# Allow loopback
-A INPUT -i lo -j ACCEPT

# Allow established connections
-A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Default drop
-A INPUT -j DROP
EOF

success "Firewall rules generated: /opt/sutazaiapp/firewall-rules.txt"

# Check for services that should be internal-only
log "Identifying services that should be internal-only..."

INTERNAL_SERVICES="redis postgres neo4j mongodb elasticsearch rabbitmq"
for service in $INTERNAL_SERVICES; do
    if docker ps --format "{{.Names}}" | grep -i "$service" | head -1 | xargs -I {} docker port {} 2>/dev/null | grep "0.0.0.0" >/dev/null 2>&1; then
        warn "Service '$service' is exposed on external interface"
    fi
done

# Generate secure docker-compose network configuration
log "Generating secure network configuration..."

cat > /opt/sutazaiapp/docker-compose.network-secure.yml << 'EOF'
# Network Security Configuration for SutazAI
# Use this as a reference to secure your docker-compose networks

networks:
  frontend:
    driver: bridge
    internal: false  # Allow external access for web services
    ipam:
      config:
        - subnet: 172.20.0.0/24
  
  backend:
    driver: bridge
    internal: true   # Internal-only network for databases
    ipam:
      config:
        - subnet: 172.21.0.0/24
  
  monitoring:
    driver: bridge
    internal: true   # Internal monitoring network
    ipam:
      config:
        - subnet: 172.22.0.0/24

# Example service network assignments
services:
  # Public-facing services
  nginx:
    networks:
      - frontend
  
  backend:
    networks:
      - frontend
      - backend
  
  # Internal services - no external access
  postgres:
    networks:
      - backend
  
  redis:
    networks:
      - backend
  
  neo4j:
    networks:
      - backend
  
  # Monitoring services
  prometheus:
    networks:
      - monitoring
      - backend  # To collect metrics
EOF

success "Secure network configuration generated: /opt/sutazaiapp/docker-compose.network-secure.yml"

echo
echo "==========================================="
echo "    Network Security Assessment Complete"
echo "==========================================="
echo

warn "CRITICAL RECOMMENDATIONS:"
echo "1. Apply firewall rules from: /opt/sutazaiapp/firewall-rules.txt"
echo "2. Implement network segmentation using: /opt/sutazaiapp/docker-compose.network-secure.yml"
echo "3. Move database services to internal-only networks"
echo "4. Configure reverse proxy for external access"
echo "5. Implement VPN access for administrative interfaces"
echo "6. Regular port scanning and security audits"

echo
success "Network security assessment completed successfully!"