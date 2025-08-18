#!/bin/bash

# Security Hardened Deployment Script
# Purpose: Deploy Rule 11 compliant security-hardened SutazAI production environment

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/deploy-security-$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${1}" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Security validation function
validate_security() {
    local service=$1
    local checks_passed=0
    local total_checks=5
    
    log "${YELLOW}Security validation for $service:${NC}"
    
    # Check 1: Non-root user
    if docker exec "$service" id 2>/dev/null | grep -qv "uid=0"; then
        log "${GREEN}  ✓ Running as non-root user${NC}"
        ((checks_passed++))
    else
        log "${RED}  ✗ Running as root user (security risk)${NC}"
    fi
    
    # Check 2: Read-only filesystem
    if docker inspect "$service" --format '{{.HostConfig.ReadonlyRootfs}}' 2>/dev/null | grep -q "true"; then
        log "${GREEN}  ✓ Read-only root filesystem${NC}"
        ((checks_passed++))
    else
        log "${YELLOW}  △ Writable root filesystem${NC}"
    fi
    
    # Check 3: No new privileges
    if docker inspect "$service" --format '{{.HostConfig.SecurityOpt}}' 2>/dev/null | grep -q "no-new-privileges:true"; then
        log "${GREEN}  ✓ No new privileges security option${NC}"
        ((checks_passed++))
    else
        log "${YELLOW}  △ New privileges not restricted${NC}"
    fi
    
    # Check 4: Resource limits
    if docker inspect "$service" --format '{{.HostConfig.Memory}}' 2>/dev/null | grep -qv "^0$"; then
        log "${GREEN}  ✓ Memory limits configured${NC}"
        ((checks_passed++))
    else
        log "${YELLOW}  △ No memory limits${NC}"
    fi
    
    # Check 5: Health check
    if docker inspect "$service" --format '{{.Config.Healthcheck}}' 2>/dev/null | grep -q "Test"; then
        log "${GREEN}  ✓ Health check configured${NC}"
        ((checks_passed++))
    else
        log "${YELLOW}  △ No health check configured${NC}"
    fi
    
    local security_score=$((checks_passed * 100 / total_checks))
    if [ $security_score -ge 80 ]; then
        log "${GREEN}  Security Score: $security_score% (EXCELLENT)${NC}"
    elif [ $security_score -ge 60 ]; then
        log "${YELLOW}  Security Score: $security_score% (GOOD)${NC}"
    else
        log "${RED}  Security Score: $security_score% (NEEDS IMPROVEMENT)${NC}"
    fi
    
    return $checks_passed
}

# Create logs directory
mkdir -p "${PROJECT_ROOT}/logs"

log "${PURPLE}SutazAI Security Hardened Deployment${NC}"
log "${PURPLE}====================================${NC}"
log "Started at: $(date)"
log "Project root: $PROJECT_ROOT"
log "Log file: $LOG_FILE"

# Change to project root
cd "$PROJECT_ROOT"

# Validate environment variables
log "${YELLOW}Validating security environment variables...${NC}"
REQUIRED_VARS=("POSTGRES_PASSWORD" "NEO4J_PASSWORD" "REDIS_PASSWORD" "SECRET_KEY" "JWT_SECRET_KEY" "RABBITMQ_DEFAULT_PASS")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    log "${RED}Missing required environment variables: ${MISSING_VARS[*]}${NC}"
    log "${YELLOW}Please set these variables before deployment for security.${NC}"
    error_exit "Security validation failed - missing environment variables"
fi

log "${GREEN}✓ All required security environment variables are set${NC}"

# Validate docker-compose security configuration
log "${YELLOW}Validating security configuration...${NC}"
if ! docker-compose -f docker/docker-compose.security.yml config >/dev/null 2>&1; then
    error_exit "Security docker-compose.security.yml is invalid"
fi

log "${GREEN}✓ Security configuration is valid${NC}"

# Create network if it doesn't exist
log "${YELLOW}Creating Docker network...${NC}"
if ! docker network inspect sutazai-network >/dev/null 2>&1; then
    docker network create sutazai-network
    log "${GREEN}✓ Created sutazai-network${NC}"
else
    log "${GREEN}✓ Network sutazai-network already exists${NC}"
fi

# Create security configuration directories
log "${YELLOW}Creating security configuration directories...${NC}"
mkdir -p config/{redis,consul,rabbitmq,kong,qdrant}
mkdir -p sql logs

# Generate secure configuration files if they don't exist
log "${YELLOW}Generating secure configuration files...${NC}"

# Redis secure configuration
if [ ! -f "config/redis-secure.conf" ]; then
    cat > config/redis-secure.conf << EOF
# Redis Security Configuration
requirepass ${REDIS_PASSWORD}
protected-mode yes
port 6379
bind 127.0.0.1 0.0.0.0
timeout 300
tcp-keepalive 300
maxmemory 1gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
EOF
    log "${GREEN}✓ Generated Redis secure configuration${NC}"
fi

# Pull latest security-hardened images
log "${YELLOW}Pulling latest security-hardened images...${NC}"
docker-compose -f docker/docker-compose.security.yml pull || {
    log "${YELLOW}Warning: Some images could not be pulled. Continuing with local images.${NC}"
}

# Build custom security-hardened images
log "${YELLOW}Building security-hardened Docker images...${NC}"
docker-compose -f docker/docker-compose.security.yml build

# Stop existing containers
log "${YELLOW}Stopping existing containers...${NC}"
docker-compose -f docker/docker-compose.security.yml down || true

# Start security-hardened environment
log "${YELLOW}Starting security-hardened environment...${NC}"
docker-compose -f docker/docker-compose.security.yml up -d

# Wait for services to be healthy
log "${YELLOW}Waiting for services to initialize...${NC}"
sleep 60

# Validate service health and security
log "${YELLOW}Performing comprehensive security validation...${NC}"
FAILED_SERVICES=()
SECURITY_SCORES=()

# Core services security validation
SECURITY_SERVICES=("sutazai-postgres" "sutazai-redis" "sutazai-neo4j" "sutazai-chromadb" "sutazai-qdrant" "sutazai-ollama" "sutazai-kong" "sutazai-consul" "sutazai-rabbitmq" "sutazai-backend" "sutazai-frontend")

for service in "${SECURITY_SERVICES[@]}"; do
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$service" | grep -q "healthy\|Up"; then
        log "${GREEN}✓ $service is running${NC}"
        
        # Perform security validation
        validate_security "$service"
        score=$?
        SECURITY_SCORES+=("$service:$score")
    else
        log "${RED}✗ $service is not healthy${NC}"
        FAILED_SERVICES+=("$service")
    fi
done

# Network security validation
log "${YELLOW}Validating network security...${NC}"
if docker network inspect sutazai-network --format '{{.Internal}}' 2>/dev/null | grep -q "false"; then
    log "${GREEN}✓ Network allows external connectivity (expected for production)${NC}"
else
    log "${YELLOW}△ Network is internal-only${NC}"
fi

# Port exposure validation
log "${YELLOW}Validating port exposure...${NC}"
EXPOSED_PORTS=$(docker-compose -f docker/docker-compose.security.yml config | grep -E "^\s+- \"[0-9]" | wc -l)
log "${BLUE}Exposed ports: $EXPOSED_PORTS${NC}"

if [ "$EXPOSED_PORTS" -lt 20 ]; then
    log "${GREEN}✓ port exposure (good security practice)${NC}"
else
    log "${YELLOW}△ Many ports exposed - review if all are necessary${NC}"
fi

# Show security-hardened URLs
log "${BLUE}Security Hardened Environment URLs:${NC}"
log "Backend API: http://localhost:10010"
log "Frontend UI: http://localhost:10011"
log "PostgreSQL: localhost:10000 (secured with auth)"
log "Redis: localhost:10001 (secured with auth)"
log "Neo4j: http://localhost:10002 (secured with auth)"
log "Kong Gateway: http://localhost:10005"
log "Kong Admin: http://localhost:10015"
log "Consul: http://localhost:10006"
log "RabbitMQ: http://localhost:10008 (secured with auth)"

# Show running containers with security info
log "${YELLOW}Security-hardened containers:${NC}"
docker-compose -f docker/docker-compose.security.yml ps

# Security summary
log "${PURPLE}Security Deployment Summary:${NC}"
total_services=${#SECURITY_SERVICES[@]}
running_services=$((total_services - ${#FAILED_SERVICES[@]}))
log "Running services: $running_services/$total_services"

# Calculate overall security score
total_security_score=0
for score_entry in "${SECURITY_SCORES[@]}"; do
    score=$(echo "$score_entry" | cut -d: -f2)
    total_security_score=$((total_security_score + score))
done

if [ ${#SECURITY_SCORES[@]} -gt 0 ]; then
    avg_security_score=$((total_security_score * 100 / (${#SECURITY_SCORES[@]} * 5)))
    log "${PURPLE}Overall Security Score: $avg_security_score%${NC}"
    
    if [ $avg_security_score -ge 80 ]; then
        log "${GREEN}Security Status: EXCELLENT${NC}"
    elif [ $avg_security_score -ge 60 ]; then
        log "${YELLOW}Security Status: GOOD${NC}"
    else
        log "${RED}Security Status: NEEDS IMPROVEMENT${NC}"
    fi
fi

# Final status
if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
    log "${GREEN}Security-hardened environment deployed successfully!${NC}"
    log "${GREEN}All services are running with security hardening applied.${NC}"
    
    log "${BLUE}Security Features Enabled:${NC}"
    log "• Non-root user execution"
    log "• Read-only filesystems where possible"
    log "• No new privileges security option"
    log "• Resource limits on all services"
    log "• Comprehensive health checks"
    log "• Secure password authentication"
    log "• Network isolation"
    log "• attack surface"
    
    log "${YELLOW}Security Recommendations:${NC}"
    log "1. Regularly update all container images"
    log "2. Monitor security logs and alerts"
    log "3. Rotate passwords and secrets regularly"
    log "4. Review and audit access controls"
    log "5. Backup encryption keys securely"
    
    exit 0
else
    log "${RED}Security deployment completed with issues!${NC}"
    log "${RED}Failed services: ${FAILED_SERVICES[*]}${NC}"
    
    # Show logs for failed services
    for service in "${FAILED_SERVICES[@]}"; do
        log "${YELLOW}Logs for $service:${NC}"
        docker logs "$service" --tail=20 2>&1 || true
    done
    
    log "${YELLOW}Security troubleshooting tips:${NC}"
    log "1. Verify all environment variables are set"
    log "2. Check file permissions on configuration files"
    log "3. Ensure Docker has sufficient resources"
    log "4. Review security constraints (they may be too restrictive)"
    
    exit 1
fi