# Deployment Runbook - Perfect Jarvis System

**Document Version:** 1.0  
**Last Updated:** 2025-08-08  
**Author:** DevOps Team  

## 🎯 Purpose

This runbook provides comprehensive deployment procedures for the Perfect Jarvis system, covering development, staging, and production environments with proper validation and rollback strategies.

## 📋 Table of Contents

- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Environment Overview](#environment-overview)
- [Deployment Strategies](#deployment-strategies)
- [Step-by-Step Deployment](#step-by-step-deployment)
- [Blue/Green Deployment](#bluegreen-deployment)
- [Database Migration](#database-migration)
- [Configuration Management](#configuration-management)
- [Post-Deployment Validation](#post-deployment-validation)
- [Rollback Procedures](#rollback-procedures)
- [Troubleshooting](#troubleshooting)

## ✅ Pre-Deployment Checklist

### Code Quality Requirements
- [ ] All tests passing (`pytest` for backend, integration tests)
- [ ] Code review completed and approved
- [ ] Security scan completed (no critical vulnerabilities)
- [ ] Performance benchmarks within acceptable ranges
- [ ] Documentation updated for any API changes
- [ ] Feature flags configured appropriately

### Infrastructure Requirements
- [ ] Target environment is healthy (all services green)
- [ ] Sufficient resources available (CPU, memory, disk)
- [ ] Database backup completed and verified
- [ ] Configuration files validated
- [ ] SSL certificates are valid and not expiring
- [ ] Monitoring and alerting systems operational

### Dependency Verification
- [ ] All required Docker images built and pushed
- [ ] Database schema migrations tested
- [ ] Third-party service dependencies verified
- [ ] Network connectivity validated
- [ ] External API integrations tested

### Team Coordination
- [ ] Deployment window scheduled and communicated
- [ ] On-call engineer identified and available
- [ ] Rollback plan reviewed with team
- [ ] Stakeholders notified of deployment schedule
- [ ] Emergency contact list updated

### Deployment Artifacts
- [ ] Docker images tagged with version
- [ ] Configuration files prepared for target environment
- [ ] Database migration scripts validated
- [ ] Rollback scripts prepared and tested
- [ ] Environment-specific secrets validated

## 🌍 Environment Overview

### Development Environment
- **Purpose:** Feature development and initial testing
- **URL:** `http://dev-jarvis.internal:10010`
- **Database:** PostgreSQL (dev instance)
- **Monitoring:** Basic health checks
- **Auto-deployment:** On merge to `develop` branch

### Staging Environment
- **Purpose:** Integration testing and UAT
- **URL:** `https://staging-jarvis.company.com`
- **Database:** PostgreSQL (staging instance with prod-like data)
- **Monitoring:** Full monitoring stack
- **Deployment:** Manual trigger after dev validation

### Production Environment
- **Purpose:** Live system serving users
- **URL:** `https://jarvis.company.com`
- **Database:** PostgreSQL (production cluster)
- **Monitoring:** Full monitoring with alerting
- **Deployment:** Manual with approval process

### Environment Configuration Matrix

| Component | Development | Staging | Production |
|-----------|-------------|---------|------------|
| Backend Replicas | 1 | 2 | 3 |
| Database | Single instance | Single with backup | HA Cluster |
| Redis | Single instance | Single with backup | Cluster mode |
| Ollama | CPU-only | CPU-optimized | CPU-optimized |
| Monitoring | Basic | Full stack | Full + alerting |
| SSL | Self-signed | Valid cert | Valid cert |
| Backups | None | Daily | Hourly |

## 🚀 Deployment Strategies

### 1. Rolling Deployment (Default)
**Use Case:** Regular updates with zero downtime  
**Risk Level:** Medium  
**Rollback Speed:** Fast (2-3 minutes)

**Process:**
1. Stop one instance at a time
2. Update and start new version
3. Validate health before proceeding to next instance
4. Complete when all instances updated

### 2. Blue/Green Deployment (Recommended for Production)
**Use Case:** Major releases, database schema changes  
**Risk Level:** Low  
**Rollback Speed:** Instant (DNS/load balancer switch)

**Process:**
1. Deploy new version to "green" environment
2. Run full validation suite
3. Switch traffic from "blue" to "green"
4. Monitor for issues
5. Keep "blue" environment for instant rollback

### 3. Canary Deployment
**Use Case:** High-risk changes, new features  
**Risk Level:** Very Low  
**Rollback Speed:** Instant

**Process:**
1. Deploy to small percentage of instances (10%)
2. Monitor key metrics and user feedback
3. Gradually increase percentage (25%, 50%, 100%)
4. Full rollout only after validation at each stage

## 📊 Step-by-Step Deployment

### Phase 1: Pre-Deployment Setup (15-30 minutes)

#### 1.1 Environment Preparation
```bash
#!/bin/bash
# pre_deployment_setup.sh
set -e

ENVIRONMENT=$1
VERSION=$2
DEPLOYMENT_ID="deploy_$(date +%Y%m%d_%H%M%S)"

echo "=== PRE-DEPLOYMENT SETUP ==="
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Deployment ID: $DEPLOYMENT_ID"

# Create deployment log
DEPLOY_LOG="/opt/sutazaiapp/logs/${DEPLOYMENT_ID}.log"
exec > >(tee -a "$DEPLOY_LOG") 2>&1

# Verify environment health
echo "1. Checking environment health..."
curl -f -s "http://localhost:10010/health" | jq '.status' || {
    echo "❌ Environment unhealthy - aborting deployment"
    exit 1
}

# Check system resources
echo "2. Checking system resources..."
MEMORY_USAGE=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')
CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')

if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    echo "❌ Memory usage too high: ${MEMORY_USAGE}%"
    exit 1
fi

if (( $(echo "$DISK_USAGE > 85" | bc -l) )); then
    echo "❌ Disk usage too high: ${DISK_USAGE}%"
    exit 1
fi

echo "✅ System resources OK (Memory: ${MEMORY_USAGE}%, Disk: ${DISK_USAGE}%)"

# Backup current configuration
echo "3. Backing up current configuration..."
tar -czf "/opt/sutazaiapp/backups/config_backup_${DEPLOYMENT_ID}.tar.gz" \
    docker-compose.yml \
    .env \
    config/ \
    backend/app/ 2>/dev/null || true

echo "✅ Pre-deployment setup complete"
echo "Deployment log: $DEPLOY_LOG"
```

#### 1.2 Database Backup
```bash
#!/bin/bash
# backup_before_deployment.sh
DEPLOYMENT_ID=$1
BACKUP_DIR="/opt/sutazaiapp/backups"
BACKUP_FILE="$BACKUP_DIR/pre_deploy_${DEPLOYMENT_ID}.sql"

echo "Creating pre-deployment database backup..."

# Stop write operations to ensure consistency
docker exec sutazai-backend curl -X POST http://localhost:8000/admin/maintenance-mode || true

# Create PostgreSQL backup
docker exec sutazai-postgres pg_dump -U sutazai sutazai > "$BACKUP_FILE"

# Verify backup
if [ -s "$BACKUP_FILE" ]; then
    echo "✅ Database backup created: $BACKUP_FILE"
    gzip "$BACKUP_FILE"
    echo "✅ Backup compressed: ${BACKUP_FILE}.gz"
else
    echo "❌ Database backup failed!"
    exit 1
fi

# Create Redis backup
docker exec sutazai-redis redis-cli BGSAVE
sleep 5
docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/redis_pre_deploy_${DEPLOYMENT_ID}.rdb"

echo "✅ All backups completed"
```

### Phase 2: Application Deployment (10-20 minutes)

#### 2.1 Container Image Preparation
```bash
#!/bin/bash
# prepare_images.sh
VERSION=$1
REGISTRY="your-registry.com/jarvis"

echo "=== PREPARING CONTAINER IMAGES ==="

# Build backend image
echo "1. Building backend image..."
docker build -t "${REGISTRY}/backend:${VERSION}" \
    -f backend/Dockerfile \
    ./backend/

# Build frontend image
echo "2. Building frontend image..."
docker build -t "${REGISTRY}/frontend:${VERSION}" \
    -f frontend/Dockerfile \
    ./frontend/

# Tag images
docker tag "${REGISTRY}/backend:${VERSION}" "${REGISTRY}/backend:latest"
docker tag "${REGISTRY}/frontend:${VERSION}" "${REGISTRY}/frontend:latest"

# Push to registry (if using external registry)
if [[ "$REGISTRY" != *"localhost"* ]]; then
    echo "3. Pushing images to registry..."
    docker push "${REGISTRY}/backend:${VERSION}"
    docker push "${REGISTRY}/frontend:${VERSION}"
    docker push "${REGISTRY}/backend:latest"
    docker push "${REGISTRY}/frontend:latest"
fi

echo "✅ Container images prepared"
```

#### 2.2 Rolling Deployment
```bash
#!/bin/bash
# rolling_deployment.sh
VERSION=$1
ENVIRONMENT=${2:-production}

echo "=== ROLLING DEPLOYMENT ==="
echo "Version: $VERSION"
echo "Environment: $ENVIRONMENT"

# Update docker-compose with new image versions
sed -i.bak "s|image: .*backend:.*|image: your-registry.com/jarvis/backend:${VERSION}|g" docker-compose.yml
sed -i.bak "s|image: .*frontend:.*|image: your-registry.com/jarvis/frontend:${VERSION}|g" docker-compose.yml

# Rolling update function
rolling_update() {
    local service=$1
    local max_attempts=3
    
    echo "Updating service: $service"
    
    for attempt in $(seq 1 $max_attempts); do
        echo "Attempt $attempt/$max_attempts for $service"
        
        # Stop old instance
        docker-compose stop $service
        
        # Remove old container
        docker-compose rm -f $service
        
        # Start new instance
        docker-compose up -d $service
        
        # Wait for service to be ready
        echo "Waiting for $service to be ready..."
        timeout 300 bash -c "
            while ! docker-compose ps $service | grep -q 'Up'; do
                sleep 5
            done
        "
        
        # Verify health
        if [[ "$service" == "backend" ]]; then
            if timeout 60 bash -c 'until curl -f -s http://localhost:10010/health; do sleep 2; done'; then
                echo "✅ $service updated successfully"
                break
            else
                echo "❌ $service health check failed on attempt $attempt"
                if [ $attempt -eq $max_attempts ]; then
                    echo "❌ All attempts failed for $service"
                    return 1
                fi
            fi
        else
            echo "✅ $service updated successfully"
            break
        fi
    done
}

# Update services in order
services=("backend" "frontend")
for service in "${services[@]}"; do
    rolling_update "$service" || {
        echo "❌ Rolling update failed for $service"
        exit 1
    }
done

echo "✅ Rolling deployment completed successfully"
```

### Phase 3: Database Migration (5-15 minutes)

#### 3.1 Schema Migration
```bash
#!/bin/bash
# run_migrations.sh
ENVIRONMENT=$1

echo "=== DATABASE MIGRATION ==="

# Check if migrations are needed
MIGRATION_NEEDED=$(docker exec sutazai-backend python -c "
from app.db import check_migrations_needed
print('true' if check_migrations_needed() else 'false')
" 2>/dev/null || echo "unknown")

if [ "$MIGRATION_NEEDED" = "unknown" ]; then
    echo "⚠️ Cannot determine migration status - proceeding with caution"
elif [ "$MIGRATION_NEEDED" = "false" ]; then
    echo "✅ No database migrations needed"
    return 0
fi

echo "📊 Running database migrations..."

# Set maintenance mode if supported
curl -X POST http://localhost:10010/admin/maintenance-mode 2>/dev/null || true

# Run migrations
docker exec sutazai-backend python -c "
import sys
sys.path.append('/app')
from app.db import migrate_database
try:
    migrate_database()
    print('✅ Database migration completed')
except Exception as e:
    print(f'❌ Migration failed: {e}')
    sys.exit(1)
" || {
    echo "❌ Database migration failed!"
    echo "🔄 Rolling back deployment..."
    exit 1
}

# Verify migration
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
    SELECT schemaname, tablename 
    FROM pg_tables 
    WHERE schemaname = 'public' 
    ORDER BY tablename;
" | grep -q "public" || {
    echo "❌ Migration verification failed"
    exit 1
}

# Remove maintenance mode
curl -X DELETE http://localhost:10010/admin/maintenance-mode 2>/dev/null || true

echo "✅ Database migration completed successfully"
```

### Phase 4: Configuration Update (5 minutes)

#### 4.1 Environment Configuration
```bash
#!/bin/bash
# update_configuration.sh
ENVIRONMENT=$1
VERSION=$2

echo "=== CONFIGURATION UPDATE ==="

# Backup current configuration
cp .env .env.backup
cp docker-compose.yml docker-compose.yml.backup

# Update version in environment file
sed -i "s/JARVIS_VERSION=.*/JARVIS_VERSION=${VERSION}/" .env

# Update feature flags for new version
case $ENVIRONMENT in
    "production")
        sed -i "s/SUTAZAI_ENTERPRISE_FEATURES=.*/SUTAZAI_ENTERPRISE_FEATURES=true/" .env
        sed -i "s/SUTAZAI_DEBUG=.*/SUTAZAI_DEBUG=false/" .env
        ;;
    "staging")
        sed -i "s/SUTAZAI_ENTERPRISE_FEATURES=.*/SUTAZAI_ENTERPRISE_FEATURES=true/" .env
        sed -i "s/SUTAZAI_DEBUG=.*/SUTAZAI_DEBUG=true/" .env
        ;;
    "development")
        sed -i "s/SUTAZAI_DEBUG=.*/SUTAZAI_DEBUG=true/" .env
        ;;
esac

# Validate configuration
docker-compose config --quiet || {
    echo "❌ Configuration validation failed"
    echo "🔄 Restoring backup configuration"
    mv .env.backup .env
    mv docker-compose.yml.backup docker-compose.yml
    exit 1
}

echo "✅ Configuration updated successfully"
```

## 🔄 Blue/Green Deployment

### Blue/Green Setup
```bash
#!/bin/bash
# blue_green_deployment.sh
VERSION=$1
CURRENT_COLOR=${2:-blue}
NEW_COLOR=$([[ "$CURRENT_COLOR" == "blue" ]] && echo "green" || echo "blue")

echo "=== BLUE/GREEN DEPLOYMENT ==="
echo "Current environment: $CURRENT_COLOR"
echo "New environment: $NEW_COLOR"
echo "Version: $VERSION"

# Prepare new environment
prepare_environment() {
    local color=$1
    local version=$2
    
    echo "Preparing $color environment with version $version"
    
    # Create environment-specific compose file
    cat > "docker-compose.${color}.yml" << EOF
version: '3.8'
services:
  backend-${color}:
    build: ./backend
    ports:
      - "${color == 'blue' && echo '10010' || echo '10012'}:8000"
    environment:
      - ENVIRONMENT=${color}
      - VERSION=${version}
    networks:
      - sutazai-network-${color}
      
  frontend-${color}:
    build: ./frontend
    ports:
      - "${color == 'blue' && echo '10011' || echo '10013'}:8501"
    networks:
      - sutazai-network-${color}

networks:
  sutazai-network-${color}:
    external: false
EOF
    
    # Deploy to new environment
    docker-compose -f "docker-compose.${color}.yml" up -d
    
    # Wait for health check
    local health_port=$([[ "$color" == "blue" ]] && echo "10010" || echo "10012")
    timeout 300 bash -c "
        while ! curl -f -s http://localhost:${health_port}/health; do
            sleep 5
        done
    "
}

# Validation suite for new environment
validate_environment() {
    local color=$1
    local health_port=$([[ "$color" == "blue" ]] && echo "10010" || echo "10012")
    
    echo "Validating $color environment..."
    
    # Health check
    curl -f -s "http://localhost:${health_port}/health" | jq '.status' | grep -q "healthy" || return 1
    
    # API functionality test
    curl -f -s -X POST "http://localhost:${health_port}/simple-chat" \
        -H "Content-Type: application/json" \
        -d '{"message": "deployment test"}' | jq '.response' | grep -q "test" || return 1
    
    # Performance test
    local response_time=$(curl -w "%{time_total}" -s -o /dev/null "http://localhost:${health_port}/health")
    if (( $(echo "$response_time > 2.0" | bc -l) )); then
        echo "❌ Response time too slow: ${response_time}s"
        return 1
    fi
    
    echo "✅ $color environment validation passed"
    return 0
}

# Traffic switching function
switch_traffic() {
    local from_color=$1
    local to_color=$2
    
    echo "Switching traffic from $from_color to $to_color"
    
    # Update load balancer configuration (example with nginx)
    cat > /etc/nginx/sites-available/jarvis << EOF
upstream jarvis_backend {
    server localhost:$([[ "$to_color" == "blue" ]] && echo "10010" || echo "10012");
}

upstream jarvis_frontend {
    server localhost:$([[ "$to_color" == "blue" ]] && echo "10011" || echo "10013");
}

server {
    listen 80;
    server_name jarvis.company.com;
    
    location /api {
        proxy_pass http://jarvis_backend;
    }
    
    location / {
        proxy_pass http://jarvis_frontend;
    }
}
EOF
    
    # Reload nginx
    nginx -t && nginx -s reload
    
    echo "✅ Traffic switched to $to_color environment"
}

# Main deployment flow
main() {
    # Prepare new environment
    prepare_environment "$NEW_COLOR" "$VERSION" || {
        echo "❌ Failed to prepare $NEW_COLOR environment"
        exit 1
    }
    
    # Validate new environment
    validate_environment "$NEW_COLOR" || {
        echo "❌ Validation failed for $NEW_COLOR environment"
        echo "🧹 Cleaning up failed deployment"
        docker-compose -f "docker-compose.${NEW_COLOR}.yml" down
        exit 1
    }
    
    # Switch traffic
    switch_traffic "$CURRENT_COLOR" "$NEW_COLOR"
    
    # Monitor for 5 minutes
    echo "🔍 Monitoring new environment for 5 minutes..."
    for i in {1..30}; do
        if ! validate_environment "$NEW_COLOR"; then
            echo "❌ Post-deployment validation failed"
            echo "🔄 Rolling back to $CURRENT_COLOR"
            switch_traffic "$NEW_COLOR" "$CURRENT_COLOR"
            exit 1
        fi
        sleep 10
    done
    
    echo "✅ Blue/Green deployment completed successfully"
    echo "🧹 Cleaning up old $CURRENT_COLOR environment"
    docker-compose -f "docker-compose.${CURRENT_COLOR}.yml" down
    
    # Update current environment marker
    echo "$NEW_COLOR" > /opt/sutazaiapp/.current_environment
}

main
```

## 📊 Configuration Management

### Environment-Specific Configurations

#### Development (.env.development)
```bash
# Development Environment Configuration
ENVIRONMENT=development
JARVIS_VERSION=latest
SUTAZAI_DEBUG=true
SUTAZAI_ENTERPRISE_FEATURES=false
SUTAZAI_ENABLE_KNOWLEDGE_GRAPH=false
SUTAZAI_ENABLE_COGNITIVE=false

# Database
POSTGRES_HOST=sutazai-postgres
POSTGRES_PORT=5432
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai
POSTGRES_DB=sutazai

# Redis
REDIS_HOST=sutazai-redis
REDIS_PORT=6379

# Ollama
OLLAMA_HOST=sutazai-ollama
OLLAMA_PORT=11434
OLLAMA_KEEP_ALIVE=5m

# Monitoring (minimal)
PROMETHEUS_ENABLED=false
GRAFANA_ENABLED=false

# Performance
UVICORN_WORKERS=1
MAX_CONCURRENT_REQUESTS=10
```

#### Staging (.env.staging)
```bash
# Staging Environment Configuration
ENVIRONMENT=staging
JARVIS_VERSION=v17.0.0
SUTAZAI_DEBUG=true
SUTAZAI_ENTERPRISE_FEATURES=true
SUTAZAI_ENABLE_KNOWLEDGE_GRAPH=true
SUTAZAI_ENABLE_COGNITIVE=false

# Database (production-like)
POSTGRES_HOST=staging-postgres.internal
POSTGRES_PORT=5432
POSTGRES_USER=sutazai_staging
POSTGRES_PASSWORD=${POSTGRES_STAGING_PASSWORD}
POSTGRES_DB=sutazai_staging

# Redis
REDIS_HOST=staging-redis.internal
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_STAGING_PASSWORD}

# Ollama
OLLAMA_HOST=staging-ollama.internal
OLLAMA_PORT=11434
OLLAMA_KEEP_ALIVE=10m

# Monitoring (full stack)
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
ALERT_WEBHOOK_URL=${STAGING_ALERT_WEBHOOK}

# Performance
UVICORN_WORKERS=2
MAX_CONCURRENT_REQUESTS=50
RATE_LIMIT_ENABLED=true
```

#### Production (.env.production)
```bash
# Production Environment Configuration
ENVIRONMENT=production
JARVIS_VERSION=v17.0.0
SUTAZAI_DEBUG=false
SUTAZAI_ENTERPRISE_FEATURES=true
SUTAZAI_ENABLE_KNOWLEDGE_GRAPH=true
SUTAZAI_ENABLE_COGNITIVE=true

# Database (production cluster)
POSTGRES_HOST=prod-postgres-cluster.internal
POSTGRES_PORT=5432
POSTGRES_USER=sutazai_prod
POSTGRES_PASSWORD=${POSTGRES_PROD_PASSWORD}
POSTGRES_DB=sutazai_prod
POSTGRES_SSL_MODE=require

# Redis (cluster mode)
REDIS_HOST=prod-redis-cluster.internal
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PROD_PASSWORD}
REDIS_CLUSTER_ENABLED=true

# Ollama (optimized)
OLLAMA_HOST=prod-ollama-cluster.internal
OLLAMA_PORT=11434
OLLAMA_KEEP_ALIVE=30m
OLLAMA_NUM_PARALLEL=2

# Security
SSL_CERTIFICATE_PATH=/opt/certs/jarvis.crt
SSL_PRIVATE_KEY_PATH=/opt/certs/jarvis.key
FORCE_HTTPS=true

# Monitoring (full with alerting)
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
LOKI_ENABLED=true
ALERT_WEBHOOK_URL=${PROD_ALERT_WEBHOOK}
PAGERDUTY_SERVICE_KEY=${PAGERDUTY_SERVICE_KEY}

# Performance (optimized)
UVICORN_WORKERS=4
MAX_CONCURRENT_REQUESTS=200
RATE_LIMIT_ENABLED=true
CACHE_ENABLED=true
CACHE_TTL=300

# Backup
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 */6 * * *"  # Every 6 hours
BACKUP_RETENTION_DAYS=30
```

### Secret Management

#### Development Secrets (dev-secrets.env)
```bash
# Development secrets (stored locally)
POSTGRES_STAGING_PASSWORD=dev_password_123
REDIS_STAGING_PASSWORD=dev_redis_123
STAGING_ALERT_WEBHOOK=https://hooks.slack.com/dev/webhook
```

#### Production Secrets (managed via secret management system)
```bash
#!/bin/bash
# load_production_secrets.sh
# This script loads production secrets from a secure secret management system

# Example: AWS Secrets Manager
export POSTGRES_PROD_PASSWORD=$(aws secretsmanager get-secret-value --secret-id prod/postgres/password --query SecretString --output text)
export REDIS_PROD_PASSWORD=$(aws secretsmanager get-secret-value --secret-id prod/redis/password --query SecretString --output text)
export PROD_ALERT_WEBHOOK=$(aws secretsmanager get-secret-value --secret-id prod/monitoring/webhook --query SecretString --output text)
export PAGERDUTY_SERVICE_KEY=$(aws secretsmanager get-secret-value --secret-id prod/pagerduty/key --query SecretString --output text)

# Example: HashiCorp Vault
# export POSTGRES_PROD_PASSWORD=$(vault kv get -field=password secret/prod/postgres)
# export REDIS_PROD_PASSWORD=$(vault kv get -field=password secret/prod/redis)

# Example: Kubernetes Secrets
# export POSTGRES_PROD_PASSWORD=$(kubectl get secret postgres-secret -o jsonpath='{.data.password}' | base64 -d)

echo "✅ Production secrets loaded"
```

## ✅ Post-Deployment Validation

### Comprehensive Validation Suite
```bash
#!/bin/bash
# post_deployment_validation.sh
ENVIRONMENT=$1
VERSION=$2
VALIDATION_LOG="/opt/sutazaiapp/logs/validation_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$VALIDATION_LOG") 2>&1

echo "=== POST-DEPLOYMENT VALIDATION ==="
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Timestamp: $(date -Iseconds)"

# Test Results
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# Test execution function
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -n "Testing $test_name... "
    
    if eval "$test_command" &>/dev/null; then
        echo "✅ PASS"
        ((TESTS_PASSED++))
    else
        echo "❌ FAIL"
        ((TESTS_FAILED++))
        FAILED_TESTS+=("$test_name")
    fi
}

# Health Check Tests
echo "1. HEALTH CHECK TESTS"
run_test "Backend Health" "curl -f -s http://localhost:10010/health | jq -e '.status == \"healthy\"'"
run_test "Frontend Accessibility" "curl -f -s http://localhost:10011 | grep -q 'Streamlit'"
run_test "Database Connectivity" "docker exec sutazai-postgres pg_isready -U sutazai"
run_test "Redis Connectivity" "docker exec sutazai-redis redis-cli ping | grep -q PONG"
run_test "Ollama Service" "curl -f -s http://localhost:10104/api/tags | jq -e '.models | length > 0'"

# Functional Tests
echo "2. FUNCTIONAL TESTS"
run_test "Simple Chat" "curl -f -s -X POST http://localhost:10010/simple-chat -H 'Content-Type: application/json' -d '{\"message\": \"test\"}' | jq -e '.response'"
run_test "Public Thinking" "curl -f -s -X POST http://localhost:10010/public/think -H 'Content-Type: application/json' -d '{\"query\": \"test\", \"reasoning_type\": \"general\"}' | jq -e '.response'"
run_test "Health Metrics" "curl -f -s http://localhost:10010/public/metrics | jq -e '.system.cpu_percent'"
run_test "Prometheus Metrics" "curl -f -s http://localhost:10010/prometheus-metrics | grep -q sutazai_uptime_seconds"

# Performance Tests
echo "3. PERFORMANCE TESTS"
RESPONSE_TIME=$(curl -w "%{time_total}" -s -o /dev/null http://localhost:10010/health)
run_test "Response Time (<2s)" "[[ \$(echo \"$RESPONSE_TIME < 2.0\" | bc -l) == 1 ]]"

MEMORY_USAGE=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')
run_test "Memory Usage (<80%)" "[[ \$(echo \"$MEMORY_USAGE < 80\" | bc -l) == 1 ]]"

CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
run_test "CPU Usage (<70%)" "[[ \$(echo \"$CPU_USAGE < 70\" | bc -l) == 1 ]]"

# Security Tests (basic)
echo "4. SECURITY TESTS"
run_test "No Debug Info in Prod" "[[ \"$ENVIRONMENT\" != \"production\" ]] || ! curl -s http://localhost:10010/health | grep -q debug"
run_test "HTTPS Redirect (Prod)" "[[ \"$ENVIRONMENT\" != \"production\" ]] || curl -I http://localhost:10010 | grep -q '301\\|302'"

# Integration Tests
echo "5. INTEGRATION TESTS"
run_test "Agent List" "curl -f -s http://localhost:10010/agents | jq -e '.agents | length > 0'"
run_test "Model List" "curl -f -s http://localhost:10010/models | jq -e '.models | length > 0'"

# Database Tests
echo "6. DATABASE TESTS"
run_test "PostgreSQL Version" "docker exec sutazai-postgres psql -U sutazai -d sutazai -c 'SELECT version();' | grep -q PostgreSQL"
run_test "Database Size Check" "docker exec sutazai-postgres psql -U sutazai -d sutazai -c 'SELECT pg_size_pretty(pg_database_size(current_database()));' | grep -q -E '[0-9]+ [GM]B'"

# Environment-Specific Tests
if [[ "$ENVIRONMENT" == "production" ]]; then
    echo "7. PRODUCTION-SPECIFIC TESTS"
    run_test "SSL Certificate Valid" "curl -f -s https://jarvis.company.com/health | jq -e '.status'"
    run_test "Monitoring Alerts" "curl -f -s http://localhost:10200/api/v1/alerts | jq -e '.data | length >= 0'"
    run_test "Backup Process" "ls -la /opt/sutazaiapp/backups/ | grep -q $(date +%Y%m%d)"
fi

# Summary
echo "=== VALIDATION SUMMARY ==="
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo "Success Rate: $(echo "scale=2; $TESTS_PASSED / ($TESTS_PASSED + $TESTS_FAILED) * 100" | bc)%"

if [[ $TESTS_FAILED -gt 0 ]]; then
    echo "❌ FAILED TESTS:"
    printf '%s\n' "${FAILED_TESTS[@]}"
    echo ""
    echo "🔄 DEPLOYMENT VALIDATION FAILED - Consider rollback"
    exit 1
else
    echo "✅ ALL TESTS PASSED - Deployment validation successful"
fi

echo "Validation log: $VALIDATION_LOG"
```

### Monitoring Setup Validation
```bash
#!/bin/bash
# validate_monitoring.sh

echo "=== MONITORING VALIDATION ==="

# Prometheus validation
echo "1. Validating Prometheus..."
curl -s http://localhost:10200/api/v1/targets | jq -e '.data.activeTargets | length > 0' || {
    echo "❌ Prometheus targets not configured"
    exit 1
}

# Grafana validation
echo "2. Validating Grafana..."
curl -s http://admin:admin@localhost:10201/api/health | jq -e '.database == "ok"' || {
    echo "❌ Grafana database connection failed"
    exit 1
}

# AlertManager validation
echo "3. Validating AlertManager..."
curl -s http://localhost:10203/api/v1/status | jq -e '.data.uptime' || {
    echo "❌ AlertManager not responding"
    exit 1
}

echo "✅ Monitoring stack validation completed"
```

## 🔄 Rollback Procedures

### Automatic Rollback Triggers
```bash
#!/bin/bash
# automatic_rollback_detector.sh
CURRENT_VERSION=$1
PREVIOUS_VERSION=$2
ROLLBACK_THRESHOLD=3  # Number of failed checks before rollback

FAILED_CHECKS=0
ROLLBACK_LOG="/opt/sutazaiapp/logs/rollback_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$ROLLBACK_LOG") 2>&1

echo "=== AUTOMATIC ROLLBACK MONITORING ==="
echo "Current Version: $CURRENT_VERSION"
echo "Previous Version: $PREVIOUS_VERSION"

# Monitoring loop (run for 10 minutes post-deployment)
for i in {1..60}; do  # 10 minutes with 10-second intervals
    
    # Health check
    if ! curl -f -s http://localhost:10010/health | jq -e '.status == "healthy"' >/dev/null; then
        echo "❌ Health check failed (attempt $i)"
        ((FAILED_CHECKS++))
    fi
    
    # Performance check
    RESPONSE_TIME=$(curl -w "%{time_total}" -s -o /dev/null http://localhost:10010/health 2>/dev/null || echo "999")
    if (( $(echo "$RESPONSE_TIME > 5.0" | bc -l) )); then
        echo "❌ Performance degradation detected: ${RESPONSE_TIME}s (attempt $i)"
        ((FAILED_CHECKS++))
    fi
    
    # Error rate check
    ERROR_COUNT=$(docker logs sutazai-backend --since=10s 2>/dev/null | grep -i error | wc -l)
    if [[ $ERROR_COUNT -gt 5 ]]; then
        echo "❌ High error rate detected: $ERROR_COUNT errors (attempt $i)"
        ((FAILED_CHECKS++))
    fi
    
    # Check if rollback threshold reached
    if [[ $FAILED_CHECKS -ge $ROLLBACK_THRESHOLD ]]; then
        echo "🚨 ROLLBACK THRESHOLD REACHED - Initiating automatic rollback"
        trigger_rollback "$CURRENT_VERSION" "$PREVIOUS_VERSION"
        exit 1
    fi
    
    # Reset counter if checks are passing
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        FAILED_CHECKS=0
        echo "✅ System recovered (attempt $i)"
    fi
    
    sleep 10
done

echo "✅ Monitoring completed - No rollback needed"
```

### Manual Rollback Process
```bash
#!/bin/bash
# manual_rollback.sh
ROLLBACK_TO_VERSION=$1
REASON=$2
ROLLBACK_ID="rollback_$(date +%Y%m%d_%H%M%S)"

echo "=== MANUAL ROLLBACK INITIATED ==="
echo "Rolling back to version: $ROLLBACK_TO_VERSION"
echo "Reason: $REASON"
echo "Rollback ID: $ROLLBACK_ID"

# Create rollback log
ROLLBACK_LOG="/opt/sutazaiapp/logs/${ROLLBACK_ID}.log"
exec > >(tee -a "$ROLLBACK_LOG") 2>&1

# Step 1: Stop current services gracefully
echo "1. Stopping current services..."
docker-compose stop backend frontend

# Step 2: Restore database from backup (if needed)
if [[ "$REASON" == *"database"* ]] || [[ "$REASON" == *"migration"* ]]; then
    echo "2. Database rollback required - restoring from backup..."
    
    BACKUP_FILE="/opt/sutazaiapp/backups/pre_deploy_$(date +%Y%m%d)*.sql.gz"
    LATEST_BACKUP=$(ls -t $BACKUP_FILE 2>/dev/null | head -1)
    
    if [[ -f "$LATEST_BACKUP" ]]; then
        echo "Restoring database from: $LATEST_BACKUP"
        gunzip -c "$LATEST_BACKUP" | docker exec -i sutazai-postgres psql -U sutazai sutazai
    else
        echo "❌ No database backup found for today!"
        exit 1
    fi
fi

# Step 3: Revert configuration files
echo "3. Reverting configuration..."
if [[ -f ".env.backup" ]]; then
    cp .env.backup .env
fi
if [[ -f "docker-compose.yml.backup" ]]; then
    cp docker-compose.yml.backup docker-compose.yml
fi

# Step 4: Update to previous version
echo "4. Updating to previous version..."
sed -i "s/JARVIS_VERSION=.*/JARVIS_VERSION=${ROLLBACK_TO_VERSION}/" .env
sed -i "s|image: .*backend:.*|image: your-registry.com/jarvis/backend:${ROLLBACK_TO_VERSION}|g" docker-compose.yml
sed -i "s|image: .*frontend:.*|image: your-registry.com/jarvis/frontend:${ROLLBACK_TO_VERSION}|g" docker-compose.yml

# Step 5: Start services
echo "5. Starting services with previous version..."
docker-compose up -d backend frontend

# Step 6: Verify rollback
echo "6. Verifying rollback..."
timeout 120 bash -c 'until curl -f -s http://localhost:10010/health; do sleep 5; done' || {
    echo "❌ Rollback verification failed!"
    exit 1
}

# Step 7: Validate functionality
echo "7. Running post-rollback validation..."
curl -f -s -X POST http://localhost:10010/simple-chat \
    -H "Content-Type: application/json" \
    -d '{"message": "rollback test"}' | jq '.response' || {
    echo "❌ Post-rollback functionality test failed!"
    exit 1
}

echo "✅ Rollback completed successfully"
echo "✅ System running on version: $ROLLBACK_TO_VERSION"
echo "Rollback log: $ROLLBACK_LOG"

# Notify team
curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"🔄 Jarvis rollback completed to version $ROLLBACK_TO_VERSION. Reason: $REASON\"}" \
    "$SLACK_WEBHOOK_URL" 2>/dev/null || true
```

### Blue/Green Rollback
```bash
#!/bin/bash
# blue_green_rollback.sh
CURRENT_COLOR=$1
PREVIOUS_COLOR=$([[ "$CURRENT_COLOR" == "blue" ]] && echo "green" || echo "blue")

echo "=== BLUE/GREEN ROLLBACK ==="
echo "Rolling back from $CURRENT_COLOR to $PREVIOUS_COLOR"

# Verify previous environment is still available
PREVIOUS_PORT=$([[ "$PREVIOUS_COLOR" == "blue" ]] && echo "10010" || echo "10012")
if ! curl -f -s "http://localhost:${PREVIOUS_PORT}/health" >/dev/null; then
    echo "❌ Previous environment ($PREVIOUS_COLOR) is not available!"
    echo "🚨 Manual intervention required"
    exit 1
fi

# Switch traffic back
echo "Switching traffic back to $PREVIOUS_COLOR environment..."

# Update load balancer (nginx example)
cat > /etc/nginx/sites-available/jarvis << EOF
upstream jarvis_backend {
    server localhost:${PREVIOUS_PORT};
}

server {
    listen 80;
    server_name jarvis.company.com;
    
    location / {
        proxy_pass http://jarvis_backend;
    }
}
EOF

# Reload nginx
nginx -t && nginx -s reload

echo "✅ Traffic switched back to $PREVIOUS_COLOR environment"

# Clean up failed environment
echo "🧹 Cleaning up failed $CURRENT_COLOR environment"
docker-compose -f "docker-compose.${CURRENT_COLOR}.yml" down

# Update environment marker
echo "$PREVIOUS_COLOR" > /opt/sutazaiapp/.current_environment

echo "✅ Blue/Green rollback completed"
```

## 🔧 Troubleshooting

### Common Deployment Issues

#### Issue 1: Container Won't Start
**Symptoms:**
- Container exits immediately
- "Unhealthy" status in docker-compose ps

**Diagnosis:**
```bash
# Check container logs
docker-compose logs backend

# Check resource constraints
docker stats --no-stream

# Verify image integrity
docker images | grep jarvis
```

**Resolution:**
```bash
# Rebuild image if corrupted
docker-compose build --no-cache backend

# Check for port conflicts
netstat -tulpn | grep :10010

# Verify environment variables
docker-compose config
```

#### Issue 2: Database Migration Failure
**Symptoms:**
- Backend starts but shows degraded status
- Database connection errors

**Diagnosis:**
```bash
# Check PostgreSQL logs
docker-compose logs sutazai-postgres

# Test database connectivity
docker exec sutazai-postgres pg_isready -U sutazai

# Check migration status
docker exec sutazai-backend python -c "from app.db import get_migration_status; print(get_migration_status())"
```

**Resolution:**
```bash
# Manual migration rollback
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
    DROP SCHEMA IF EXISTS migration_temp CASCADE;
    -- Restore from backup if needed
"

# Restore from backup
BACKUP_FILE="/opt/sutazaiapp/backups/pre_deploy_$(date +%Y%m%d)*.sql.gz"
gunzip -c "$BACKUP_FILE" | docker exec -i sutazai-postgres psql -U sutazai sutazai
```

#### Issue 3: High Memory Usage
**Symptoms:**
- System becomes unresponsive
- OOM killer activating

**Diagnosis:**
```bash
# Check memory usage by container
docker stats --no-stream | sort -k7 -nr

# Check system memory
free -h
dmesg | grep -i "killed process"
```

**Resolution:**
```bash
# Reduce Ollama memory usage
docker exec sutazai-ollama ollama rm model_name  # Remove unused models

# Limit container memory
docker-compose down
# Edit docker-compose.yml to add memory limits
docker-compose up -d

# Emergency cleanup
docker system prune -f --volumes
```

#### Issue 4: Load Balancer Issues
**Symptoms:**
- 502/503 errors
- Traffic not routing properly

**Diagnosis:**
```bash
# Check nginx configuration
nginx -t

# Check upstream health
curl -I http://localhost:10010/health
curl -I http://localhost:10012/health  # Blue/Green

# Check nginx logs
tail -f /var/log/nginx/error.log
```

**Resolution:**
```bash
# Reload nginx configuration
nginx -s reload

# Restart nginx if needed
systemctl restart nginx

# Manual traffic switch
# Update upstream configuration and reload
```

### Emergency Procedures

#### Complete System Recovery
```bash
#!/bin/bash
# emergency_recovery.sh
echo "=== EMERGENCY SYSTEM RECOVERY ==="

# Stop all services
docker-compose down

# Clean Docker system
docker system prune -f --volumes

# Restore from known good backup
BACKUP_DIR="/opt/sutazaiapp/backups"
LATEST_CONFIG=$(ls -t "$BACKUP_DIR"/config_backup_*.tar.gz | head -1)
LATEST_DB=$(ls -t "$BACKUP_DIR"/*pre_deploy*.sql.gz | head -1)

echo "Restoring configuration from: $LATEST_CONFIG"
tar -xzf "$LATEST_CONFIG" -C /opt/sutazaiapp/

echo "Restoring database from: $LATEST_DB"
docker-compose up -d sutazai-postgres
sleep 30
gunzip -c "$LATEST_DB" | docker exec -i sutazai-postgres psql -U sutazai sutazai

# Start system
docker-compose up -d

echo "✅ Emergency recovery completed"
```

#### Contact Information for Escalation
```bash
# Emergency Contacts
ONCALL_ENGINEER="+1-xxx-xxx-1111"
ENGINEERING_MANAGER="+1-xxx-xxx-2222"
SYSTEM_ADMIN="+1-xxx-xxx-3333"

# Emergency notification
send_emergency_alert() {
    local message=$1
    
    # Slack notification
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🚨 EMERGENCY: $message\"}" \
        "$EMERGENCY_SLACK_WEBHOOK"
    
    # Email notification (if configured)
    echo "$message" | mail -s "JARVIS EMERGENCY" "oncall@company.com"
    
    # PagerDuty integration (if configured)
    curl -X POST \
        -H "Authorization: Token token=${PAGERDUTY_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"incident\": {
                \"type\": \"incident\",
                \"title\": \"Jarvis Deployment Emergency\",
                \"service\": {
                    \"id\": \"${PAGERDUTY_SERVICE_ID}\",
                    \"type\": \"service_reference\"
                },
                \"body\": {
                    \"type\": \"incident_body\",
                    \"details\": \"$message\"
                }
            }
        }" \
        "https://api.pagerduty.com/incidents"
}
```

---

## 📋 Deployment Checklist Summary

### Pre-Deployment (30 minutes)
- [ ] All tests passing
- [ ] Code review completed
- [ ] Security scan passed
- [ ] Backup created and verified
- [ ] Team notified
- [ ] Rollback plan prepared

### Deployment (20-45 minutes)
- [ ] Environment health verified
- [ ] Images built and tagged
- [ ] Rolling/Blue-Green deployment executed
- [ ] Database migrations completed
- [ ] Configuration updated
- [ ] Services restarted successfully

### Post-Deployment (15-30 minutes)
- [ ] Health checks passing
- [ ] Functionality tests completed
- [ ] Performance validated
- [ ] Monitoring alerts configured
- [ ] Team notified of completion
- [ ] Documentation updated

### Monitoring (Ongoing)
- [ ] System metrics monitored
- [ ] Error rates within normal range
- [ ] User feedback monitored
- [ ] Performance metrics tracked
- [ ] Rollback triggers configured

---

*This deployment runbook is based on the actual Perfect Jarvis system architecture and real deployment scenarios. Update procedures as the system evolves and lessons are learned from actual deployments.*