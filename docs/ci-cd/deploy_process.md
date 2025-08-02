# Deployment Process Guide

## Overview

The SutazAI deployment process is designed for reliability, security, and scalability across multiple environments. This guide covers both automated CI/CD deployments and manual deployment procedures for different scenarios.

## Deployment Architecture

### Environment Structure
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │     Staging     │    │   Production    │
│                 │    │                 │    │                 │
│ • Local setup   │───►│ • Integration   │───►│ • Live system   │
│ • Feature dev   │    │ • Testing       │    │ • Blue/Green    │
│ • Unit tests    │    │ • User acceptance│    │ • Multi-region  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Deployment Targets
- **Local Development**: Single-machine deployment for development
- **Staging Environment**: Pre-production testing environment
- **Production Environment**: Live system with high availability
- **Disaster Recovery**: Backup deployment region

## Automated Deployment (CI/CD)

### Deployment Triggers

#### Automatic Triggers
```yaml
# Staging deployment triggers
staging_triggers:
  - push to develop branch
  - manual workflow dispatch
  - scheduled deployments (nightly)

# Production deployment triggers  
production_triggers:
  - push to main branch (with approval)
  - manual workflow dispatch (with approval)
  - hotfix deployments (emergency)
```

#### Manual Deployment Workflow
```bash
# Trigger manual deployment via GitHub CLI
gh workflow run deploy.yml \
  --ref main \
  --field environment=production \
  --field version=v1.2.3 \
  --field skip_tests=false
```

### Deployment Pipeline Stages

#### 1. Pre-Deployment Validation
```bash
#!/bin/bash
# Pre-deployment validation script

validate_environment() {
    echo "🔍 Validating deployment environment..."
    
    # Check cluster connectivity
    kubectl cluster-info || {
        echo "❌ Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    # Verify namespace exists
    kubectl get namespace $NAMESPACE || {
        echo "❌ Namespace $NAMESPACE does not exist"
        exit 1
    }
    
    # Check resource quotas
    AVAILABLE_CPU=$(kubectl describe quota -n $NAMESPACE | grep cpu | awk '{print $3}')
    REQUIRED_CPU="2000m"
    
    if [[ "$AVAILABLE_CPU" < "$REQUIRED_CPU" ]]; then
        echo "❌ Insufficient CPU resources"
        exit 1
    fi
    
    echo "✅ Environment validation passed"
}

validate_image() {
    echo "🔍 Validating container image..."
    
    # Check if image exists
    docker manifest inspect $IMAGE_NAME:$IMAGE_TAG > /dev/null || {
        echo "❌ Image $IMAGE_NAME:$IMAGE_TAG not found"
        exit 1
    }
    
    # Security scan results check
    if [[ -f "scan-results.json" ]]; then
        HIGH_VULNS=$(jq '.vulnerabilities[] | select(.severity=="HIGH") | length' scan-results.json)
        if [[ $HIGH_VULNS -gt 0 ]]; then
            echo "❌ Image contains $HIGH_VULNS high severity vulnerabilities"
            exit 1
        fi
    fi
    
    echo "✅ Image validation passed"
}
```

#### 2. Database Migration
```bash
#!/bin/bash
# Database migration script

run_migrations() {
    echo "🗄️ Running database migrations..."
    
    # Create migration job
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration-$(date +%s)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: migration
        image: $IMAGE_NAME:$IMAGE_TAG
        command: ["python", "-m", "alembic", "upgrade", "head"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
      restartPolicy: Never
  backoffLimit: 3
EOF
    
    # Wait for migration to complete
    kubectl wait --for=condition=complete job/db-migration-* --timeout=300s -n $NAMESPACE
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Database migration completed"
    else
        echo "❌ Database migration failed"
        kubectl logs job/db-migration-* -n $NAMESPACE
        exit 1
    fi
}

backup_database() {
    echo "💾 Creating database backup..."
    
    # Create backup before migration
    BACKUP_NAME="backup-$(date +%Y%m%d-%H%M%S)"
    
    kubectl run pg-dump-$BACKUP_NAME \
        --image=postgres:15 \
        --rm -i --restart=Never \
        --namespace=$NAMESPACE \
        --env="PGPASSWORD=$DB_PASSWORD" \
        -- pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_NAME.sql
    
    # Upload backup to S3
    aws s3 cp $BACKUP_NAME.sql s3://sutazai-backups/database/
    
    echo "✅ Database backup created: $BACKUP_NAME"
}
```

#### 3. Application Deployment
```bash
#!/bin/bash
# Application deployment script

deploy_application() {
    local strategy=${1:-rolling}
    
    echo "🚀 Deploying application using $strategy strategy..."
    
    case $strategy in
        "rolling")
            rolling_deployment
            ;;
        "blue-green")
            blue_green_deployment
            ;;
        "canary")
            canary_deployment
            ;;
        *)
            echo "❌ Unknown deployment strategy: $strategy"
            exit 1
            ;;
    esac
}

rolling_deployment() {
    echo "📊 Executing rolling deployment..."
    
    # Update deployment with new image
    kubectl set image deployment/sutazai-backend \
        backend=$IMAGE_NAME:$IMAGE_TAG \
        -n $NAMESPACE
    
    kubectl set image deployment/sutazai-frontend \
        frontend=$IMAGE_NAME:$IMAGE_TAG \
        -n $NAMESPACE
    
    # Wait for rollout to complete
    kubectl rollout status deployment/sutazai-backend -n $NAMESPACE --timeout=600s
    kubectl rollout status deployment/sutazai-frontend -n $NAMESPACE --timeout=600s
    
    echo "✅ Rolling deployment completed"
}

blue_green_deployment() {
    echo "🔄 Executing blue-green deployment..."
    
    # Determine current and next versions
    CURRENT_VERSION=$(kubectl get service sutazai-service -n $NAMESPACE -o jsonpath='{.spec.selector.version}')
    NEXT_VERSION=$([ "$CURRENT_VERSION" = "blue" ] && echo "green" || echo "blue")
    
    echo "Current version: $CURRENT_VERSION, Deploying: $NEXT_VERSION"
    
    # Deploy to inactive environment
    envsubst < k8s/deployment-template.yaml | \
        sed "s/{{VERSION}}/$NEXT_VERSION/g" | \
        kubectl apply -f - -n $NAMESPACE
    
    # Wait for deployment
    kubectl rollout status deployment/sutazai-$NEXT_VERSION -n $NAMESPACE --timeout=600s
    
    # Health check on new version
    if health_check "sutazai-$NEXT_VERSION:8000"; then
        # Switch traffic
        kubectl patch service sutazai-service -n $NAMESPACE \
            -p '{"spec":{"selector":{"version":"'$NEXT_VERSION'"}}}'
        
        echo "✅ Traffic switched to $NEXT_VERSION"
        
        # Scale down old version
        kubectl scale deployment/sutazai-$CURRENT_VERSION --replicas=0 -n $NAMESPACE
    else
        echo "❌ Health check failed, rolling back"
        kubectl delete deployment/sutazai-$NEXT_VERSION -n $NAMESPACE
        exit 1
    fi
}

canary_deployment() {
    echo "🐤 Executing canary deployment..."
    
    # Deploy canary version with 10% traffic
    envsubst < k8s/canary-deployment.yaml | kubectl apply -f - -n $NAMESPACE
    
    # Configure traffic split
    kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: sutazai-canary
  namespace: $NAMESPACE
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: sutazai-canary
  - route:
    - destination:
        host: sutazai-stable
      weight: 90
    - destination:
        host: sutazai-canary
      weight: 10
EOF
    
    echo "✅ Canary deployment active with 10% traffic"
    echo "Monitor metrics and run: ./scripts/promote-canary.sh to complete"
}
```

#### 4. Post-Deployment Verification
```bash
#!/bin/bash
# Post-deployment verification

health_check() {
    local endpoint=${1:-"http://sutazai-service:8000"}
    local max_attempts=${2:-30}
    local attempt=1
    
    echo "🏥 Running health checks..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -sf "$endpoint/health" > /dev/null; then
            echo "✅ Health check passed (attempt $attempt)"
            return 0
        fi
        
        echo "⏳ Health check failed, retrying... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    echo "❌ Health check failed after $max_attempts attempts"
    return 1
}

integration_tests() {
    echo "🧪 Running integration tests..."
    
    # Run test suite against deployed application
    kubectl run integration-tests \
        --image=$IMAGE_NAME:$IMAGE_TAG \
        --rm -i --restart=Never \
        --namespace=$NAMESPACE \
        --env="API_BASE_URL=http://sutazai-service:8000" \
        -- python -m pytest tests/integration/ -v
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Integration tests passed"
    else
        echo "❌ Integration tests failed"
        return 1
    fi
}

performance_test() {
    echo "⚡ Running performance tests..."
    
    # Basic load test
    kubectl run load-test \
        --image=loadimpact/k6 \
        --rm -i --restart=Never \
        --namespace=$NAMESPACE \
        -- run --vus 10 --duration 30s - <<EOF
import http from 'k6/http';
import { check } from 'k6';

export default function() {
  let response = http.get('http://sutazai-service:8000/health');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
}
EOF
    
    echo "✅ Performance test completed"
}

smoke_tests() {
    echo "💨 Running smoke tests..."
    
    local base_url="http://sutazai-service:8000"
    
    # Test critical endpoints
    endpoints=(
        "/health"
        "/api/v1/agents"
        "/api/v1/tasks"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -sf "$base_url$endpoint" > /dev/null; then
            echo "✅ $endpoint is responding"
        else
            echo "❌ $endpoint is not responding"
            return 1
        fi
    done
    
    echo "✅ All smoke tests passed"
}
```

## Manual Deployment Procedures

### Local Development Deployment

#### Single Command Deployment
```bash
#!/bin/bash
# Local development deployment

# Quick start script
./scripts/deploy-local.sh --profile development

# What this does:
# 1. Checks prerequisites (Docker, Docker Compose)
# 2. Builds containers locally
# 3. Starts all services
# 4. Runs health checks
# 5. Shows access URLs
```

#### Step-by-Step Local Setup
```bash
# 1. Clone repository
git clone https://github.com/your-org/sutazai.git
cd sutazai

# 2. Setup environment
cp .env.example .env
nano .env  # Edit configuration

# 3. Build and start services
docker-compose up --build -d

# 4. Wait for services to be ready
./scripts/wait-for-services.sh

# 5. Access the application
open http://localhost:8501  # Frontend
open http://localhost:8000/docs  # API documentation
```

### Staging Environment Deployment

#### Prerequisites Check
```bash
#!/bin/bash
# Staging deployment prerequisites

check_staging_prereqs() {
    echo "🔍 Checking staging deployment prerequisites..."
    
    # Check kubectl access
    kubectl get nodes --context=staging-cluster || {
        echo "❌ Cannot access staging cluster"
        exit 1
    }
    
    # Check container registry access
    docker login ghcr.io || {
        echo "❌ Cannot login to container registry"
        exit 1
    }
    
    # Check required secrets
    kubectl get secret database-secret -n staging || {
        echo "❌ Database secret not found"
        exit 1
    }
    
    echo "✅ Staging prerequisites satisfied"
}
```

#### Staging Deployment Script
```bash
#!/bin/bash
# Deploy to staging environment

deploy_staging() {
    local image_tag=${1:-latest}
    
    echo "🎯 Deploying to staging environment..."
    
    # Set context
    kubectl config use-context staging-cluster
    
    # Update deployment manifests
    envsubst < k8s/staging/deployment.yaml | kubectl apply -f -
    
    # Update image tags
    kubectl set image deployment/sutazai-backend \
        backend=ghcr.io/your-org/sutazai:$image_tag \
        -n staging
    
    # Wait for rollout
    kubectl rollout status deployment/sutazai-backend -n staging --timeout=600s
    
    # Run verification
    health_check "https://staging.sutazai.com"
    integration_tests
    
    echo "✅ Staging deployment completed"
}

# Usage
deploy_staging v1.2.3
```

### Production Deployment

#### Production Deployment Checklist
```markdown
## Pre-Deployment Checklist

### Planning
- [ ] Deployment window scheduled and communicated
- [ ] Rollback plan prepared and tested
- [ ] Database backup completed
- [ ] Monitoring alerts configured
- [ ] Incident response team on standby

### Technical Validation
- [ ] Image security scan passed
- [ ] All tests passing in CI/CD
- [ ] Staging environment validated
- [ ] Performance benchmarks met
- [ ] Database migrations tested

### Approvals
- [ ] Technical lead approval
- [ ] Security team approval  
- [ ] Business stakeholder approval
- [ ] Change advisory board approval (if required)

### Communication
- [ ] Users notified of maintenance window
- [ ] Support team briefed on changes
- [ ] Monitoring team alerted
- [ ] Documentation updated
```

#### Production Deployment Process
```bash
#!/bin/bash
# Production deployment with safety checks

deploy_production() {
    local image_tag=$1
    
    if [[ -z "$image_tag" ]]; then
        echo "❌ Image tag required"
        echo "Usage: deploy_production <image_tag>"
        exit 1
    fi
    
    echo "🎯 Starting production deployment for $image_tag"
    
    # Confirmation prompt
    read -p "Are you sure you want to deploy to production? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Deployment cancelled"
        exit 1
    fi
    
    # Final validation
    validate_production_readiness $image_tag
    
    # Create deployment checkpoint
    create_checkpoint
    
    # Execute deployment
    case $DEPLOYMENT_STRATEGY in
        "blue-green")
            blue_green_production_deployment $image_tag
            ;;
        "canary")
            canary_production_deployment $image_tag
            ;;
        *)
            echo "❌ Invalid deployment strategy: $DEPLOYMENT_STRATEGY"
            exit 1
            ;;
    esac
    
    # Post-deployment verification
    production_verification
    
    # Update deployment tracking
    update_deployment_log $image_tag
    
    echo "✅ Production deployment completed successfully"
}

validate_production_readiness() {
    local image_tag=$1
    
    echo "🔍 Validating production readiness..."
    
    # Check staging validation
    staging_url="https://staging.sutazai.com"
    if ! curl -sf "$staging_url/health" > /dev/null; then
        echo "❌ Staging environment is not healthy"
        exit 1
    fi
    
    # Check image exists and is scanned
    docker manifest inspect ghcr.io/your-org/sutazai:$image_tag > /dev/null || {
        echo "❌ Image not found: $image_tag"
        exit 1
    }
    
    # Verify no critical issues in monitoring
    if check_critical_alerts; then
        echo "❌ Critical alerts active, cannot deploy"
        exit 1
    fi
    
    echo "✅ Production readiness validated"
}
```

## Rollback Procedures

### Automatic Rollback
```bash
#!/bin/bash
# Automatic rollback based on health checks

auto_rollback() {
    local deployment_name=$1
    local namespace=$2
    local health_endpoint=$3
    
    echo "🔄 Initiating automatic rollback..."
    
    # Wait for rollout and check health
    if ! kubectl rollout status deployment/$deployment_name -n $namespace --timeout=300s; then
        echo "⚠️ Rollout failed, initiating rollback"
        kubectl rollout undo deployment/$deployment_name -n $namespace
        return
    fi
    
    # Health check with timeout
    if ! health_check_with_timeout $health_endpoint 60; then
        echo "⚠️ Health check failed, initiating rollback"
        kubectl rollout undo deployment/$deployment_name -n $namespace
        
        # Wait for rollback to complete
        kubectl rollout status deployment/$deployment_name -n $namespace --timeout=300s
        
        # Verify rollback health
        if health_check_with_timeout $health_endpoint 30; then
            echo "✅ Rollback completed successfully"
        else
            echo "❌ Rollback health check failed - manual intervention required"
            exit 1
        fi
    fi
}
```

### Manual Rollback
```bash
#!/bin/bash
# Manual rollback to previous version

manual_rollback() {
    local environment=$1
    local target_version=${2:-"previous"}
    
    echo "🔄 Manual rollback initiated for $environment"
    
    case $environment in
        "staging")
            kubectl config use-context staging-cluster
            namespace="staging"
            ;;
        "production")
            kubectl config use-context production-cluster
            namespace="production"
            ;;
        *)
            echo "❌ Invalid environment: $environment"
            exit 1
            ;;
    esac
    
    if [[ "$target_version" == "previous" ]]; then
        # Rollback to previous version
        kubectl rollout undo deployment/sutazai-backend -n $namespace
        kubectl rollout undo deployment/sutazai-frontend -n $namespace
    else
        # Rollback to specific version
        kubectl set image deployment/sutazai-backend \
            backend=ghcr.io/your-org/sutazai:$target_version \
            -n $namespace
        kubectl set image deployment/sutazai-frontend \
            frontend=ghcr.io/your-org/sutazai:$target_version \
            -n $namespace
    fi
    
    # Wait for rollback completion
    kubectl rollout status deployment/sutazai-backend -n $namespace --timeout=600s
    kubectl rollout status deployment/sutazai-frontend -n $namespace --timeout=600s
    
    # Verify rollback
    health_check "https://$environment.sutazai.com"
    
    echo "✅ Rollback completed for $environment"
}
```

## Monitoring and Alerts

### Deployment Monitoring
```bash
#!/bin/bash
# Setup deployment monitoring

setup_deployment_monitoring() {
    echo "📊 Setting up deployment monitoring..."
    
    # Create deployment tracking
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: deployment-info
  namespace: $NAMESPACE
data:
  version: "$IMAGE_TAG"
  deployment_time: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  deployed_by: "$USER"
  git_commit: "$(git rev-parse HEAD)"
EOF
    
    # Send deployment event to monitoring
    curl -X POST "https://monitoring.sutazai.com/api/events" \
        -H "Content-Type: application/json" \
        -d '{
            "event": "deployment_started",
            "environment": "'$ENVIRONMENT'",
            "version": "'$IMAGE_TAG'",
            "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
        }'
}
```

### Alert Configuration
```yaml
# Deployment alerts
alerts:
  - name: deployment_failure
    condition: |
      increase(deployment_failures_total[5m]) > 0
    severity: critical
    message: "Deployment failed in {{ $labels.environment }}"
    
  - name: rollback_triggered
    condition: |
      increase(deployment_rollbacks_total[5m]) > 0
    severity: warning
    message: "Rollback triggered in {{ $labels.environment }}"
    
  - name: deployment_duration
    condition: |
      deployment_duration_seconds > 1800
    severity: warning
    message: "Deployment taking longer than 30 minutes"
```

## Disaster Recovery

### Backup and Restore
```bash
#!/bin/bash
# Disaster recovery procedures

create_full_backup() {
    local backup_name="disaster-recovery-$(date +%Y%m%d-%H%M%S)"
    
    echo "💾 Creating full system backup: $backup_name"
    
    # Database backup
    kubectl exec -n production deployment/postgres -- \
        pg_dump -U postgres sutazai > $backup_name-database.sql
    
    # Persistent volume backup
    kubectl get pv -o yaml > $backup_name-pv.yaml
    kubectl get pvc -n production -o yaml > $backup_name-pvc.yaml
    
    # Configuration backup
    kubectl get configmap -n production -o yaml > $backup_name-config.yaml
    kubectl get secret -n production -o yaml > $backup_name-secrets.yaml
    
    # Upload to remote storage
    tar czf $backup_name.tar.gz $backup_name-*
    aws s3 cp $backup_name.tar.gz s3://sutazai-disaster-recovery/
    
    echo "✅ Backup completed: $backup_name"
}

restore_from_backup() {
    local backup_name=$1
    
    echo "🔄 Restoring from backup: $backup_name"
    
    # Download backup
    aws s3 cp s3://sutazai-disaster-recovery/$backup_name.tar.gz .
    tar xzf $backup_name.tar.gz
    
    # Restore database
    kubectl exec -n production deployment/postgres -- \
        psql -U postgres -d sutazai < $backup_name-database.sql
    
    # Restore configurations
    kubectl apply -f $backup_name-config.yaml
    kubectl apply -f $backup_name-secrets.yaml
    
    # Restart applications
    kubectl rollout restart deployment/sutazai-backend -n production
    kubectl rollout restart deployment/sutazai-frontend -n production
    
    echo "✅ Restore completed"
}
```

This comprehensive deployment process ensures reliable, secure, and scalable deployments of the SutazAI system across all environments.