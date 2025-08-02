# CI/CD Pipeline Guide

## Overview

The SutazAI project implements a comprehensive CI/CD pipeline that automates testing, building, security scanning, and deployment processes. The pipeline ensures code quality, security compliance, and reliable deployments across multiple environments.

## Pipeline Architecture

### Pipeline Stages Overview
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Source    │    │   Build     │    │    Test     │    │  Security   │
│  Control    │───►│   Stage     │───►│   Stage     │───►│   Stage     │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                            │                     │              │
                            ▼                     ▼              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Deploy     │    │   Package   │    │   Quality   │    │  Security   │
│   Stage     │◄───│   Stage     │◄───│   Gates     │◄───│   Scan      │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## GitHub Actions Workflow

### Main Workflow (`.github/workflows/main.yml`)
```yaml
name: SutazAI CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  # Code Quality and Linting
  lint-and-format:
    name: Code Quality Check
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
          
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install black isort flake8 pylint mypy bandit safety
          pip install -r requirements.txt
          
      - name: Format check with Black
        run: black --check --diff .
        
      - name: Import sorting check
        run: isort --check-only --diff .
        
      - name: Lint with flake8
        run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        
      - name: Type checking with mypy
        run: mypy backend/ --ignore-missing-imports
        
      - name: Security check with bandit
        run: bandit -r backend/ -f json -o bandit-report.json
        
      - name: Dependency security check
        run: safety check --json --output safety-report.json
        
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json

  # Unit and Integration Tests
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    needs: lint-and-format
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: sutazai_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
          
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-test.txt
          
      - name: Setup test environment
        run: |
          cp .env.example .env.test
          echo "DATABASE_URL=postgresql://postgres:test_password@localhost:5432/sutazai_test" >> .env.test
          echo "REDIS_URL=redis://localhost:6379/1" >> .env.test
          
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=backend --cov-report=xml --cov-report=html
          
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v --cov-append --cov=backend --cov-report=xml
          
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          
      - name: Archive test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results
          path: |
            htmlcov/
            .coverage
            pytest-report.xml

  # Security Scanning
  security-scan:
    name: Security Analysis
    runs-on: ubuntu-latest
    needs: lint-and-format
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
          
      - name: Run Semgrep security analysis
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/secrets
            p/ci
            p/docker
            
      - name: SAST with CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python, javascript
          
      - name: Autobuild
        uses: github/codeql-action/autobuild@v2
        
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

  # Build Docker Images
  build:
    name: Build and Push Images
    runs-on: ubuntu-latest
    needs: [test, security-scan]
    if: github.event_name == 'push' || github.event_name == 'workflow_dispatch'
    
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
      image-tag: ${{ steps.meta.outputs.tags }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}
            
      - name: Build and push Docker image
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
          
      - name: Generate SBOM
        uses: anchore/sbom-action@v0
        with:
          image: ${{ steps.meta.outputs.tags }}
          format: spdx-json
          output-file: sbom.spdx.json
          
      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: sbom.spdx.json

  # Deploy to Staging
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop' || (github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'staging')
    environment: staging
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Setup kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'latest'
          
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
          
      - name: Update kubeconfig
        run: aws eks update-kubeconfig --name sutazai-staging --region us-east-1
        
      - name: Deploy to staging
        run: |
          envsubst < k8s/staging/deployment.yaml | kubectl apply -f -
          kubectl rollout status deployment/sutazai-backend -n staging
          kubectl rollout status deployment/sutazai-frontend -n staging
        env:
          IMAGE_TAG: ${{ needs.build.outputs.image-tag }}
          
      - name: Run smoke tests
        run: |
          kubectl wait --for=condition=ready pod -l app=sutazai-backend -n staging --timeout=300s
          ./scripts/smoke-tests.sh https://staging.sutazai.com
          
      - name: Notify deployment
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          channel: '#deployments'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}

  # Deploy to Production
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build, deploy-staging]
    if: github.ref == 'refs/heads/main' || (github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'production')
    environment: production
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Manual approval checkpoint
        uses: trstringer/manual-approval@v1
        with:
          secret: ${{ github.TOKEN }}
          approvers: team-leads,devops-team
          minimum-approvals: 2
          issue-title: "Deploy SutazAI to Production"
          issue-body: |
            Please approve deployment of SutazAI to production.
            
            **Changes:**
            ${{ github.event.head_commit.message }}
            
            **Image:** ${{ needs.build.outputs.image-tag }}
            **Commit:** ${{ github.sha }}
            
      - name: Deploy to production
        run: |
          # Blue-green deployment implementation
          ./scripts/blue-green-deploy.sh production ${{ needs.build.outputs.image-tag }}
          
      - name: Health check
        run: |
          ./scripts/health-check.sh https://api.sutazai.com
          ./scripts/integration-tests.sh https://api.sutazai.com
          
      - name: Update deployment tracking
        run: |
          echo "Deployment completed at $(date)" >> DEPLOYMENT_LOG.md
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add DEPLOYMENT_LOG.md
          git commit -m "Update deployment log" || exit 0
          git push
```

### Specialized Workflows

#### Security Scanning Workflow (`.github/workflows/security.yml`)
```yaml
name: Security Scanning

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Dependency Review
        uses: actions/dependency-review-action@v3
        
  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Build test image
        run: docker build -t sutazai:test .
        
      - name: Run Snyk container scan
        uses: snyk/actions/docker@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          image: sutazai:test
          args: --severity-threshold=high
          
  secrets-scan:
    name: Secrets Detection
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: TruffleHog OSS
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD
```

## Build Process

### Multi-Stage Dockerfile
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements*.txt ./
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim as runtime

# Create non-root user
RUN groupadd -r sutazai && useradd -r -g sutazai sutazai

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/sutazai/.local

# Copy application code
COPY --chown=sutazai:sutazai . .

# Set environment variables
ENV PATH=/home/sutazai/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER sutazai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build Scripts

#### Build Script (`scripts/build.sh`)
```bash
#!/bin/bash
set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_ID="${BUILD_ID:-$(date +%Y%m%d-%H%M%S)}"
IMAGE_NAME="${IMAGE_NAME:-sutazai}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

cleanup() {
    log "Cleaning up temporary files..."
    # Add cleanup logic here
}

trap cleanup EXIT

# Pre-build validation
validate_environment() {
    log "Validating build environment..."
    
    # Check required tools
    command -v docker >/dev/null 2>&1 || { log "Docker is required but not installed"; exit 1; }
    command -v python3 >/dev/null 2>&1 || { log "Python 3 is required but not installed"; exit 1; }
    
    # Check for required files
    [[ -f "$PROJECT_ROOT/Dockerfile" ]] || { log "Dockerfile not found"; exit 1; }
    [[ -f "$PROJECT_ROOT/requirements.txt" ]] || { log "requirements.txt not found"; exit 1; }
    
    log "Environment validation passed"
}

# Security scanning
run_security_scan() {
    log "Running security scans..."
    
    # Scan for secrets
    if command -v trufflehog >/dev/null 2>&1; then
        trufflehog filesystem "$PROJECT_ROOT" --no-update
    fi
    
    # Dependency security check
    if command -v safety >/dev/null 2>&1; then
        safety check -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    log "Security scans completed"
}

# Build Docker image
build_image() {
    log "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
    
    cd "$PROJECT_ROOT"
    
    docker build \
        --build-arg BUILD_ID="$BUILD_ID" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        --tag "$IMAGE_NAME:$IMAGE_TAG" \
        --tag "$IMAGE_NAME:$BUILD_ID" \
        .
    
    log "Docker image built successfully"
}

# Test image
test_image() {
    log "Testing Docker image..."
    
    # Start container for testing
    CONTAINER_ID=$(docker run -d -p 0:8000 "$IMAGE_NAME:$IMAGE_TAG")
    
    # Wait for service to start
    sleep 10
    
    # Get mapped port
    PORT=$(docker port "$CONTAINER_ID" 8000/tcp | cut -d: -f2)
    
    # Health check
    if curl -f "http://localhost:$PORT/health" >/dev/null 2>&1; then
        log "Health check passed"
    else
        log "Health check failed"
        docker logs "$CONTAINER_ID"
        docker stop "$CONTAINER_ID"
        exit 1
    fi
    
    # Cleanup test container
    docker stop "$CONTAINER_ID"
    
    log "Image testing completed"
}

# Main execution
main() {
    log "Starting build process (Build ID: $BUILD_ID)"
    
    validate_environment
    run_security_scan
    build_image
    test_image
    
    log "Build completed successfully"
    log "Image: $IMAGE_NAME:$IMAGE_TAG"
    log "Build ID: $BUILD_ID"
}

# Run main function
main "$@"
```

## Testing Strategy

### Test Automation Pipeline
```yaml
# Test matrix configuration
test-matrix:
  strategy:
    matrix:
      python-version: ['3.9', '3.10', '3.11']
      os: [ubuntu-latest, macos-latest, windows-latest]
      include:
        - python-version: '3.11'
          os: ubuntu-latest
          coverage: true
```

### Test Categories

#### 1. Unit Tests
```bash
# Fast unit tests
pytest tests/unit/ \
    --cov=backend \
    --cov-report=xml \
    --cov-report=html \
    --junit-xml=junit-unit.xml \
    --maxfail=1 \
    -x
```

#### 2. Integration Tests
```bash
# Integration tests with services
docker-compose -f docker-compose.test.yml up -d
pytest tests/integration/ \
    --junit-xml=junit-integration.xml \
    --maxfail=3
docker-compose -f docker-compose.test.yml down
```

#### 3. End-to-End Tests
```bash
# Full system tests
pytest tests/e2e/ \
    --junit-xml=junit-e2e.xml \
    --browser=chrome \
    --headless
```

#### 4. Performance Tests
```bash
# Load testing with locust
locust -f tests/performance/locustfile.py \
    --host=http://localhost:8000 \
    --users=100 \
    --spawn-rate=10 \
    --run-time=5m \
    --html=performance-report.html
```

## Quality Gates

### Code Quality Thresholds
```yaml
quality-gates:
  coverage:
    minimum: 80%
    target: 90%
  
  complexity:
    max-cyclomatic: 10
    max-cognitive: 15
  
  duplication:
    max-percentage: 3%
  
  security:
    max-high-vulnerabilities: 0
    max-medium-vulnerabilities: 5
  
  performance:
    max-response-time: 500ms
    min-throughput: 1000rps
```

### Quality Check Script
```bash
#!/bin/bash
# Quality gate validation

# Coverage check
COVERAGE=$(python -m coverage report --show-missing | grep TOTAL | awk '{print $4}' | sed 's/%//')
if (( $(echo "$COVERAGE < 80" | bc -l) )); then
    echo "Coverage $COVERAGE% is below threshold (80%)"
    exit 1
fi

# Security vulnerability check
HIGH_VULNS=$(cat security-report.json | jq '.vulnerabilities[] | select(.severity=="HIGH") | length')
if [[ $HIGH_VULNS -gt 0 ]]; then
    echo "Found $HIGH_VULNS high severity vulnerabilities"
    exit 1
fi

# Performance check
RESPONSE_TIME=$(cat performance-report.json | jq '.stats.avg_response_time')
if (( $(echo "$RESPONSE_TIME > 500" | bc -l) )); then
    echo "Average response time ${RESPONSE_TIME}ms exceeds threshold (500ms)"
    exit 1
fi

echo "All quality gates passed"
```

## Environment Management

### Environment Configuration
```yaml
# Staging environment
staging:
  cluster: sutazai-staging
  namespace: staging
  replicas: 2
  resources:
    cpu: 500m
    memory: 1Gi
  scaling:
    min: 2
    max: 5
    cpu-threshold: 70%

# Production environment  
production:
  cluster: sutazai-production
  namespace: production
  replicas: 3
  resources:
    cpu: 1000m
    memory: 2Gi
  scaling:
    min: 3
    max: 10
    cpu-threshold: 60%
```

### Deployment Strategies

#### Blue-Green Deployment
```bash
#!/bin/bash
# Blue-green deployment script

ENVIRONMENT=$1
IMAGE_TAG=$2

# Determine current and next versions
CURRENT=$(kubectl get service sutazai-service -o jsonpath='{.spec.selector.version}')
NEXT=$([ "$CURRENT" = "blue" ] && echo "green" || echo "blue")

echo "Deploying $NEXT version with image $IMAGE_TAG"

# Update deployment with new image
kubectl set image deployment/sutazai-$NEXT sutazai=$IMAGE_TAG

# Wait for rollout
kubectl rollout status deployment/sutazai-$NEXT --timeout=600s

# Health check
if ./scripts/health-check.sh "http://sutazai-$NEXT:8000"; then
    # Switch traffic
    kubectl patch service sutazai-service -p '{"spec":{"selector":{"version":"'$NEXT'"}}}'
    echo "Traffic switched to $NEXT"
    
    # Scale down old version
    kubectl scale deployment/sutazai-$CURRENT --replicas=0
else
    echo "Health check failed, keeping current version"
    exit 1
fi
```

## Monitoring and Observability

### Pipeline Monitoring
```yaml
# Pipeline metrics collection
metrics:
  - build_duration
  - test_duration
  - deployment_duration
  - success_rate
  - failure_rate
  - mean_time_to_recovery

alerts:
  - name: build_failure
    condition: success_rate < 95%
    notification: slack, email
    
  - name: slow_pipeline
    condition: build_duration > 30min
    notification: slack
```

### Deployment Notifications
```bash
# Slack notification function
notify_slack() {
    local status=$1
    local environment=$2
    local commit=$3
    
    curl -X POST -H 'Content-type: application/json' \
        --data "{
            \"text\": \"Deployment $status\",
            \"attachments\": [{
                \"color\": \"$([ "$status" = "success" ] && echo "good" || echo "danger")\",
                \"fields\": [
                    {\"title\": \"Environment\", \"value\": \"$environment\", \"short\": true},
                    {\"title\": \"Commit\", \"value\": \"$commit\", \"short\": true}
                ]
            }]
        }" \
        $SLACK_WEBHOOK_URL
}
```

This comprehensive CI/CD pipeline ensures reliable, secure, and automated delivery of the SutazAI system across all environments.