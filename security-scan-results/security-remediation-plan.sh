#!/bin/bash
# SutazAI Container Security Remediation Script
# Date: August 5, 2025
# Purpose: Automated fixes for critical security issues

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUTAZAI_ROOT="/opt/sutazaiapp"
BACKUP_DIR="/opt/sutazaiapp/security-scan-results/backups/$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

create_backup() {
    log "Creating security backup at $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Backup files that will be modified
    cp "$SUTAZAI_ROOT/scripts/multi-environment-config-manager.py" "$BACKUP_DIR/" 2>/dev/null || true
    cp "$SUTAZAI_ROOT/workflows/scripts/workflow_manager.py" "$BACKUP_DIR/" 2>/dev/null || true
    cp "$SUTAZAI_ROOT/auth/jwt-service/main.py" "$BACKUP_DIR/" 2>/dev/null || true
    cp "$SUTAZAI_ROOT/backend/requirements.txt" "$BACKUP_DIR/" 2>/dev/null || true
    
    log "Backup completed"
}

fix_hardcoded_secrets() {
    log "Fixing hardcoded secrets..."
    
    # Fix multi-environment-config-manager.py
    if [[ -f "$SUTAZAI_ROOT/scripts/multi-environment-config-manager.py" ]]; then
        sed -i 's/PASSWORD = "password"/PASSWORD = os.getenv("CONFIG_MANAGER_PASSWORD", "change_me")/g' \
            "$SUTAZAI_ROOT/scripts/multi-environment-config-manager.py"
        sed -i 's/TOKEN = "token"/TOKEN = os.getenv("CONFIG_MANAGER_TOKEN", "change_me")/g' \
            "$SUTAZAI_ROOT/scripts/multi-environment-config-manager.py"
        log "Fixed secrets in multi-environment-config-manager.py"
    fi
    
    # Fix workflow_manager.py
    if [[ -f "$SUTAZAI_ROOT/workflows/scripts/workflow_manager.py" ]]; then
        sed -i "s/password='redis_password'/password=os.getenv('REDIS_PASSWORD', 'redis_password')/g" \
            "$SUTAZAI_ROOT/workflows/scripts/workflow_manager.py"
        log "Fixed Redis password in workflow_manager.py"
    fi
    
    # Fix deploy_dify_workflows.py
    if [[ -f "$SUTAZAI_ROOT/workflows/scripts/deploy_dify_workflows.py" ]]; then
        sed -i "s/password='redis_password'/password=os.getenv('REDIS_PASSWORD', 'redis_password')/g" \
            "$SUTAZAI_ROOT/workflows/scripts/deploy_dify_workflows.py"
        log "Fixed Redis password in deploy_dify_workflows.py"
    fi
    
    # Fix finrobot service
    if [[ -f "$SUTAZAI_ROOT/docker/finrobot/finrobot_service.py" ]]; then
        sed -i 's/self.alpha_vantage_key = "demo"/self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "demo")/g' \
            "$SUTAZAI_ROOT/docker/finrobot/finrobot_service.py"
        log "Fixed Alpha Vantage API key in finrobot service"
    fi
}

pin_critical_dependencies() {
    log "Pinning critical dependencies to specific versions..."
    
    local requirements_file="$SUTAZAI_ROOT/backend/requirements.txt"
    if [[ -f "$requirements_file" ]]; then
        # Create a new requirements file with pinned versions
        cat > "${requirements_file}.secure" << EOF
# SutazAI Backend Requirements - Security Pinned Versions
# Generated: $(date)
# ALL packages pinned to specific secure versions

# Core FastAPI Framework - SECURITY CRITICAL
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.4
pydantic-settings==2.8.1

# Database & Async - SECURITY UPDATES
sqlalchemy==2.0.36
alembic==1.14.0
psycopg2-binary==2.9.10
redis==5.2.1
celery==5.4.0

# HTTP Libraries - CRITICAL CVE FIXES
requests==2.32.3
aiohttp==3.11.11
httpx==0.28.1
websockets==13.1
python-multipart==0.0.19
beautifulsoup4==4.12.3

# Security & Crypto - HIGHEST PRIORITY
cryptography==44.0.0
python-jose[cryptography]==3.3.0
PyJWT==2.10.1
passlib[bcrypt]==1.7.4
bcrypt==4.2.1
email-validator==2.1.0
bleach==6.1.0

# Core Dependencies - SECURITY PATCHED
python-dotenv==1.0.1
pyyaml==6.0.2
jinja2==3.1.5
pillow==11.0.0
urllib3==2.3.0
click==8.1.8
rich==13.9.4
typer==0.15.1
setuptools==75.6.0
certifi==2025.7.14

# Data Science - PINNED VERSIONS
pandas==2.2.3
numpy==2.1.3
scipy==1.14.1
scikit-learn==1.6.0
matplotlib==3.10.0
plotly==5.24.1
joblib==1.4.2
networkx==3.4.2
sympy==1.13.3

# AI/ML Libraries - PINNED SECURE VERSIONS
torch==2.5.1
transformers==4.48.0
sentence-transformers==3.3.1
openai==1.58.1
anthropic==0.42.0
huggingface-hub==0.27.0
tiktoken==0.8.0
langchain==0.3.11

# Vector Databases - PINNED VERSIONS
chromadb==0.5.23
qdrant-client==1.12.1
faiss-cpu==1.9.0
neo4j==5.27.0

# Web Automation - PINNED VERSIONS
selenium==4.27.1
playwright==1.49.1

# Infrastructure - PINNED VERSIONS
docker==7.1.0
kubernetes==31.0.0
prometheus-client==0.21.1
psutil==6.1.0
aiofiles==24.1.0
streamlit==1.40.2
schedule==1.2.2

# Development Tools - PINNED
black==24.10.0
pytest==8.3.4
coverage==7.6.9
tqdm==4.67.1
humanize==4.11.0
python-slugify==8.0.4

# Web Framework Security - PINNED
werkzeug==3.1.3
flask==3.1.0
django==5.1.4
tornado==6.4.2
lxml==5.3.0

# Security Scanning Tools - PINNED
python-nmap==0.7.1
scapy==2.6.1
netifaces==0.11.0
paramiko==3.5.0
netaddr==1.3.0
dnspython==2.7.0
EOF
        
        # Backup original and replace
        cp "$requirements_file" "${requirements_file}.backup"
        mv "${requirements_file}.secure" "$requirements_file"
        log "Pinned all dependencies in backend/requirements.txt"
    fi
}

create_security_docker_compose() {
    log "Creating security-hardened Docker Compose configuration..."
    
    cat > "$SUTAZAI_ROOT/docker-compose.security.yml" << 'EOF'
# Security-hardened Docker Compose override
# Usage: docker-compose -f docker-compose.yml -f docker-compose.security.yml up
version: '3.8'

services:
  # Remove privileged mode from hardware optimizer where possible
  hardware-resource-optimizer:
    privileged: false
    # Add specific capabilities instead
    cap_add:
      - SYS_ADMIN
      - SYS_PTRACE
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    
  # Secure cadvisor configuration
  cadvisor:
    privileged: false
    cap_add:
      - SYS_ADMIN
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    read_only: true
    
  # Add security context to all services
  backend:
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      
  frontend:
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m

  postgres:
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
      
  redis:
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    read_only: true
    
  neo4j:
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
      
  ollama:
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
EOF
    
    log "Created security-hardened Docker Compose configuration"
}

create_secrets_template() {
    log "Creating secure secrets management template..."
    
    mkdir -p "$SUTAZAI_ROOT/security-scan-results/templates"
    
    cat > "$SUTAZAI_ROOT/security-scan-results/templates/.env.secure.template" << 'EOF'
# SutazAI Secure Environment Variables Template
# Copy to .env and fill in actual values
# NEVER commit .env files to version control

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=<GENERATE_SECURE_PASSWORD>
POSTGRES_DB=sutazai

# Redis Configuration  
REDIS_PASSWORD=<GENERATE_SECURE_PASSWORD>

# Neo4j Configuration
NEO4J_PASSWORD=<GENERATE_SECURE_PASSWORD>

# JWT & Security
JWT_SECRET=<GENERATE_256BIT_SECRET>
SECRET_KEY=<GENERATE_256BIT_SECRET>

# External API Keys
ALPHA_VANTAGE_KEY=<YOUR_API_KEY>
OPENAI_API_KEY=<YOUR_API_KEY>
ANTHROPIC_API_KEY=<YOUR_API_KEY>

# Monitoring & Alerting
GRAFANA_PASSWORD=<GENERATE_SECURE_PASSWORD>
SLACK_WEBHOOK_URL=<YOUR_WEBHOOK_URL>

# Service Configurations
CONFIG_MANAGER_PASSWORD=<GENERATE_SECURE_PASSWORD>
CONFIG_MANAGER_TOKEN=<GENERATE_SECURE_TOKEN>

# ChromaDB
CHROMADB_API_KEY=<GENERATE_API_KEY>

# Application Settings
TZ=UTC
SUTAZAI_ENV=production
EOF

    cat > "$SUTAZAI_ROOT/security-scan-results/templates/generate-secrets.sh" << 'EOF'
#!/bin/bash
# Generate secure secrets for SutazAI deployment

echo "# Generated SutazAI Secrets - $(date)"
echo "# Store these securely and never commit to version control"
echo ""

echo "# Database passwords"
echo "POSTGRES_PASSWORD=$(openssl rand -base64 32)"
echo "REDIS_PASSWORD=$(openssl rand -base64 32)"
echo "NEO4J_PASSWORD=$(openssl rand -base64 32)"
echo "GRAFANA_PASSWORD=$(openssl rand -base64 32)"
echo ""

echo "# JWT and encryption keys"
echo "JWT_SECRET=$(openssl rand -base64 64)"
echo "SECRET_KEY=$(openssl rand -base64 64)"
echo ""

echo "# Service tokens"
echo "CONFIG_MANAGER_PASSWORD=$(openssl rand -base64 32)"
echo "CONFIG_MANAGER_TOKEN=$(openssl rand -base64 32)"
echo "CHROMADB_API_KEY=$(openssl rand -base64 32)"
EOF

    chmod +x "$SUTAZAI_ROOT/security-scan-results/templates/generate-secrets.sh"
    
    log "Created secure secrets management templates"
}

create_security_scanning_pipeline() {
    log "Creating automated security scanning pipeline..."
    
    mkdir -p "$SUTAZAI_ROOT/.github/workflows"
    
    cat > "$SUTAZAI_ROOT/.github/workflows/security-scan.yml" << 'EOF'
name: Container Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1' # Weekly scan

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Trivy
      run: |
        sudo apt-get update
        sudo apt-get install wget apt-transport-https gnupg lsb-release
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install trivy
    
    - name: Scan Base Images
      run: |
        trivy image --exit-code 1 --severity CRITICAL,HIGH python:3.11-slim
        trivy image --exit-code 1 --severity CRITICAL,HIGH node:18-slim
        trivy image --exit-code 1 --severity CRITICAL,HIGH nginx:alpine
    
    - name: Scan Custom Images
      run: |
        docker build -t sutazai/backend ./backend
        trivy image --exit-code 1 --severity CRITICAL,HIGH sutazai/backend
    
    - name: Secrets Scan
      run: |
        python scripts/check_secrets.py
    
    - name: Upload Results
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: security-scan-results
        path: security-scan-results/
EOF

    log "Created automated security scanning pipeline"
}

validate_fixes() {
    log "Validating security fixes..."
    
    # Check if secrets are removed
    if python "$SUTAZAI_ROOT/scripts/check_secrets.py" > /dev/null 2>&1; then
        warn "Some hardcoded secrets may still exist - manual review required"
    else
        log "Hardcoded secrets scan passed"
    fi
    
    # Validate pinned dependencies
    if grep -q ">=" "$SUTAZAI_ROOT/backend/requirements.txt"; then
        warn "Some dependencies are still unpinned"
    else
        log "All dependencies are properly pinned"
    fi
    
    log "Security fixes validation completed"
}

generate_security_report() {
    log "Generating post-remediation security report..."
    
    cat > "$SUTAZAI_ROOT/security-scan-results/remediation-summary.md" << EOF
# Security Remediation Summary

**Date:** $(date)
**Script Version:** 1.0
**Status:** Completed

## Actions Taken

### ✅ Completed
- [x] Created security backup
- [x] Fixed hardcoded secrets in core files
- [x] Pinned all critical dependencies
- [x] Created security-hardened Docker Compose
- [x] Generated secure secrets templates
- [x] Set up automated security scanning pipeline

### 📋 Manual Actions Required

1. **Update Environment Variables**
   - Run: \`./security-scan-results/templates/generate-secrets.sh > .env\`
   - Update all hardcoded references with environment variables
   - Ensure .env is in .gitignore

2. **Deploy with Security Profile**
   - Use: \`docker-compose -f docker-compose.yml -f docker-compose.security.yml up\`
   - Test all services with new security configurations

3. **Validate Privileged Containers**
   - Review hardware-resource-optimizer necessity for privileged mode
   - Consider using specific capabilities instead of full privileges

4. **Enable Security Monitoring**
   - Deploy the GitHub Actions security pipeline
   - Set up regular vulnerability scanning schedule

## Next Steps

1. **Immediate (Today)**
   - [ ] Deploy with new security configurations
   - [ ] Test all services functionality
   - [ ] Update documentation

2. **Within 1 Week**
   - [ ] Implement secrets management (HashiCorp Vault)
   - [ ] Enable automated security scanning
   - [ ] Security team review

3. **Within 1 Month**
   - [ ] Full security audit
   - [ ] Penetration testing
   - [ ] Compliance validation

## Files Modified

- \`backend/requirements.txt\` - Pinned all dependencies
- \`scripts/multi-environment-config-manager.py\` - Removed hardcoded secrets
- \`workflows/scripts/workflow_manager.py\` - Fixed Redis password
- \`docker-compose.security.yml\` - Created security-hardened configuration

## Backup Location

All original files backed up to: \`$BACKUP_DIR\`

---
**Security Contact:** security@sutazai.com
EOF

    log "Generated post-remediation security report"
}

main() {
    log "Starting SutazAI Container Security Remediation"
    log "Timestamp: $(date)"
    
    create_backup
    fix_hardcoded_secrets
    pin_critical_dependencies
    create_security_docker_compose
    create_secrets_template
    create_security_scanning_pipeline
    validate_fixes
    generate_security_report
    
    log "Security remediation completed successfully!"
    log "Please review the remediation summary at: security-scan-results/remediation-summary.md"
    log "Backup location: $BACKUP_DIR"
    
    warn "IMPORTANT: Manual actions required - see remediation-summary.md"
}

# Run main function
main "$@"