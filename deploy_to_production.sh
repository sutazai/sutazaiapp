#!/bin/bash
# SutazAI v8 Production Deployment Script
# Target: 192.168.131.128

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Server details
PROD_SERVER="192.168.131.128"
PROD_USER="ai"  # Correct username for production server
PROD_PASS="1988"
REPO_URL="https://github.com/sutazai/sutazaiapp.git"
DEPLOY_DIR="/opt/sutazaiapp"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Function to execute remote commands
exec_remote() {
    local cmd="$1"
    sshpass -p "$PROD_PASS" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$PROD_USER@$PROD_SERVER" "$cmd"
}

# Function to copy files to remote server
copy_to_remote() {
    local local_path="$1"
    local remote_path="$2"
    sshpass -p "$PROD_PASS" scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r "$local_path" "$PROD_USER@$PROD_SERVER:$remote_path"
}

# Header
echo -e "${BLUE}"
echo "==================================================================="
echo "🚀 SUTAZAI v8 PRODUCTION DEPLOYMENT"
echo "==================================================================="
echo "Target Server: $PROD_SERVER"
echo "Deploy Directory: $DEPLOY_DIR"
echo "Date: $(date)"
echo "==================================================================="
echo -e "${NC}"

# Phase 1: Test Connection
log "🔧 Phase 1: Testing Connection to Production Server"
if exec_remote "echo 'Connection successful'; uname -a; whoami"; then
    log "✅ Successfully connected to production server"
else
    error "❌ Failed to connect to production server"
    exit 1
fi

# Phase 2: System Prerequisites
log "🔧 Phase 2: Installing System Prerequisites"
exec_remote "
    # Update system
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y git docker.io docker-compose python3 python3-pip curl wget
    elif command -v yum >/dev/null 2>&1; then
        yum update -y
        yum install -y git docker docker-compose python3 python3-pip curl wget
    elif command -v dnf >/dev/null 2>&1; then
        dnf update -y
        dnf install -y git docker docker-compose python3 python3-pip curl wget
    fi
    
    # Start Docker service
    systemctl start docker
    systemctl enable docker
    
    # Add user to docker group if not root
    if [ \"\$USER\" != \"root\" ]; then
        usermod -aG docker \$USER
    fi
"

# Phase 3: Clone Repository
log "🔧 Phase 3: Cloning SutazAI v8 Repository"
exec_remote "
    # Remove existing directory if present
    rm -rf $DEPLOY_DIR
    
    # Clone repository
    git clone -b v8 $REPO_URL $DEPLOY_DIR
    cd $DEPLOY_DIR
    
    # Verify deployment
    ls -la
    echo 'Repository cloned successfully'
"

# Phase 4: Environment Configuration
log "🔧 Phase 4: Configuring Production Environment"
exec_remote "
    cd $DEPLOY_DIR
    
    # Create production environment file
    cat > .env.production << EOF
# Production Environment Variables
ENVIRONMENT=production
DATABASE_URL=postgresql://sutazai:sutazai_password@postgres:5432/sutazai
REDIS_URL=redis://redis:6379
QDRANT_URL=http://qdrant:6333
CHROMADB_URL=http://chromadb:8000
FAISS_URL=http://faiss:8088
OLLAMA_URL=http://ollama:11434

# Security
JWT_SECRET_KEY=\$(openssl rand -hex 32)
CORS_ORIGINS=[\"http://localhost:8501\", \"http://$PROD_SERVER:8501\"]

# Production specific
DEBUG=false
LOG_LEVEL=INFO
WORKERS=4
EOF

    # Set proper permissions
    chmod 600 .env.production
    chown -R \$USER:docker $DEPLOY_DIR
"

# Phase 5: Docker Setup
log "🔧 Phase 5: Setting up Docker Environment"
exec_remote "
    cd $DEPLOY_DIR
    
    # Pull essential images first
    docker compose pull postgres redis qdrant chromadb ollama
    
    # Create necessary directories
    mkdir -p logs data/models data/workspace data/vector
    
    # Set proper permissions
    chown -R 1000:1000 data/ logs/
"

# Phase 6: Deploy Core Services
log "🔧 Phase 6: Deploying Core Services"
exec_remote "
    cd $DEPLOY_DIR
    
    # Start infrastructure services first
    docker compose up -d postgres redis qdrant chromadb
    
    # Wait for services to be ready
    sleep 30
    
    # Start model management
    docker compose up -d ollama
    
    # Wait for Ollama to be ready
    sleep 20
    
    # Start main application
    docker compose up -d sutazai-backend sutazai-streamlit
    
    # Wait for main services
    sleep 15
"

# Phase 7: Deploy AI Services
log "🔧 Phase 7: Deploying AI Services"
exec_remote "
    cd $DEPLOY_DIR
    
    # Start AI services in batches to avoid overwhelming the system
    docker compose up -d autogpt localagi tabby
    sleep 10
    
    docker compose up -d langflow dify autogen
    sleep 10
    
    docker compose up -d pytorch tensorflow jax
    sleep 10
    
    docker compose up -d browser-use skyvern documind finrobot
    sleep 10
    
    docker compose up -d gpt-engineer aider bigagi agentzero
    sleep 10
"

# Phase 8: Start Monitoring
log "🔧 Phase 8: Starting Monitoring Services"
exec_remote "
    cd $DEPLOY_DIR
    
    # Start monitoring stack
    docker compose up -d prometheus grafana nginx
    
    # Wait for monitoring to be ready
    sleep 15
"

# Phase 9: System Validation
log "🔧 Phase 9: Validating Production Deployment"
exec_remote "
    cd $DEPLOY_DIR
    
    # Check service health
    docker compose ps
    
    # Test main endpoints
    curl -f http://localhost:8000/health || echo 'Backend health check failed'
    curl -f http://localhost:8501/healthz || echo 'Frontend health check failed'
    
    # Run validation script
    python3 validate_sutazai_v8_complete.py
"

# Phase 10: Final Configuration
log "🔧 Phase 10: Final Production Configuration"
exec_remote "
    cd $DEPLOY_DIR
    
    # Set up log rotation
    cat > /etc/logrotate.d/sutazai << EOF
$DEPLOY_DIR/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF

    # Create systemd service for auto-restart
    cat > /etc/systemd/system/sutazai.service << EOF
[Unit]
Description=SutazAI v8 AGI/ASI System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$DEPLOY_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
ExecReload=/usr/bin/docker compose restart

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable sutazai
    
    echo 'Production deployment completed successfully!'
"

# Final Summary
echo -e "${GREEN}"
echo "==================================================================="
echo "🎉 SUTAZAI v8 PRODUCTION DEPLOYMENT COMPLETE"
echo "==================================================================="
echo -e "${NC}"

echo "📊 Production Server Status:"
echo "   • Server: $PROD_SERVER"
echo "   • Deploy Directory: $DEPLOY_DIR"
echo "   • Services: Running in Docker containers"

echo ""
echo "🌐 Access Points:"
echo "   • Main Interface: http://$PROD_SERVER:8501"
echo "   • API Endpoint: http://$PROD_SERVER:8000"
echo "   • API Documentation: http://$PROD_SERVER:8000/docs"
echo "   • Monitoring: http://$PROD_SERVER:3000"

echo ""
echo "🔧 Management Commands:"
echo "   • Check status: ssh $PROD_USER@$PROD_SERVER 'cd $DEPLOY_DIR && docker compose ps'"
echo "   • View logs: ssh $PROD_USER@$PROD_SERVER 'cd $DEPLOY_DIR && docker compose logs -f'"
echo "   • Restart services: ssh $PROD_USER@$PROD_SERVER 'cd $DEPLOY_DIR && docker compose restart'"

echo ""
log "🎊 SutazAI v8 Production Deployment Complete!"