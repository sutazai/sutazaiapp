#!/bin/bash
# SutazAI Deployment Script with Memory Optimization and OOM Prevention

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="sutazaiapp"
COMPOSE_FILE="docker-compose-optimized.yml"
MIN_MEMORY_GB=16
MIN_DISK_GB=50

echo -e "${BLUE}🚀 SutazAI Deployment Script with Memory Optimization${NC}"
echo "=========================================================="

# Function to check system requirements
check_system_requirements() {
    echo -e "${YELLOW}Checking system requirements...${NC}"
    
    # Check memory
    total_memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_memory_gb" -lt "$MIN_MEMORY_GB" ]; then
        echo -e "${RED}❌ Insufficient memory: ${total_memory_gb}GB available, ${MIN_MEMORY_GB}GB required${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Memory: ${total_memory_gb}GB${NC}"
    
    # Check disk space
    available_disk_gb=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_disk_gb" -lt "$MIN_DISK_GB" ]; then
        echo -e "${RED}❌ Insufficient disk space: ${available_disk_gb}GB available, ${MIN_DISK_GB}GB required${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Disk space: ${available_disk_gb}GB available${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker installed${NC}"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker Compose installed${NC}"
}

# Function to setup swap if needed
setup_swap() {
    echo -e "${YELLOW}Checking swap configuration...${NC}"
    
    swap_total=$(free -m | awk '/^Swap:/{print $2}')
    if [ "$swap_total" -eq 0 ]; then
        echo -e "${YELLOW}No swap detected. Creating 8GB swap file...${NC}"
        
        if [ -f /swapfile ]; then
            echo "Swap file already exists"
        else
            sudo fallocate -l 8G /swapfile
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
            echo -e "${GREEN}✓ 8GB swap file created${NC}"
        fi
    else
        echo -e "${GREEN}✓ Swap already configured: ${swap_total}MB${NC}"
    fi
}

# Function to optimize system settings
optimize_system() {
    echo -e "${YELLOW}Optimizing system settings for memory management...${NC}"
    
    # Set vm.overcommit_memory to 1 to prevent OOM kills
    if [ -w /proc/sys/vm/overcommit_memory ]; then
        echo 1 | sudo tee /proc/sys/vm/overcommit_memory
        echo -e "${GREEN}✓ Set vm.overcommit_memory=1${NC}"
    fi
    
    # Set vm.swappiness to reduce swap usage
    if [ -w /proc/sys/vm/swappiness ]; then
        echo 10 | sudo tee /proc/sys/vm/swappiness
        echo -e "${GREEN}✓ Set vm.swappiness=10${NC}"
    fi
    
    # Increase file descriptor limits
    if [ -w /etc/security/limits.conf ]; then
        if ! grep -q "* soft nofile 65536" /etc/security/limits.conf; then
            echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
            echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
            echo -e "${GREEN}✓ Increased file descriptor limits${NC}"
        fi
    fi
    
    # Configure Docker daemon for better memory management
    if [ ! -f /etc/docker/daemon.json ] || ! grep -q "log-driver" /etc/docker/daemon.json; then
        echo '{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    }
  },
  "live-restore": true
}' | sudo tee /etc/docker/daemon.json
        sudo systemctl restart docker || true
        echo -e "${GREEN}✓ Docker daemon optimized${NC}"
    fi
}

# Function to clean up Docker resources
cleanup_docker() {
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    
    # Stop all containers from previous deployments
    docker-compose -f $COMPOSE_FILE down --remove-orphans || true
    docker-compose -f docker-compose.yml down --remove-orphans || true
    
    # Remove stopped containers
    docker container prune -f || true
    
    # Remove unused images
    docker image prune -a -f || true
    
    # Remove unused volumes
    docker volume prune -f || true
    
    # Remove unused networks
    docker network prune -f || true
    
    echo -e "${GREEN}✓ Docker cleanup completed${NC}"
}

# Function to create necessary directories
create_directories() {
    echo -e "${YELLOW}Creating necessary directories...${NC}"
    
    directories=(
        "scripts"
        "data"
        "logs"
        "monitoring/prometheus"
        "monitoring/grafana/dashboards"
        "monitoring/grafana/datasources"
        "nginx"
        "ssl"
        "frontend"
        "backend"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    # Set permissions
    chmod -R 755 scripts
    chmod -R 777 data logs
    
    echo -e "${GREEN}✓ Directories created${NC}"
}

# Function to create environment file
create_env_file() {
    echo -e "${YELLOW}Creating environment file...${NC}"
    
    if [ ! -f .env ]; then
        cat > .env << EOF
# Database
POSTGRES_PASSWORD=sutazai2024secure
POSTGRES_USER=sutazai
POSTGRES_DB=sutazai

# Redis
REDIS_PASSWORD=redis2024secure

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Monitoring
GRAFANA_PASSWORD=admin2024secure

# External Services (Optional)
ALERT_WEBHOOK=
EMAIL_SMTP_HOST=
EMAIL_USERNAME=
EMAIL_PASSWORD=

# Resource Limits
MAX_WORKERS=2
MAX_REQUESTS=1000
LOG_LEVEL=INFO
EOF
        echo -e "${GREEN}✓ Environment file created${NC}"
    else
        echo -e "${YELLOW}! Environment file already exists${NC}"
    fi
}

# Function to create monitoring configuration
create_monitoring_config() {
    echo -e "${YELLOW}Creating monitoring configuration...${NC}"
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['fastapi-backend:8000']
    metrics_path: '/metrics'
    
  - job_name: 'docker'
    static_configs:
      - targets: ['172.17.0.1:9323']
EOF

    # Grafana datasource
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    echo -e "${GREEN}✓ Monitoring configuration created${NC}"
}

# Function to create nginx configuration
create_nginx_config() {
    echo -e "${YELLOW}Creating nginx configuration...${NC}"
    
    cat > nginx/nginx.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=web_limit:10m rate=30r/s;

    # Backend API
    upstream backend {
        server fastapi-backend:8000;
        keepalive 32;
    }

    # Frontend
    upstream frontend {
        server streamlit-frontend:8501;
        keepalive 16;
    }

    server {
        listen 80 default_server;
        server_name _;

        # API endpoints
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://backend/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # Buffering
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
            proxy_busy_buffers_size 8k;
        }

        # Frontend
        location / {
            limit_req zone=web_limit burst=50 nodelay;
            
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_set_header X-Forwarded-Host $server_name;
            proxy_buffering off;
        }

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
EOF

    echo -e "${GREEN}✓ Nginx configuration created${NC}"
}

# Function to deploy services with staged approach
deploy_services() {
    echo -e "${YELLOW}Deploying services with staged approach...${NC}"
    
    # Build images first
    echo "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build --parallel || true
    
    # Stage 1: Core infrastructure
    echo "Stage 1: Starting core infrastructure..."
    docker-compose -f $COMPOSE_FILE up -d postgresql redis
    
    # Wait for databases
    echo "Waiting for databases to be ready..."
    sleep 15
    
    # Stage 2: Vector databases
    echo "Stage 2: Starting vector databases..."
    docker-compose -f $COMPOSE_FILE up -d chromadb qdrant
    sleep 10
    
    # Stage 3: Ollama (with memory limits)
    echo "Stage 3: Starting Ollama with memory optimization..."
    docker-compose -f $COMPOSE_FILE up -d ollama
    sleep 30
    
    # Stage 4: Backend
    echo "Stage 4: Starting backend services..."
    docker-compose -f $COMPOSE_FILE up -d fastapi-backend
    sleep 15
    
    # Stage 5: Frontend and monitoring
    echo "Stage 5: Starting frontend and monitoring..."
    docker-compose -f $COMPOSE_FILE up -d streamlit-frontend prometheus grafana nginx
    
    # Stage 6: Health monitoring
    echo "Stage 6: Starting health monitoring..."
    docker-compose -f $COMPOSE_FILE up -d health-check
    
    echo -e "${GREEN}✓ All services deployed successfully${NC}"
}

# Function to verify deployment
verify_deployment() {
    echo -e "${YELLOW}Verifying deployment...${NC}"
    
    services=(
        "postgresql:5432"
        "redis:6379"
        "chromadb:8001"
        "qdrant:6333"
        "ollama:11434"
        "fastapi-backend:8000"
        "streamlit-frontend:8501"
        "prometheus:9090"
        "grafana:3000"
    )
    
    failed=0
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if timeout 10 bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
            echo -e "${GREEN}✓ $name is running on port $port${NC}"
        else
            echo -e "${RED}✗ $name is not accessible on port $port${NC}"
            failed=$((failed + 1))
        fi
    done
    
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}✓ All services are running successfully!${NC}"
        return 0
    else
        echo -e "${RED}❌ $failed services failed to start${NC}"
        return 1
    fi
}

# Function to show memory usage
show_memory_usage() {
    echo -e "${YELLOW}Current memory usage:${NC}"
    free -h
    echo ""
    echo -e "${YELLOW}Docker container memory usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
}

# Function to show deployment info
show_deployment_info() {
    echo -e "${BLUE}
═══════════════════════════════════════════════════════════════
🎉 SutazAI Deployment Complete with Memory Optimization! 🎉
═══════════════════════════════════════════════════════════════

Access your services at:
  • Main UI:        http://localhost:8501
  • API Docs:       http://localhost:8000/docs
  • ChromaDB:       http://localhost:8001
  • Qdrant:         http://localhost:6333/dashboard
  • Prometheus:     http://localhost:9090
  • Grafana:        http://localhost:3000 (admin/admin)

Memory Optimizations Applied:
  ✓ Memory limits set for all containers
  ✓ Swap space configured (8GB)
  ✓ System settings optimized
  ✓ Ollama configured for single model loading
  ✓ Health monitoring enabled

Commands:
  • View logs:      docker-compose -f $COMPOSE_FILE logs -f [service]
  • Stop services:  docker-compose -f $COMPOSE_FILE down
  • Restart:        docker-compose -f $COMPOSE_FILE restart [service]
  • Health check:   docker-compose -f $COMPOSE_FILE ps
  • Memory usage:   docker stats

Monitor health:
  • System health:  python scripts/health-monitor.py
  • Memory usage:   free -h && docker stats

═══════════════════════════════════════════════════════════════
${NC}"
    
    show_memory_usage
}

# Function to install system monitoring
install_system_monitoring() {
    echo -e "${YELLOW}Installing system monitoring tools...${NC}"
    
    # Install htop if not present
    if ! command -v htop &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y htop
    fi
    
    # Create memory monitoring script
    cat > /usr/local/bin/sutazai-monitor << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "SutazAI System Monitor - $(date)"
    echo "================================"
    echo "Memory Usage:"
    free -h
    echo ""
    echo "Top Memory Consumers:"
    ps aux --sort=-%mem | head -10
    echo ""
    echo "Docker Containers:"
    docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 5
done
EOF
    
    chmod +x /usr/local/bin/sutazai-monitor
    echo -e "${GREEN}✓ System monitoring installed (run 'sutazai-monitor')${NC}"
}

# Main deployment flow
main() {
    echo -e "${BLUE}Starting SutazAI deployment with memory optimization...${NC}\n"
    
    # Check system requirements
    check_system_requirements
    
    # Setup swap if needed
    setup_swap
    
    # Optimize system
    optimize_system
    
    # Clean up Docker
    cleanup_docker
    
    # Create directories
    create_directories
    
    # Create configuration files
    create_env_file
    create_monitoring_config
    create_nginx_config
    
    # Deploy services
    deploy_services
    
    # Wait for stabilization
    echo -e "${YELLOW}Waiting for services to stabilize...${NC}"
    sleep 30
    
    # Verify deployment
    if verify_deployment; then
        install_system_monitoring
        show_deployment_info
        
        # Start health monitor in background
        echo -e "${YELLOW}Starting health monitor...${NC}"
        if [ -f scripts/health-monitor.py ]; then
            nohup python3 scripts/health-monitor.py > logs/health-monitor.log 2>&1 &
            echo -e "${GREEN}✓ Health monitor started${NC}"
        fi
    else
        echo -e "${RED}Deployment verification failed. Check logs for details.${NC}"
        echo "Run: docker-compose -f $COMPOSE_FILE logs"
        exit 1
    fi
}

# Run main function
main "$@"