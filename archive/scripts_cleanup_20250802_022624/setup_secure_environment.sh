#!/bin/bash
"""
Secure Environment Setup Script
Sets up enterprise-grade security configuration for SutazAI V7
"""

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Environment detection
detect_environment() {
    if [[ "${SUTAZAI_ENV:-}" ]]; then
        echo "${SUTAZAI_ENV}"
    elif [[ "${HOSTNAME}" == *"prod"* ]] || [[ "${HOSTNAME}" == *"production"* ]]; then
        echo "production"
    elif [[ "${HOSTNAME}" == *"stage"* ]] || [[ "${HOSTNAME}" == *"staging"* ]]; then
        echo "staging"
    else
        echo "development"
    fi
}

# Generate secure random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 $length | tr -d "=+/" | cut -c1-$length
}

# Setup secure directories
setup_secure_directories() {
    log_info "Setting up secure directories..."
    
    # Create secure config directory
    sudo mkdir -p /opt/sutazaiapp/config/secure
    sudo chmod 700 /opt/sutazaiapp/config/secure
    sudo chown $(whoami):$(whoami) /opt/sutazaiapp/config/secure
    
    # Create SSL directory
    sudo mkdir -p /opt/sutazaiapp/ssl
    sudo chmod 700 /opt/sutazaiapp/ssl
    sudo chown $(whoami):$(whoami) /opt/sutazaiapp/ssl
    
    log_success "Secure directories created"
}

# Generate SSL certificates
generate_ssl_certificates() {
    log_info "Generating SSL certificates..."
    
    local ssl_dir="/opt/sutazaiapp/ssl"
    local domain=${1:-"sutazai.local"}
    
    # Generate private key
    openssl genrsa -out "$ssl_dir/key.pem" 2048
    
    # Generate certificate signing request
    openssl req -new -key "$ssl_dir/key.pem" -out "$ssl_dir/csr.pem" \
        -subj "/C=US/ST=State/L=City/O=SutazAI/CN=$domain"
    
    # Generate self-signed certificate
    openssl x509 -req -days 365 -in "$ssl_dir/csr.pem" \
        -signkey "$ssl_dir/key.pem" -out "$ssl_dir/cert.pem"
    
    # Set secure permissions
    chmod 600 "$ssl_dir"/*.pem
    
    # Clean up CSR
    rm "$ssl_dir/csr.pem"
    
    log_success "SSL certificates generated"
}

# Setup environment variables
setup_environment_variables() {
    local env_name=$(detect_environment)
    log_info "Setting up environment variables for: $env_name"
    
    # Create environment file
    local env_file="/opt/sutazaiapp/.env.${env_name}"
    
    cat > "$env_file" << EOF
# SutazAI V7 Secure Environment Configuration
# Generated on: $(date)
# Environment: $env_name

# Application Environment
SUTAZAI_ENV=$env_name
PYTHONPATH=/opt/sutazaiapp

# Security Secrets (Auto-generated)
AUTH_SECRET_KEY=$(generate_password 64)
JWT_SECRET=$(generate_password 64)
ENCRYPTION_KEY=$(generate_password 32)

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=sutazai
POSTGRES_USER=sutazai_app
POSTGRES_PASSWORD=$(generate_password 32)

# Monitoring Configuration
GRAFANA_ADMIN_PASSWORD=$(generate_password 24)
PROMETHEUS_PASSWORD=$(generate_password 24)

# Vector Database
QDRANT_API_KEY=$(generate_password 32)

# Redis Configuration
REDIS_PASSWORD=$(generate_password 24)

# SSL Configuration
SSL_CERT_PATH=/opt/sutazaiapp/ssl/cert.pem
SSL_KEY_PATH=/opt/sutazaiapp/ssl/key.pem

EOF

    # Environment-specific settings
    if [[ "$env_name" == "production" ]]; then
        cat >> "$env_file" << EOF
# Production Settings
DEBUG=false
LOG_LEVEL=INFO
ALLOWED_HOSTS=sutazai.company.com,admin.sutazai.company.com
CORS_ORIGINS=https://sutazai.company.com,https://admin.sutazai.company.com

EOF
    elif [[ "$env_name" == "staging" ]]; then
        cat >> "$env_file" << EOF
# Staging Settings
DEBUG=false
LOG_LEVEL=DEBUG
ALLOWED_HOSTS=staging.sutazai.company.com,staging-admin.sutazai.company.com
CORS_ORIGINS=https://staging.sutazai.company.com,https://staging-admin.sutazai.company.com

EOF
    else
        cat >> "$env_file" << EOF
# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0
CORS_ORIGINS=http://localhost:3000,http://localhost:8501,http://127.0.0.1:3000

EOF
    fi
    
    # Set secure permissions
    chmod 600 "$env_file"
    
    # Create symlink to active environment
    ln -sf ".env.${env_name}" "/opt/sutazaiapp/.env"
    
    log_success "Environment variables configured for $env_name"
}

# Setup firewall rules
setup_firewall() {
    log_info "Configuring firewall rules..."
    
    # Enable UFW if not enabled
    if ! sudo ufw status | grep -q "Status: active"; then
        sudo ufw --force enable
    fi
    
    # Default policies
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # SSH access
    sudo ufw allow ssh
    
    # HTTP and HTTPS
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    
    # SutazAI API
    sudo ufw allow 8000/tcp
    
    # Monitoring (restrict to internal only in production)
    local env_name=$(detect_environment)
    if [[ "$env_name" != "production" ]]; then
        sudo ufw allow 3000/tcp  # Grafana
        sudo ufw allow 9090/tcp  # Prometheus
        sudo ufw allow 5601/tcp  # Kibana
    fi
    
    log_success "Firewall configured"
}

# Setup secure Docker configuration
setup_docker_security() {
    log_info "Configuring Docker security..."
    
    # Create Docker daemon configuration
    sudo mkdir -p /etc/docker
    
    cat | sudo tee /etc/docker/daemon.json > /dev/null << EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "live-restore": true,
    "userland-proxy": false,
    "no-new-privileges": true,
    "seccomp-profile": "/etc/docker/seccomp.json",
    "default-ulimits": {
        "nofile": {
            "hard": 64000,
            "soft": 64000
        }
    }
}
EOF
    
    # Restart Docker if running
    if systemctl is-active --quiet docker; then
        sudo systemctl restart docker
    fi
    
    log_success "Docker security configured"
}

# Initialize secure configuration
initialize_secure_config() {
    log_info "Initializing secure configuration..."
    
    # Source environment variables
    source /opt/sutazaiapp/.env
    
    # Initialize secure config manager (will create encryption keys)
    python3 -c "
import sys
sys.path.append('/opt/sutazaiapp')
from backend.security.secure_config import secure_config
print('Secure configuration initialized')
"
    
    log_success "Secure configuration initialized"
}

# Verify security setup
verify_security_setup() {
    log_info "Verifying security setup..."
    
    local issues=0
    
    # Check environment file
    if [[ ! -f "/opt/sutazaiapp/.env" ]]; then
        log_error "Environment file not found"
        ((issues++))
    fi
    
    # Check SSL certificates
    if [[ ! -f "/opt/sutazaiapp/ssl/cert.pem" ]] || [[ ! -f "/opt/sutazaiapp/ssl/key.pem" ]]; then
        log_error "SSL certificates not found"
        ((issues++))
    fi
    
    # Check secure directory permissions
    if [[ "$(stat -c %a /opt/sutazaiapp/config/secure)" != "700" ]]; then
        log_error "Secure directory permissions incorrect"
        ((issues++))
    fi
    
    # Check firewall status
    if ! sudo ufw status | grep -q "Status: active"; then
        log_error "Firewall not active"
        ((issues++))
    fi
    
    if [[ $issues -eq 0 ]]; then
        log_success "Security setup verification passed"
        return 0
    else
        log_error "Security setup verification failed with $issues issues"
        return 1
    fi
}

# Main execution
main() {
    echo "======================================"
    echo "SutazAI V7 Security Setup Script"
    echo "======================================"
    echo
    
    local env_name=$(detect_environment)
    log_info "Detected environment: $env_name"
    
    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. This is not recommended for security."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Exiting..."
            exit 1
        fi
    fi
    
    # Execute setup steps
    setup_secure_directories
    generate_ssl_certificates
    setup_environment_variables
    setup_firewall
    setup_docker_security
    initialize_secure_config
    
    # Verify setup
    if verify_security_setup; then
        echo
        log_success "🔒 Security setup completed successfully!"
        echo
        echo "Next steps:"
        echo "1. Review the generated .env file: /opt/sutazaiapp/.env"
        echo "2. Backup your encryption keys securely"
        echo "3. Update DNS records for SSL certificates (if needed)"
        echo "4. Start the SutazAI services with: ./scripts/start_sutazai.sh"
        echo
    else
        log_error "Security setup completed with issues. Please review and fix."
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi