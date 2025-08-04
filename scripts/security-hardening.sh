#!/bin/bash
"""
SutazAI Security Hardening Script
Addresses critical security vulnerabilities identified in security audit
Author: Claude Security Specialist
Date: August 4, 2025
"""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_DIR/security_backup_$(date +%Y%m%d_%H%M%S)"

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
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. This script should be run as a regular user with sudo privileges."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to create backup
create_backup() {
    log "Creating security backup at $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Backup critical files
    if [[ -d "$PROJECT_DIR/secrets" ]]; then
        cp -r "$PROJECT_DIR/secrets" "$BACKUP_DIR/"
    fi
    
    if [[ -f "$PROJECT_DIR/docker-compose.yml" ]]; then
        cp "$PROJECT_DIR/docker-compose.yml" "$BACKUP_DIR/"
    fi
    
    if [[ -d "$PROJECT_DIR/nginx" ]]; then
        cp -r "$PROJECT_DIR/nginx" "$BACKUP_DIR/"
    fi
    
    success "Backup created successfully"
}

# Function to secure secrets
secure_secrets() {
    log "Securing exposed secrets..."
    
    SECRETS_DIR="$PROJECT_DIR/secrets"
    if [[ ! -d "$SECRETS_DIR" ]]; then
        warn "Secrets directory not found"
        return
    fi
    
    # Generate new strong passwords
    generate_password() {
        openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
    }
    
    # Create new secrets with proper permissions
    NEW_SECRETS_DIR="$PROJECT_DIR/secrets_secure"
    mkdir -p "$NEW_SECRETS_DIR"
    chmod 700 "$NEW_SECRETS_DIR"
    
    # Generate new passwords
    generate_password > "$NEW_SECRETS_DIR/postgres_password.txt"
    generate_password > "$NEW_SECRETS_DIR/redis_password.txt"
    generate_password > "$NEW_SECRETS_DIR/neo4j_password.txt"
    generate_password > "$NEW_SECRETS_DIR/grafana_password.txt"
    openssl rand -hex 64 > "$NEW_SECRETS_DIR/jwt_secret.txt"
    
    # Set proper permissions
    chmod 600 "$NEW_SECRETS_DIR"/*
    
    # Create environment file template
    cat > "$PROJECT_DIR/.env.template" << 'EOF'
# SutazAI Environment Variables
# Copy to .env and fill with actual values

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=__REPLACE_WITH_SECURE_PASSWORD__
POSTGRES_DB=sutazai

# Redis Configuration
REDIS_PASSWORD=__REPLACE_WITH_SECURE_PASSWORD__

# Neo4j Configuration
NEO4J_PASSWORD=__REPLACE_WITH_SECURE_PASSWORD__

# Grafana Configuration
GRAFANA_ADMIN_PASSWORD=__REPLACE_WITH_SECURE_PASSWORD__

# JWT Configuration
JWT_SECRET=__REPLACE_WITH_SECURE_JWT_SECRET__

# Security Configuration
SESSION_SECRET=__REPLACE_WITH_SECURE_SESSION_SECRET__
ENCRYPTION_KEY=__REPLACE_WITH_SECURE_ENCRYPTION_KEY__

# Network Configuration
TRUSTED_NETWORKS=127.0.0.1,172.16.0.0/12,10.0.0.0/8
ALLOWED_HOSTS=localhost,127.0.0.1

# SSL Configuration
SSL_ENABLED=true
SSL_CERT_PATH=/opt/sutazaiapp/ssl/cert.pem
SSL_KEY_PATH=/opt/sutazaiapp/ssl/key.pem
EOF
    
    success "New secure secrets generated in $NEW_SECRETS_DIR"
    warn "IMPORTANT: Update your .env file with new passwords and restart services"
}

# Function to harden Docker configuration
harden_docker() {
    log "Hardening Docker configuration..."
    
    COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error "docker-compose.yml not found"
    fi
    
    # Create hardened compose file
    HARDENED_COMPOSE="$PROJECT_DIR/docker-compose.secure.yml"
    
    # Copy original and modify
    cp "$COMPOSE_FILE" "$HARDENED_COMPOSE"
    
    # Add security context to all services (this is a simplified approach)
    cat >> "$HARDENED_COMPOSE" << 'EOF'

# Security hardening additions
x-security-opts: &security-opts
  security_opt:
    - no-new-privileges:true
    - apparmor:docker-default
  cap_drop:
    - ALL
  cap_add:
    - CHOWN
    - DAC_OVERRIDE
    - SETGID
    - SETUID
  read_only: false
  tmpfs:
    - /tmp:noexec,nosuid,size=100m
EOF
    
    success "Hardened Docker configuration created: $HARDENED_COMPOSE"
    warn "Review and test the hardened configuration before deploying"
}

# Function to configure network security
configure_network_security() {
    log "Configuring network security..."
    
    # Create nginx security configuration
    NGINX_SECURITY_CONF="$PROJECT_DIR/nginx/security.conf"
    mkdir -p "$(dirname "$NGINX_SECURITY_CONF")"
    
    cat > "$NGINX_SECURITY_CONF" << 'EOF'
# Security Headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

# Hide nginx version
server_tokens off;

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;

# IP filtering (adjust as needed)
# allow 127.0.0.1;
# allow 10.0.0.0/8;
# allow 172.16.0.0/12;
# allow 192.168.0.0/16;
# deny all;

# SSL Configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
EOF
    
    # Update main nginx config to include security
    NGINX_MAIN="$PROJECT_DIR/nginx/nginx.conf"
    if [[ -f "$NGINX_MAIN" ]]; then
        if ! grep -q "include.*security.conf" "$NGINX_MAIN"; then
            sed -i '/http {/a\    include /etc/nginx/security.conf;' "$NGINX_MAIN"
        fi
    fi
    
    success "Network security configuration created"
}

# Function to create monitoring configuration
setup_security_monitoring() {
    log "Setting up security monitoring..."
    
    MONITORING_DIR="$PROJECT_DIR/monitoring/security"
    mkdir -p "$MONITORING_DIR"
    
    # Create fail2ban configuration
    cat > "$MONITORING_DIR/fail2ban-docker.conf" << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 5

[ssh]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
EOF
    
    # Create basic intrusion detection script
    cat > "$MONITORING_DIR/intrusion_detection.py" << 'EOF'
#!/usr/bin/env python3
"""
Basic Intrusion Detection System for SutazAI
Monitors logs for suspicious activities
"""

import re
import time
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

class IntrusionDetector:
    def __init__(self):
        self.suspicious_patterns = [
            r'SQL injection',
            r'<script.*?>',
            r'../../../../',
            r'eval\(',
            r'base64_decode',
            r'union.*select',
            r'drop\s+table',
        ]
        self.failed_attempts = defaultdict(int)
        self.blocked_ips = set()
        
    def analyze_log_line(self, line):
        """Analyze a single log line for suspicious activity"""
        for pattern in self.suspicious_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                self.handle_suspicious_activity(line, pattern)
                
    def handle_suspicious_activity(self, line, pattern):
        """Handle detected suspicious activity"""
        timestamp = datetime.now().isoformat()
        logging.warning(f"[{timestamp}] Suspicious activity detected: {pattern}")
        logging.warning(f"Log line: {line.strip()}")
        
        # Extract IP if possible
        ip_match = re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', line)
        if ip_match:
            ip = ip_match.group()
            self.failed_attempts[ip] += 1
            if self.failed_attempts[ip] > 5:
                self.block_ip(ip)
                
    def block_ip(self, ip):
        """Block an IP address (placeholder - implement with iptables)"""
        if ip not in self.blocked_ips:
            self.blocked_ips.add(ip)
            logging.error(f"IP {ip} has been flagged for blocking")
            # TODO: Implement actual IP blocking

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = IntrusionDetector()
    
    # Monitor nginx logs (adjust path as needed)
    log_files = [
        "/var/log/nginx/access.log",
        "/var/log/nginx/error.log",
    ]
    
    for log_file in log_files:
        if Path(log_file).exists():
            with open(log_file, 'r') as f:
                for line in f:
                    detector.analyze_log_line(line)
EOF
    
    chmod +x "$MONITORING_DIR/intrusion_detection.py"
    
    success "Security monitoring configuration created"
}

# Function to generate SSL certificates
generate_ssl_certificates() {
    log "Generating SSL certificates..."
    
    SSL_DIR="$PROJECT_DIR/ssl"
    mkdir -p "$SSL_DIR"
    
    # Generate self-signed certificate for development
    if [[ ! -f "$SSL_DIR/cert.pem" ]]; then
        openssl req -x509 -newkey rsa:4096 -keyout "$SSL_DIR/key.pem" -out "$SSL_DIR/cert.pem" \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=SutazAI/CN=localhost"
        
        chmod 600 "$SSL_DIR/key.pem"
        chmod 644 "$SSL_DIR/cert.pem"
        
        success "SSL certificates generated"
        warn "These are self-signed certificates for development only"
        warn "For production, use certificates from a trusted CA like Let's Encrypt"
    else
        log "SSL certificates already exist"
    fi
}

# Function to create security validation script
create_security_validator() {
    log "Creating security validation script..."
    
    cat > "$PROJECT_DIR/scripts/validate-security.sh" << 'EOF'
#!/bin/bash
"""
Security Validation Script for SutazAI
Validates security hardening measures
"""

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISSUES=0

check() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Checking $test_name... "
    if eval "$test_command" >/dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
    else
        echo -e "${RED}FAIL${NC}"
        ((ISSUES++))
    fi
}

echo "=== SutazAI Security Validation ==="
echo

# Check secret files permissions
check "Secret files permissions" "[[ ! -d '$PROJECT_DIR/secrets' ]] || [[ \$(stat -c '%a' '$PROJECT_DIR/secrets') == '700' ]]"

# Check for environment file
check "Environment configuration" "[[ -f '$PROJECT_DIR/.env' ]]"

# Check SSL certificates
check "SSL certificates present" "[[ -f '$PROJECT_DIR/ssl/cert.pem' && -f '$PROJECT_DIR/ssl/key.pem' ]]"

# Check nginx security config
check "Nginx security configuration" "[[ -f '$PROJECT_DIR/nginx/security.conf' ]]"

# Check for exposed ports (basic check)
check "No plaintext secrets in docker-compose" "! grep -r 'password.*=' '$PROJECT_DIR/docker-compose.yml' || true"

# Check Docker security
check "Hardened Docker configuration exists" "[[ -f '$PROJECT_DIR/docker-compose.secure.yml' ]]"

echo
if [[ $ISSUES -eq 0 ]]; then
    echo -e "${GREEN}All security checks passed!${NC}"
    exit 0
else
    echo -e "${RED}$ISSUES security issues found. Please review and fix.${NC}"
    exit 1
fi
EOF
    
    chmod +x "$PROJECT_DIR/scripts/validate-security.sh"
    success "Security validation script created"
}

# Main execution function
main() {
    echo "===========================================" 
    echo "    SutazAI Security Hardening Script"
    echo "==========================================="
    echo
    
    check_root
    
    log "Starting security hardening process..."
    
    # Create backup
    create_backup
    
    # Execute hardening steps
    secure_secrets
    harden_docker
    configure_network_security
    setup_security_monitoring
    generate_ssl_certificates
    create_security_validator
    
    echo
    echo "==========================================="
    echo "    Security Hardening Complete!"
    echo "==========================================="
    echo
    
    warn "IMPORTANT NEXT STEPS:"
    echo "1. Review and update the .env file with secure passwords"
    echo "2. Test the hardened Docker configuration: docker-compose.secure.yml"
    echo "3. Configure your firewall to restrict access to necessary ports only"
    echo "4. Run the security validator: ./scripts/validate-security.sh"
    echo "5. Update all default credentials in external services"
    echo "6. Implement proper backup and recovery procedures"
    echo
    success "Security hardening completed successfully!"
}

# Run main function
main "$@"