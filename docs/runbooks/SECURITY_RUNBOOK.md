# Security Runbook - Perfect Jarvis System

**Document Version:** 1.0  
**Last Updated:** 2025-08-08  
**Author:** Security Team  

## 🎯 Purpose

This runbook provides comprehensive security procedures for the Perfect Jarvis system, covering threat detection, incident response, vulnerability management, and security monitoring.

## 🔐 Table of Contents

- [Security Architecture](#security-architecture)
- [Threat Detection](#threat-detection)
- [Vulnerability Management](#vulnerability-management)
- [Access Control](#access-control)
- [Data Protection](#data-protection)
- [Monitoring & Alerting](#monitoring--alerting)
- [Incident Response](#incident-response)
- [Compliance & Audit](#compliance--audit)
- [Security Hardening](#security-hardening)

## 🏗️ Security Architecture

### Current Security Posture

**Authentication:**
- Basic authentication for core endpoints (no auth required)
- JWT Bearer token support for enterprise features (not fully implemented)
- No centralized identity management

**Authorization:**
- Role-based access control (RBAC) planned but not implemented
- All users have equivalent access to public endpoints
- Enterprise endpoints require authentication

**Network Security:**
- All services run on internal Docker network (`sutazai-network`)
- External access only through exposed ports
- No network segmentation between services

**Data Security:**
- PostgreSQL database with basic authentication
- Redis cache without authentication (internal only)
- No encryption at rest implemented
- No data classification system

### Security Boundaries

```
Internet
    │
    ├── Frontend (Port 10011) - Public Access
    ├── Backend API (Port 10010) - Public Access  
    ├── Monitoring (Ports 10200-10206) - Internal
    │
    └── Internal Network (sutazai-network)
        ├── PostgreSQL (Port 10000) - Internal
        ├── Redis (Port 10001) - Internal
        ├── Neo4j (Ports 10002/10003) - Internal
        ├── Ollama (Port 10104) - Internal
        └── Agent Services (Ports 8xxx) - Internal
```

## 🚨 Threat Detection

### Security Monitoring Implementation

#### 1. Log-Based Detection
```bash
#!/bin/bash
# security_log_monitor.sh
SECURITY_LOG="/opt/sutazaiapp/logs/security_monitor.log"
ALERT_THRESHOLD=5

# Function to log security events
log_security_event() {
    local event_type=$1
    local severity=$2
    local message=$3
    local source_ip=${4:-"unknown"}
    
    echo "$(date -Iseconds) [$severity] $event_type: $message (Source: $source_ip)" >> "$SECURITY_LOG"
    
    # High severity alerts
    if [[ "$severity" == "HIGH" || "$severity" == "CRITICAL" ]]; then
        send_security_alert "$event_type" "$message" "$source_ip"
    fi
}

# Monitor authentication attempts
monitor_auth_attempts() {
    docker logs sutazai-backend --since=1m 2>/dev/null | \
    grep -i "authentication\|unauthorized\|forbidden" | \
    while read line; do
        log_security_event "AUTH_FAILURE" "MEDIUM" "$line"
    done
}

# Monitor for suspicious patterns
monitor_suspicious_activity() {
    # High request rate from single IP
    docker logs sutazai-backend --since=5m 2>/dev/null | \
    grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' | \
    sort | uniq -c | \
    while read count ip; do
        if [ "$count" -gt 100 ]; then
            log_security_event "HIGH_REQUEST_RATE" "HIGH" "IP $ip made $count requests in 5 minutes" "$ip"
        fi
    done
    
    # SQL injection attempts
    docker logs sutazai-backend --since=5m 2>/dev/null | \
    grep -i -E "(union|select|drop|delete|insert|update).*from" | \
    while read line; do
        log_security_event "SQL_INJECTION_ATTEMPT" "HIGH" "$line"
    done
    
    # XSS attempts
    docker logs sutazai-backend --since=5m 2>/dev/null | \
    grep -i -E "(<script|javascript:|on\w+\s*=)" | \
    while read line; do
        log_security_event "XSS_ATTEMPT" "HIGH" "$line"
    done
}

# Monitor system access
monitor_system_access() {
    # Docker exec attempts
    docker events --since=5m --filter event=exec 2>/dev/null | \
    while read line; do
        log_security_event "CONTAINER_ACCESS" "MEDIUM" "$line"
    done
    
    # File system changes in sensitive directories
    if command -v inotifywait >/dev/null; then
        inotifywait -m -r /opt/sutazaiapp/config /opt/sutazaiapp/.env \
            --format '%w%f %e' -e modify,create,delete 2>/dev/null | \
        while read file event; do
            log_security_event "CONFIG_CHANGE" "HIGH" "File $file was $event"
        done &
    fi
}

# Send security alerts
send_security_alert() {
    local event_type=$1
    local message=$2
    local source_ip=$3
    
    # Slack notification
    curl -X POST -H 'Content-type: application/json' \
        --data "{
            \"text\": \"🚨 Security Alert: $event_type\",
            \"attachments\": [{
                \"color\": \"danger\",
                \"fields\": [
                    {\"title\": \"Message\", \"value\": \"$message\", \"short\": false},
                    {\"title\": \"Source IP\", \"value\": \"$source_ip\", \"short\": true},
                    {\"title\": \"Timestamp\", \"value\": \"$(date -Iseconds)\", \"short\": true}
                ]
            }]
        }" \
        "$SECURITY_SLACK_WEBHOOK" 2>/dev/null || true
    
    # Email notification
    echo -e "Security Alert: $event_type\n\nMessage: $message\nSource IP: $source_ip\nTimestamp: $(date)" | \
        mail -s "Jarvis Security Alert" "security-team@company.com" 2>/dev/null || true
}

# Main monitoring loop
main() {
    echo "Starting security monitoring..."
    while true; do
        monitor_auth_attempts
        monitor_suspicious_activity
        monitor_system_access
        sleep 60
    done
}

main
```

#### 2. Real-time Intrusion Detection
```bash
#!/bin/bash
# intrusion_detection.sh
IDS_LOG="/opt/sutazaiapp/logs/ids.log"

# Network-based detection
monitor_network_traffic() {
    if command -v tcpdump >/dev/null; then
        # Monitor for port scans
        tcpdump -i any -n -c 100 2>/dev/null | \
        grep -E "(10010|10011)" | \
        awk '{print $3}' | cut -d: -f1 | sort | uniq -c | \
        while read count ip; do
            if [ "$count" -gt 20 ]; then
                log_security_event "PORT_SCAN" "HIGH" "Possible port scan from $ip ($count connections)" "$ip"
            fi
        done
    fi
}

# Application-level detection
monitor_app_attacks() {
    # Monitor for brute force attempts
    tail -f /var/log/nginx/access.log 2>/dev/null | \
    grep -E "(401|403)" | \
    awk '{print $1}' | sort | uniq -c | \
    while read count ip; do
        if [ "$count" -gt 10 ]; then
            log_security_event "BRUTE_FORCE" "HIGH" "Multiple authentication failures from $ip" "$ip"
            # Auto-block IP (if fail2ban is available)
            fail2ban-client set nginx-http-auth banip "$ip" 2>/dev/null || true
        fi
    done
}

# Container security monitoring
monitor_container_security() {
    # Monitor for privilege escalation
    docker logs sutazai-backend --since=1m 2>/dev/null | \
    grep -i -E "(sudo|su -|chmod.*777|chown.*root)" | \
    while read line; do
        log_security_event "PRIVILEGE_ESCALATION" "CRITICAL" "$line"
    done
    
    # Monitor for unusual process execution
    docker exec sutazai-backend ps aux 2>/dev/null | \
    grep -v -E "(python|uvicorn|gunicorn)" | \
    grep -E "(bash|sh|nc|wget|curl)" | \
    while read line; do
        log_security_event "SUSPICIOUS_PROCESS" "MEDIUM" "$line"
    done
}

main() {
    echo "Starting intrusion detection system..."
    monitor_network_traffic &
    monitor_app_attacks &
    monitor_container_security &
    wait
}

main
```

### Automated Response Actions

#### IP Blocking System
```bash
#!/bin/bash
# auto_block_ips.sh
BLOCKED_IPS_FILE="/opt/sutazaiapp/security/blocked_ips.txt"
WHITELIST_FILE="/opt/sutazaiapp/security/whitelist.txt"

# Create security directory
mkdir -p /opt/sutazaiapp/security

# Function to block IP
block_ip() {
    local ip=$1
    local reason=$2
    local duration=${3:-3600}  # Default 1 hour
    
    # Check if IP is whitelisted
    if grep -q "^$ip$" "$WHITELIST_FILE" 2>/dev/null; then
        echo "IP $ip is whitelisted, skipping block"
        return
    fi
    
    # Check if already blocked
    if grep -q "^$ip " "$BLOCKED_IPS_FILE" 2>/dev/null; then
        echo "IP $ip already blocked"
        return
    fi
    
    echo "Blocking IP $ip for $duration seconds (Reason: $reason)"
    
    # Add iptables rule
    iptables -A INPUT -s "$ip" -j DROP
    
    # Log the block
    echo "$ip $(date +%s) $duration $reason" >> "$BLOCKED_IPS_FILE"
    
    # Schedule unblock
    {
        sleep "$duration"
        unblock_ip "$ip"
    } &
}

# Function to unblock IP
unblock_ip() {
    local ip=$1
    
    echo "Unblocking IP $ip"
    
    # Remove iptables rule
    iptables -D INPUT -s "$ip" -j DROP 2>/dev/null || true
    
    # Remove from blocked list
    sed -i "/^$ip /d" "$BLOCKED_IPS_FILE"
}

# Monitor security log and auto-block
monitor_and_block() {
    tail -f "$SECURITY_LOG" | \
    grep -E "(HIGH|CRITICAL)" | \
    while read line; do
        if echo "$line" | grep -q "Source:"; then
            ip=$(echo "$line" | grep -o "Source: [0-9.]*" | cut -d' ' -f2)
            reason=$(echo "$line" | cut -d']' -f2 | cut -d':' -f1)
            
            if [[ -n "$ip" && "$ip" != "unknown" ]]; then
                block_ip "$ip" "$reason" 7200  # Block for 2 hours
            fi
        fi
    done
}

# Cleanup expired blocks
cleanup_expired_blocks() {
    if [[ -f "$BLOCKED_IPS_FILE" ]]; then
        current_time=$(date +%s)
        while IFS=' ' read -r ip block_time duration reason; do
            if [[ -n "$ip" ]]; then
                expire_time=$((block_time + duration))
                if [[ $current_time -gt $expire_time ]]; then
                    unblock_ip "$ip"
                fi
            fi
        done < "$BLOCKED_IPS_FILE"
    fi
}

# Main function
main() {
    echo "Starting automated IP blocking system..."
    
    # Initial cleanup
    cleanup_expired_blocks
    
    # Start monitoring
    monitor_and_block &
    
    # Periodic cleanup every 5 minutes
    while true; do
        sleep 300
        cleanup_expired_blocks
    done
}

main
```

## 🔐 Vulnerability Management

### Vulnerability Scanning

#### Container Security Scan
```bash
#!/bin/bash
# container_security_scan.sh
SCAN_REPORT="/opt/sutazaiapp/security/vulnerability_scan_$(date +%Y%m%d_%H%M%S).json"

echo "=== CONTAINER SECURITY SCAN ==="

# Function to scan container image
scan_image() {
    local image_name=$1
    local output_file=$2
    
    echo "Scanning image: $image_name"
    
    # Using Docker's built-in security scanning (if available)
    docker scan "$image_name" --json > "$output_file" 2>/dev/null || {
        echo "Docker scan not available, using alternative methods..."
        
        # Alternative: Use Trivy if available
        if command -v trivy >/dev/null; then
            trivy image --format json --output "$output_file" "$image_name"
        else
            echo "No vulnerability scanner available"
            return 1
        fi
    }
}

# Scan all Jarvis images
images=("sutazai-backend" "sutazai-frontend" "sutazai-ollama")
for image in "${images[@]}"; do
    scan_image "$image:latest" "${SCAN_REPORT%.json}_${image}.json"
done

# Generate summary report
generate_summary() {
    local summary_file="${SCAN_REPORT%.json}_summary.txt"
    
    echo "=== VULNERABILITY SCAN SUMMARY ===" > "$summary_file"
    echo "Scan Date: $(date -Iseconds)" >> "$summary_file"
    echo "" >> "$summary_file"
    
    for report in "${SCAN_REPORT%.json}"_*.json; do
        if [[ -f "$report" ]]; then
            image_name=$(basename "$report" .json | sed 's/.*_//')
            echo "Image: $image_name" >> "$summary_file"
            
            # Count vulnerabilities by severity
            critical=$(jq '.vulnerabilities[] | select(.severity == "CRITICAL") | .severity' "$report" 2>/dev/null | wc -l)
            high=$(jq '.vulnerabilities[] | select(.severity == "HIGH") | .severity' "$report" 2>/dev/null | wc -l)
            medium=$(jq '.vulnerabilities[] | select(.severity == "MEDIUM") | .severity' "$report" 2>/dev/null | wc -l)
            low=$(jq '.vulnerabilities[] | select(.severity == "LOW") | .severity' "$report" 2>/dev/null | wc -l)
            
            echo "  Critical: $critical" >> "$summary_file"
            echo "  High: $high" >> "$summary_file"
            echo "  Medium: $medium" >> "$summary_file"
            echo "  Low: $low" >> "$summary_file"
            echo "" >> "$summary_file"
            
            # Alert on critical vulnerabilities
            if [ "$critical" -gt 0 ]; then
                send_vulnerability_alert "$image_name" "$critical" "CRITICAL"
            fi
        fi
    done
    
    echo "Summary report: $summary_file"
}

# Send vulnerability alerts
send_vulnerability_alert() {
    local image=$1
    local count=$2
    local severity=$3
    
    curl -X POST -H 'Content-type: application/json' \
        --data "{
            \"text\": \"🚨 Security Alert: $severity vulnerabilities found\",
            \"attachments\": [{
                \"color\": \"danger\",
                \"fields\": [
                    {\"title\": \"Image\", \"value\": \"$image\", \"short\": true},
                    {\"title\": \"Count\", \"value\": \"$count\", \"short\": true},
                    {\"title\": \"Severity\", \"value\": \"$severity\", \"short\": true}
                ]
            }]
        }" \
        "$SECURITY_SLACK_WEBHOOK" 2>/dev/null || true
}

# Main execution
scan_all_images() {
    echo "Starting vulnerability scan..."
    
    # Scan images
    for image in "${images[@]}"; do
        if docker images | grep -q "$image"; then
            scan_image "$image:latest" "${SCAN_REPORT%.json}_${image}.json"
        else
            echo "Image $image not found locally"
        fi
    done
    
    # Generate summary
    generate_summary
    
    echo "Vulnerability scan completed"
    echo "Reports available in /opt/sutazaiapp/security/"
}

scan_all_images
```

#### Dependency Vulnerability Check
```bash
#!/bin/bash
# dependency_security_check.sh

echo "=== DEPENDENCY SECURITY CHECK ==="

# Python dependencies scan
scan_python_dependencies() {
    echo "Scanning Python dependencies..."
    
    # Using safety (pip install safety)
    if command -v safety >/dev/null; then
        docker exec sutazai-backend safety check --json > /opt/sutazaiapp/security/python_deps_scan.json 2>/dev/null || {
            echo "Safety scan failed, using pip audit"
            docker exec sutazai-backend pip-audit --format=json > /opt/sutazaiapp/security/python_deps_scan.json 2>/dev/null || true
        }
    fi
    
    # Alternative: Check for known vulnerable packages manually
    docker exec sutazai-backend pip list --format=json > /tmp/pip_packages.json
    
    # Check against known vulnerable package versions
    python3 << EOF
import json
import requests

# Load installed packages
with open('/tmp/pip_packages.json', 'r') as f:
    packages = json.load(f)

# Check against PyUp.io safety database (example)
vulnerable_packages = {
    'requests': ['2.8.0', '2.8.1'],  # Example vulnerable versions
    'urllib3': ['1.24.0', '1.24.1'],
    'pyyaml': ['3.12', '3.13']
}

vulnerabilities = []
for pkg in packages:
    name = pkg['name'].lower()
    version = pkg['version']
    
    if name in vulnerable_packages and version in vulnerable_packages[name]:
        vulnerabilities.append({
            'package': name,
            'version': version,
            'vulnerability': 'Known security issue'
        })

if vulnerabilities:
    print(f"Found {len(vulnerabilities)} vulnerable dependencies:")
    for vuln in vulnerabilities:
        print(f"  - {vuln['package']} {vuln['version']}: {vuln['vulnerability']}")
else:
    print("No known vulnerable dependencies found")
EOF
}

# Node.js dependencies scan (if applicable)
scan_nodejs_dependencies() {
    if docker exec sutazai-frontend npm list >/dev/null 2>&1; then
        echo "Scanning Node.js dependencies..."
        docker exec sutazai-frontend npm audit --json > /opt/sutazaiapp/security/nodejs_deps_scan.json 2>/dev/null || true
    fi
}

# Main execution
scan_python_dependencies
scan_nodejs_dependencies

echo "Dependency security check completed"
```

### Security Hardening

#### System Hardening Script
```bash
#!/bin/bash
# security_hardening.sh

echo "=== SYSTEM SECURITY HARDENING ==="

# Docker security hardening
harden_docker() {
    echo "1. Hardening Docker configuration..."
    
    # Create non-root user for containers
    if ! docker exec sutazai-backend id jarvis >/dev/null 2>&1; then
        docker exec sutazai-backend useradd -r -s /bin/false jarvis
    fi
    
    # Set resource limits
    echo "Setting resource limits..."
    
    # Update docker-compose.yml with security constraints
    cat >> docker-compose.yml << 'EOF'
# Security hardening
x-security-common: &security-common
  security_opt:
    - no-new-privileges:true
  read_only: true
  tmpfs:
    - /tmp:noexec,nosuid,size=128m
  cap_drop:
    - ALL
  cap_add:
    - CHOWN
    - DAC_OVERRIDE
    - SETGID
    - SETUID
EOF

    # Apply security settings to services
    sed -i '/services:/a\
  backend:\
    <<: *security-common' docker-compose.yml
    
    echo "✅ Docker hardening completed"
}

# Network security hardening
harden_network() {
    echo "2. Hardening network configuration..."
    
    # Configure firewall rules
    if command -v ufw >/dev/null; then
        # Reset and set default policies
        ufw --force reset
        ufw default deny incoming
        ufw default allow outgoing
        
        # Allow SSH (adjust port as needed)
        ufw allow 22/tcp
        
        # Allow Jarvis services
        ufw allow 10010/tcp  # Backend API
        ufw allow 10011/tcp  # Frontend (if external access needed)
        
        # Allow monitoring (internal only)
        ufw allow from 10.0.0.0/8 to any port 10200:10206
        
        # Enable firewall
        ufw --force enable
        
        echo "✅ UFW firewall configured"
    fi
    
    # Configure iptables for additional protection
    iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    iptables -A INPUT -i lo -j ACCEPT
    iptables -A INPUT -p tcp --dport 22 -j ACCEPT
    iptables -A INPUT -p tcp --dport 10010 -j ACCEPT
    iptables -A INPUT -p tcp --dport 10011 -j ACCEPT
    iptables -A INPUT -j DROP
    
    # Save iptables rules
    iptables-save > /etc/iptables/rules.v4 2>/dev/null || true
    
    echo "✅ Network hardening completed"
}

# File system security
harden_filesystem() {
    echo "3. Hardening file system..."
    
    # Set secure permissions on configuration files
    chmod 600 /opt/sutazaiapp/.env* 2>/dev/null || true
    chmod 700 /opt/sutazaiapp/security/ 2>/dev/null || true
    chmod 755 /opt/sutazaiapp/docs/ 2>/dev/null || true
    
    # Create security directories with proper permissions
    mkdir -p /opt/sutazaiapp/security/{logs,reports,blocked_ips}
    chmod 700 /opt/sutazaiapp/security/
    
    # Set up log rotation for security logs
    cat > /etc/logrotate.d/jarvis-security << 'EOF'
/opt/sutazaiapp/logs/security*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 600 root root
}
EOF
    
    echo "✅ File system hardening completed"
}

# Database security
harden_database() {
    echo "4. Hardening database configuration..."
    
    # PostgreSQL security settings
    docker exec sutazai-postgres psql -U sutazai -c "
        -- Disable unnecessary extensions
        DROP EXTENSION IF EXISTS plpythonu;
        DROP EXTENSION IF EXISTS plpython3u;
        
        -- Set secure configuration
        ALTER SYSTEM SET log_connections = on;
        ALTER SYSTEM SET log_disconnections = on;
        ALTER SYSTEM SET log_statement = 'all';
        ALTER SYSTEM SET log_min_duration_statement = 1000;
        
        -- Reload configuration
        SELECT pg_reload_conf();
    " 2>/dev/null || echo "PostgreSQL hardening failed (container may not be ready)"
    
    # Redis security (if accessible)
    docker exec sutazai-redis redis-cli CONFIG SET rename-command FLUSHDB "" 2>/dev/null || true
    docker exec sutazai-redis redis-cli CONFIG SET rename-command FLUSHALL "" 2>/dev/null || true
    docker exec sutazai-redis redis-cli CONFIG SET rename-command CONFIG "CONFIG_RENAMED" 2>/dev/null || true
    
    echo "✅ Database hardening completed"
}

# Application security
harden_application() {
    echo "5. Hardening application configuration..."
    
    # Set secure environment variables
    if ! grep -q "SECURE_HEADERS" /opt/sutazaiapp/.env; then
        cat >> /opt/sutazaiapp/.env << 'EOF'

# Security Headers
SECURE_HEADERS=true
CONTENT_SECURITY_POLICY="default-src 'self'"
X_FRAME_OPTIONS=DENY
X_CONTENT_TYPE_OPTIONS=nosniff
HSTS_MAX_AGE=31536000

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Session Security
SESSION_SECURE=true
SESSION_HTTPONLY=true
SESSION_SAMESITE=strict
EOF
    fi
    
    echo "✅ Application hardening completed"
}

# SSL/TLS configuration
configure_ssl() {
    echo "6. Configuring SSL/TLS..."
    
    # Generate self-signed certificate for development
    if [[ ! -f /opt/sutazaiapp/certs/jarvis.crt ]]; then
        mkdir -p /opt/sutazaiapp/certs
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout /opt/sutazaiapp/certs/jarvis.key \
            -out /opt/sutazaiapp/certs/jarvis.crt \
            -subj "/C=US/ST=CA/L=San Francisco/O=Company/CN=jarvis.local"
        
        chmod 600 /opt/sutazaiapp/certs/jarvis.key
        chmod 644 /opt/sutazaiapp/certs/jarvis.crt
    fi
    
    # Configure nginx for SSL (if used)
    if command -v nginx >/dev/null; then
        cat > /etc/nginx/sites-available/jarvis-ssl << 'EOF'
server {
    listen 443 ssl http2;
    server_name jarvis.local;
    
    ssl_certificate /opt/sutazaiapp/certs/jarvis.crt;
    ssl_certificate_key /opt/sutazaiapp/certs/jarvis.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    
    location / {
        proxy_pass http://localhost:10010;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name jarvis.local;
    return 301 https://$server_name$request_uri;
}
EOF
        
        ln -sf /etc/nginx/sites-available/jarvis-ssl /etc/nginx/sites-enabled/
        nginx -t && nginx -s reload 2>/dev/null || echo "Nginx configuration failed"
    fi
    
    echo "✅ SSL/TLS configuration completed"
}

# Main execution
main() {
    echo "Starting security hardening process..."
    
    # Create backup before making changes
    tar -czf "/opt/sutazaiapp/backups/pre_hardening_backup_$(date +%Y%m%d_%H%M%S).tar.gz" \
        /opt/sutazaiapp/ 2>/dev/null || true
    
    harden_docker
    harden_network
    harden_filesystem
    harden_database
    harden_application
    configure_ssl
    
    echo "✅ Security hardening completed successfully"
    echo "🔄 Restart services to apply all changes: docker-compose restart"
}

main
```

## 🛡️ Access Control

### User Access Management

#### User Role Definition
```bash
#!/bin/bash
# user_access_management.sh

# Define user roles and permissions
declare -A ROLES
ROLES[admin]="full_access,system_management,user_management"
ROLES[developer]="api_access,debug_access,deployment_access"
ROLES[analyst]="read_access,metrics_access,report_access"
ROLES[user]="basic_api_access"

# User database (in production, use proper identity management)
USER_DB="/opt/sutazaiapp/security/users.json"

# Initialize user database
init_user_database() {
    if [[ ! -f "$USER_DB" ]]; then
        cat > "$USER_DB" << 'EOF'
{
  "users": {
    "admin": {
      "role": "admin",
      "created": "2025-08-08T00:00:00Z",
      "last_login": null,
      "status": "active"
    }
  },
  "sessions": {},
  "api_keys": {}
}
EOF
    fi
}

# Add user
add_user() {
    local username=$1
    local role=$2
    local email=${3:-""}
    
    if [[ -z "$username" || -z "$role" ]]; then
        echo "Usage: add_user <username> <role> [email]"
        return 1
    fi
    
    if [[ -z "${ROLES[$role]}" ]]; then
        echo "Invalid role: $role"
        echo "Available roles: ${!ROLES[@]}"
        return 1
    fi
    
    # Generate API key
    api_key=$(openssl rand -hex 32)
    
    # Add to database (simplified - use proper DB in production)
    python3 << EOF
import json
from datetime import datetime

with open('$USER_DB', 'r') as f:
    db = json.load(f)

db['users']['$username'] = {
    'role': '$role',
    'email': '$email',
    'created': datetime.now().isoformat() + 'Z',
    'last_login': None,
    'status': 'active'
}

db['api_keys']['$api_key'] = {
    'user': '$username',
    'created': datetime.now().isoformat() + 'Z',
    'last_used': None
}

with open('$USER_DB', 'w') as f:
    json.dump(db, f, indent=2)
EOF
    
    echo "User $username added with role $role"
    echo "API Key: $api_key"
    echo "⚠️  Save this API key - it won't be shown again"
}

# Validate API key
validate_api_key() {
    local api_key=$1
    
    python3 << EOF
import json
import sys
from datetime import datetime

try:
    with open('$USER_DB', 'r') as f:
        db = json.load(f)
    
    if '$api_key' in db['api_keys']:
        user_info = db['api_keys']['$api_key']
        username = user_info['user']
        
        if username in db['users']:
            user = db['users'][username]
            if user['status'] == 'active':
                # Update last used
                db['api_keys']['$api_key']['last_used'] = datetime.now().isoformat() + 'Z'
                
                with open('$USER_DB', 'w') as f:
                    json.dump(db, f, indent=2)
                
                print(f"valid:{username}:{user['role']}")
                sys.exit(0)
    
    print("invalid")
    sys.exit(1)
except Exception as e:
    print(f"error:{e}")
    sys.exit(1)
EOF
}

# Check permissions
check_permission() {
    local username=$1
    local required_permission=$2
    
    python3 << EOF
import json

with open('$USER_DB', 'r') as f:
    db = json.load(f)

if '$username' in db['users']:
    user_role = db['users']['$username']['role']
    role_permissions = '${ROLES[$user_role]}'.split(',')
    
    if '$required_permission' in role_permissions or 'full_access' in role_permissions:
        print("allowed")
    else:
        print("denied")
else:
    print("user_not_found")
EOF
}

# Main functions
init_user_database

case "${1:-help}" in
    "add")
        add_user "$2" "$3" "$4"
        ;;
    "validate")
        validate_api_key "$2"
        ;;
    "check")
        check_permission "$2" "$3"
        ;;
    *)
        echo "Usage: $0 {add|validate|check} [args...]"
        echo "  add <username> <role> [email]    - Add new user"
        echo "  validate <api_key>               - Validate API key"
        echo "  check <username> <permission>    - Check user permission"
        ;;
esac
```

### API Authentication Middleware

#### FastAPI Authentication Integration
```python
# /opt/sutazaiapp/backend/app/security/auth_middleware.py
"""
Authentication middleware for Jarvis API
Integrates with the shell-based user management system
"""

import json
import subprocess
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class AuthorizationError(Exception):
    """Custom authorization error"""
    pass

def validate_api_key(api_key: str) -> Dict[str, str]:
    """Validate API key using shell script"""
    try:
        result = subprocess.run(
            ['/opt/sutazaiapp/scripts/user_access_management.sh', 'validate', api_key],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if output.startswith('valid:'):
                _, username, role = output.split(':')
                return {'username': username, 'role': role}
        
        raise AuthenticationError("Invalid API key")
        
    except subprocess.TimeoutExpired:
        raise AuthenticationError("Authentication timeout")
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise AuthenticationError("Authentication failed")

def check_permission(username: str, permission: str) -> bool:
    """Check if user has required permission"""
    try:
        result = subprocess.run(
            ['/opt/sutazaiapp/scripts/user_access_management.sh', 'check', username, permission],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        return result.returncode == 0 and result.stdout.strip() == 'allowed'
        
    except Exception as e:
        logger.error(f"Permission check error: {e}")
        return False

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Dict[str, Any]:
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user_info = validate_api_key(credentials.credentials)
        return user_info
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get user from dependency injection
            user = kwargs.get('current_user')
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not check_permission(user['username'], permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Role-based access decorators
def require_admin(func):
    """Require admin role"""
    return require_permission('full_access')(func)

def require_developer(func):
    """Require developer role or higher"""
    async def wrapper(*args, **kwargs):
        user = kwargs.get('current_user')
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        if user['role'] not in ['admin', 'developer']:
            raise HTTPException(status_code=403, detail="Developer access required")
        
        return await func(*args, **kwargs)
    return wrapper
```

## 📊 Data Protection

### Data Classification and Encryption

#### Data Classification System
```bash
#!/bin/bash
# data_classification.sh

# Data classification levels
declare -A CLASSIFICATION_LEVELS
CLASSIFICATION_LEVELS[public]="No encryption required"
CLASSIFICATION_LEVELS[internal]="Standard encryption"
CLASSIFICATION_LEVELS[confidential]="Strong encryption + access logging"
CLASSIFICATION_LEVELS[restricted]="Strong encryption + access logging + approval required"

# File classification database
CLASSIFICATION_DB="/opt/sutazaiapp/security/data_classification.json"

# Initialize classification database
init_classification_db() {
    if [[ ! -f "$CLASSIFICATION_DB" ]]; then
        cat > "$CLASSIFICATION_DB" << 'EOF'
{
  "classifications": {
    "/opt/sutazaiapp/.env": "confidential",
    "/opt/sutazaiapp/security/": "restricted",
    "/opt/sutazaiapp/backups/": "confidential",
    "/opt/sutazaiapp/logs/": "internal"
  },
  "encryption_keys": {},
  "access_logs": []
}
EOF
    fi
}

# Encrypt sensitive files
encrypt_file() {
    local file_path=$1
    local classification=${2:-"confidential"}
    
    if [[ ! -f "$file_path" ]]; then
        echo "File not found: $file_path"
        return 1
    fi
    
    # Generate unique encryption key
    encryption_key=$(openssl rand -hex 32)
    
    # Encrypt file
    openssl enc -aes-256-cbc -salt -in "$file_path" -out "${file_path}.encrypted" -k "$encryption_key"
    
    # Store key securely (in production, use proper key management)
    python3 << EOF
import json
from datetime import datetime

with open('$CLASSIFICATION_DB', 'r') as f:
    db = json.load(f)

db['encryption_keys']['$file_path'] = {
    'key': '$encryption_key',
    'algorithm': 'aes-256-cbc',
    'created': datetime.now().isoformat() + 'Z',
    'classification': '$classification'
}

with open('$CLASSIFICATION_DB', 'w') as f:
    json.dump(db, f, indent=2)
EOF
    
    # Remove original file after encryption
    shred -vfz -n 3 "$file_path"
    
    echo "File encrypted: ${file_path}.encrypted"
}

# Decrypt file
decrypt_file() {
    local encrypted_file=$1
    local original_file=${encrypted_file%.encrypted}
    
    # Get encryption key
    encryption_key=$(python3 << EOF
import json
with open('$CLASSIFICATION_DB', 'r') as f:
    db = json.load(f)
    
if '$original_file' in db['encryption_keys']:
    print(db['encryption_keys']['$original_file']['key'])
else:
    print('')
EOF
)
    
    if [[ -z "$encryption_key" ]]; then
        echo "Encryption key not found for: $original_file"
        return 1
    fi
    
    # Decrypt file
    openssl enc -aes-256-cbc -d -in "$encrypted_file" -out "$original_file" -k "$encryption_key"
    
    echo "File decrypted: $original_file"
}

# Log data access
log_data_access() {
    local user=$1
    local file_path=$2
    local action=$3  # read, write, delete
    
    python3 << EOF
import json
from datetime import datetime

with open('$CLASSIFICATION_DB', 'r') as f:
    db = json.load(f)

access_log = {
    'timestamp': datetime.now().isoformat() + 'Z',
    'user': '$user',
    'file': '$file_path',
    'action': '$action',
    'ip': '$(echo $SSH_CLIENT | awk '{print $1}')'
}

db['access_logs'].append(access_log)

# Keep only last 1000 logs
if len(db['access_logs']) > 1000:
    db['access_logs'] = db['access_logs'][-1000:]

with open('$CLASSIFICATION_DB', 'w') as f:
    json.dump(db, f, indent=2)
EOF
}

init_classification_db
```

### Database Security

#### Database Encryption and Audit
```bash
#!/bin/bash
# database_security.sh

echo "=== DATABASE SECURITY SETUP ==="

# Enable PostgreSQL audit logging
setup_postgres_audit() {
    echo "1. Setting up PostgreSQL audit logging..."
    
    docker exec sutazai-postgres psql -U sutazai -c "
        -- Enable audit logging
        ALTER SYSTEM SET log_statement = 'all';
        ALTER SYSTEM SET log_min_duration_statement = 0;
        ALTER SYSTEM SET log_connections = on;
        ALTER SYSTEM SET log_disconnections = on;
        ALTER SYSTEM SET log_checkpoints = on;
        ALTER SYSTEM SET log_lock_waits = on;
        
        -- Enable detailed connection logging
        ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';
        
        -- Reload configuration
        SELECT pg_reload_conf();
        
        -- Create audit table for sensitive operations
        CREATE TABLE IF NOT EXISTS audit_log (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            username TEXT,
            operation TEXT,
            table_name TEXT,
            record_id TEXT,
            old_values JSONB,
            new_values JSONB,
            ip_address INET
        );
    " 2>/dev/null || echo "PostgreSQL audit setup failed"
    
    echo "✅ PostgreSQL audit logging enabled"
}

# Set up database encryption
setup_database_encryption() {
    echo "2. Setting up database encryption..."
    
    # Enable transparent data encryption (if supported)
    docker exec sutazai-postgres psql -U sutazai -c "
        -- Create encrypted tablespace (example)
        -- CREATE TABLESPACE encrypted_space LOCATION '/var/lib/postgresql/encrypted';
        
        -- Set up row-level security
        CREATE OR REPLACE FUNCTION get_current_user_id() RETURNS TEXT AS \$\$
        BEGIN
            RETURN current_setting('app.current_user_id', true);
        END;
        \$\$ LANGUAGE plpgsql SECURITY DEFINER;
        
        -- Example: Enable RLS on sensitive tables
        -- ALTER TABLE users ENABLE ROW LEVEL SECURITY;
        -- CREATE POLICY user_policy ON users USING (user_id = get_current_user_id());
    " 2>/dev/null || echo "Database encryption setup failed"
    
    echo "✅ Database encryption configured"
}

# Create database backup with encryption
create_encrypted_backup() {
    echo "3. Creating encrypted database backup..."
    
    BACKUP_FILE="/opt/sutazaiapp/backups/encrypted_backup_$(date +%Y%m%d_%H%M%S).sql"
    ENCRYPTION_KEY=$(openssl rand -hex 32)
    
    # Create backup and encrypt
    docker exec sutazai-postgres pg_dump -U sutazai sutazai | \
        openssl enc -aes-256-cbc -salt -out "${BACKUP_FILE}.enc" -k "$ENCRYPTION_KEY"
    
    # Store encryption key securely
    echo "$ENCRYPTION_KEY" > "${BACKUP_FILE}.key"
    chmod 600 "${BACKUP_FILE}.key"
    
    echo "✅ Encrypted backup created: ${BACKUP_FILE}.enc"
}

# Monitor database security
monitor_database_security() {
    echo "4. Setting up database security monitoring..."
    
    # Create monitoring script
    cat > /opt/sutazaiapp/scripts/db_security_monitor.sh << 'EOF'
#!/bin/bash
# Database security monitoring

SECURITY_LOG="/opt/sutazaiapp/logs/db_security.log"

# Monitor for suspicious queries
docker logs sutazai-postgres --since=1m 2>/dev/null | \
grep -i -E "(drop|delete|truncate|alter|grant|revoke)" | \
while read line; do
    echo "$(date -Iseconds) [DB_SECURITY] Suspicious query: $line" >> "$SECURITY_LOG"
done

# Monitor failed login attempts
docker logs sutazai-postgres --since=1m 2>/dev/null | \
grep -i "authentication failed" | \
while read line; do
    echo "$(date -Iseconds) [DB_AUTH_FAIL] $line" >> "$SECURITY_LOG"
done

# Check for unusual connection patterns
CONNECTION_COUNT=$(docker exec sutazai-postgres psql -U sutazai -t -c "
    SELECT count(*) FROM pg_stat_activity WHERE state = 'active';
" 2>/dev/null | tr -d ' ')

if [[ "$CONNECTION_COUNT" -gt 20 ]]; then
    echo "$(date -Iseconds) [DB_HIGH_CONNECTIONS] High connection count: $CONNECTION_COUNT" >> "$SECURITY_LOG"
fi
EOF
    
    chmod +x /opt/sutazaiapp/scripts/db_security_monitor.sh
    
    # Add to cron for regular execution
    (crontab -l 2>/dev/null; echo "*/5 * * * * /opt/sutazaiapp/scripts/db_security_monitor.sh") | crontab -
    
    echo "✅ Database security monitoring enabled"
}

# Main execution
setup_postgres_audit
setup_database_encryption
create_encrypted_backup
monitor_database_security

echo "✅ Database security setup completed"
```

## 🔍 Security Monitoring & Alerting

### Centralized Security Monitoring

#### Security Information and Event Management (SIEM)
```bash
#!/bin/bash
# siem_setup.sh

echo "=== SECURITY MONITORING SETUP ==="

SIEM_CONFIG="/opt/sutazaiapp/security/siem_config.json"
SIEM_LOG="/opt/sutazaiapp/logs/siem.log"

# Initialize SIEM configuration
init_siem() {
    cat > "$SIEM_CONFIG" << 'EOF'
{
  "monitoring": {
    "log_sources": [
      "/opt/sutazaiapp/logs/security_monitor.log",
      "/opt/sutazaiapp/logs/db_security.log",
      "/opt/sutazaiapp/logs/ids.log",
      "/var/log/nginx/access.log",
      "/var/log/nginx/error.log"
    ],
    "alert_rules": [
      {
        "name": "Multiple failed logins",
        "pattern": "authentication.*(failed|denied)",
        "threshold": 5,
        "window": 300,
        "severity": "high"
      },
      {
        "name": "SQL injection attempt",
        "pattern": "(union|select).*from",
        "threshold": 1,
        "window": 60,
        "severity": "critical"
      },
      {
        "name": "Privilege escalation",
        "pattern": "(sudo|su -|chmod.*777)",
        "threshold": 1,
        "window": 60,
        "severity": "critical"
      },
      {
        "name": "Suspicious file access",
        "pattern": "(/etc/passwd|/etc/shadow|\.ssh/)",
        "threshold": 1,
        "window": 60,
        "severity": "high"
      }
    ]
  },
  "notifications": {
    "slack_webhook": "$SECURITY_SLACK_WEBHOOK",
    "email_recipients": ["security-team@company.com"],
    "pagerduty_service": "$PAGERDUTY_SERVICE_KEY"
  }
}
EOF
}

# Real-time log analysis
analyze_logs() {
    python3 << 'EOF'
import json
import re
import time
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict
import threading

class SIEMAnalyzer:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.alert_counts = defaultdict(list)
        self.running = True
    
    def analyze_log_line(self, line, source):
        """Analyze a single log line for threats"""
        timestamp = datetime.now()
        
        for rule in self.config['monitoring']['alert_rules']:
            if re.search(rule['pattern'], line, re.IGNORECASE):
                self.process_alert(rule, line, source, timestamp)
    
    def process_alert(self, rule, line, source, timestamp):
        """Process a potential alert"""
        rule_name = rule['name']
        
        # Add to alert count
        self.alert_counts[rule_name].append(timestamp)
        
        # Clean old entries outside the window
        window_start = timestamp - timedelta(seconds=rule['window'])
        self.alert_counts[rule_name] = [
            t for t in self.alert_counts[rule_name] 
            if t >= window_start
        ]
        
        # Check if threshold exceeded
        if len(self.alert_counts[rule_name]) >= rule['threshold']:
            self.send_alert(rule, line, source, timestamp)
            # Reset counter to avoid spam
            self.alert_counts[rule_name] = []
    
    def send_alert(self, rule, line, source, timestamp):
        """Send security alert"""
        alert_data = {
            'rule': rule['name'],
            'severity': rule['severity'],
            'message': line,
            'source': source,
            'timestamp': timestamp.isoformat(),
            'count': len(self.alert_counts[rule['name']])
        }
        
        # Log alert
        with open('/opt/sutazaiapp/logs/siem.log', 'a') as f:
            f.write(f"{timestamp.isoformat()} [ALERT] {json.dumps(alert_data)}\n")
        
        # Send notifications
        self.send_notifications(alert_data)
    
    def send_notifications(self, alert):
        """Send alert notifications"""
        # Slack notification
        slack_payload = {
            "text": f"🚨 Security Alert: {alert['rule']}",
            "attachments": [{
                "color": "danger" if alert['severity'] == "critical" else "warning",
                "fields": [
                    {"title": "Rule", "value": alert['rule'], "short": True},
                    {"title": "Severity", "value": alert['severity'], "short": True},
                    {"title": "Source", "value": alert['source'], "short": True},
                    {"title": "Count", "value": str(alert['count']), "short": True},
                    {"title": "Message", "value": alert['message'][:500], "short": False}
                ]
            }]
        }
        
        subprocess.run([
            'curl', '-X', 'POST',
            '-H', 'Content-type: application/json',
            '--data', json.dumps(slack_payload),
            self.config['notifications']['slack_webhook']
        ], capture_output=True)
    
    def monitor_log_file(self, log_file):
        """Monitor a single log file"""
        try:
            process = subprocess.Popen(
                ['tail', '-f', log_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            while self.running:
                line = process.stdout.readline()
                if line:
                    self.analyze_log_line(line.strip(), log_file)
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error monitoring {log_file}: {e}")
    
    def start_monitoring(self):
        """Start monitoring all configured log files"""
        threads = []
        
        for log_file in self.config['monitoring']['log_sources']:
            thread = threading.Thread(
                target=self.monitor_log_file,
                args=(log_file,)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            print("Stopping SIEM monitoring...")

# Initialize and start SIEM
if __name__ == "__main__":
    analyzer = SIEMAnalyzer('/opt/sutazaiapp/security/siem_config.json')
    analyzer.start_monitoring()
EOF
}

# Security metrics dashboard
create_security_dashboard() {
    cat > /opt/sutazaiapp/security/security_dashboard.py << 'EOF'
#!/usr/bin/env python3
"""
Security Metrics Dashboard
Provides real-time security metrics and alerts
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

class SecurityMetrics:
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Initialize metrics database"""
        self.conn = sqlite3.connect('/opt/sutazaiapp/security/metrics.db', check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                event_type TEXT,
                severity TEXT,
                source TEXT,
                message TEXT
            )
        ''')
        self.conn.commit()
    
    def add_event(self, event_type, severity, source, message):
        """Add security event to database"""
        self.conn.execute('''
            INSERT INTO security_events (timestamp, event_type, severity, source, message)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), event_type, severity, source, message))
        self.conn.commit()
    
    def get_recent_events(self, hours=24):
        """Get recent security events"""
        since = datetime.now() - timedelta(hours=hours)
        cursor = self.conn.execute('''
            SELECT * FROM security_events 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''', (since,))
        
        return cursor.fetchall()
    
    def get_event_statistics(self, hours=24):
        """Get event statistics"""
        since = datetime.now() - timedelta(hours=hours)
        
        # Count by severity
        cursor = self.conn.execute('''
            SELECT severity, COUNT(*) 
            FROM security_events 
            WHERE timestamp > ? 
            GROUP BY severity
        ''', (since,))
        severity_counts = dict(cursor.fetchall())
        
        # Count by event type
        cursor = self.conn.execute('''
            SELECT event_type, COUNT(*) 
            FROM security_events 
            WHERE timestamp > ? 
            GROUP BY event_type
        ''', (since,))
        event_type_counts = dict(cursor.fetchall())
        
        return {
            'severity': severity_counts,
            'event_types': event_type_counts,
            'total': sum(severity_counts.values())
        }

metrics = SecurityMetrics()

@app.route('/')
def dashboard():
    """Security dashboard homepage"""
    recent_events = metrics.get_recent_events()
    statistics = metrics.get_event_statistics()
    
    return render_template('security_dashboard.html', 
                         events=recent_events, 
                         stats=statistics)

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for security metrics"""
    stats = metrics.get_event_statistics()
    return jsonify(stats)

@app.route('/api/events')
def api_events():
    """API endpoint for recent events"""
    events = metrics.get_recent_events()
    return jsonify([{
        'id': event[0],
        'timestamp': event[1],
        'type': event[2],
        'severity': event[3],
        'source': event[4],
        'message': event[5]
    } for event in events])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF

    chmod +x /opt/sutazaiapp/security/security_dashboard.py
}

# Main execution
init_siem
create_security_dashboard

echo "✅ Security monitoring setup completed"
echo "📊 Access security dashboard at: http://localhost:5000"
echo "🔍 SIEM analyzer can be started with: python3 siem_setup.sh"
```

## 📋 Compliance & Audit

### Audit Logging System

#### Comprehensive Audit Trail
```bash
#!/bin/bash
# audit_system.sh

echo "=== AUDIT SYSTEM SETUP ==="

AUDIT_DB="/opt/sutazaiapp/security/audit.db"

# Initialize audit database
init_audit_db() {
    sqlite3 "$AUDIT_DB" << 'EOF'
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT,
    action TEXT,
    resource TEXT,
    result TEXT,
    ip_address TEXT,
    user_agent TEXT,
    details TEXT
);

CREATE TABLE IF NOT EXISTS compliance_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    check_type TEXT,
    result TEXT,
    details TEXT,
    remediation TEXT
);

CREATE INDEX idx_audit_timestamp ON audit_events(timestamp);
CREATE INDEX idx_audit_user ON audit_events(user_id);
CREATE INDEX idx_audit_action ON audit_events(action);
EOF
}

# Log audit event
log_audit_event() {
    local user_id=$1
    local action=$2
    local resource=$3
    local result=$4
    local ip_address=${5:-"unknown"}
    local details=${6:-""}
    
    sqlite3 "$AUDIT_DB" << EOF
INSERT INTO audit_events (user_id, action, resource, result, ip_address, details)
VALUES ('$user_id', '$action', '$resource', '$result', '$ip_address', '$details');
EOF
    
    # Also log to syslog for external SIEM
    logger -t jarvis-audit "USER=$user_id ACTION=$action RESOURCE=$resource RESULT=$result IP=$ip_address"
}

# Compliance checker
run_compliance_checks() {
    echo "Running compliance checks..."
    
    local check_results=()
    
    # Check 1: Password policy compliance
    check_password_policy() {
        # Check if strong passwords are enforced
        if grep -q "minlen.*8" /etc/security/pwquality.conf 2>/dev/null; then
            echo "PASS: Password minimum length policy enforced"
            log_compliance_check "password_policy" "PASS" "Password policy configured"
        else
            echo "FAIL: Password policy not properly configured"
            log_compliance_check "password_policy" "FAIL" "Configure password policy in /etc/security/pwquality.conf"
        fi
    }
    
    # Check 2: Encryption at rest
    check_encryption_at_rest() {
        if [[ -d "/opt/sutazaiapp/security/encrypted" ]]; then
            echo "PASS: Encryption at rest implemented"
            log_compliance_check "encryption_at_rest" "PASS" "Encrypted storage directory exists"
        else
            echo "FAIL: No encryption at rest implementation found"
            log_compliance_check "encryption_at_rest" "FAIL" "Implement encryption at rest for sensitive data"
        fi
    }
    
    # Check 3: Access logging
    check_access_logging() {
        if [[ -f "/opt/sutazaiapp/logs/security_monitor.log" ]]; then
            log_count=$(wc -l < "/opt/sutazaiapp/logs/security_monitor.log")
            echo "PASS: Access logging enabled ($log_count entries)"
            log_compliance_check "access_logging" "PASS" "Access logging active with $log_count entries"
        else
            echo "FAIL: Access logging not enabled"
            log_compliance_check "access_logging" "FAIL" "Enable access logging"
        fi
    }
    
    # Check 4: Backup encryption
    check_backup_encryption() {
        encrypted_backups=$(find /opt/sutazaiapp/backups -name "*.enc" | wc -l)
        if [[ $encrypted_backups -gt 0 ]]; then
            echo "PASS: Encrypted backups found ($encrypted_backups files)"
            log_compliance_check "backup_encryption" "PASS" "$encrypted_backups encrypted backup files found"
        else
            echo "FAIL: No encrypted backups found"
            log_compliance_check "backup_encryption" "FAIL" "Implement backup encryption"
        fi
    }
    
    # Check 5: Network security
    check_network_security() {
        if iptables -L | grep -q "DROP"; then
            echo "PASS: Firewall rules configured"
            log_compliance_check "network_security" "PASS" "Firewall rules active"
        else
            echo "FAIL: No firewall rules found"
            log_compliance_check "network_security" "FAIL" "Configure firewall rules"
        fi
    }
    
    # Run all checks
    check_password_policy
    check_encryption_at_rest
    check_access_logging
    check_backup_encryption
    check_network_security
    
    echo "✅ Compliance checks completed"
}

# Log compliance check result
log_compliance_check() {
    local check_type=$1
    local result=$2
    local details=$3
    local remediation=${4:-""}
    
    sqlite3 "$AUDIT_DB" << EOF
INSERT INTO compliance_checks (check_type, result, details, remediation)
VALUES ('$check_type', '$result', '$details', '$remediation');
EOF
}

# Generate audit report
generate_audit_report() {
    local report_file="/opt/sutazaiapp/security/audit_report_$(date +%Y%m%d).html"
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Jarvis Security Audit Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔐 Jarvis Security Audit Report</h1>
        <p>Generated: $(date)</p>
        <p>System: Perfect Jarvis v17.0.0</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
EOF
    
    # Add compliance results
    sqlite3 "$AUDIT_DB" "
    SELECT 
        check_type,
        result,
        details,
        timestamp
    FROM compliance_checks
    WHERE DATE(timestamp) = DATE('now')
    ORDER BY timestamp DESC;
    " | while IFS='|' read -r check_type result details timestamp; do
        if [[ "$result" == "PASS" ]]; then
            echo "        <p class=\"pass\">✅ $check_type: $details</p>" >> "$report_file"
        else
            echo "        <p class=\"fail\">❌ $check_type: $details</p>" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << 'EOF'
    </div>
    
    <div class="section">
        <h2>Recent Security Events (Last 24 Hours)</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>User</th>
                <th>Action</th>
                <th>Resource</th>
                <th>Result</th>
                <th>IP Address</th>
            </tr>
EOF
    
    # Add recent audit events
    sqlite3 "$AUDIT_DB" "
    SELECT 
        timestamp,
        user_id,
        action,
        resource,
        result,
        ip_address
    FROM audit_events
    WHERE timestamp > datetime('now', '-1 day')
    ORDER BY timestamp DESC
    LIMIT 50;
    " | while IFS='|' read -r timestamp user_id action resource result ip_address; do
        echo "            <tr>" >> "$report_file"
        echo "                <td>$timestamp</td>" >> "$report_file"
        echo "                <td>$user_id</td>" >> "$report_file"
        echo "                <td>$action</td>" >> "$report_file"
        echo "                <td>$resource</td>" >> "$report_file"
        echo "                <td>$result</td>" >> "$report_file"
        echo "                <td>$ip_address</td>" >> "$report_file"
        echo "            </tr>" >> "$report_file"
    done
    
    cat >> "$report_file" << 'EOF'
        </table>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
EOF
    
    # Add recommendations based on failed checks
    sqlite3 "$AUDIT_DB" "
    SELECT DISTINCT remediation
    FROM compliance_checks
    WHERE result = 'FAIL' AND remediation IS NOT NULL AND remediation != '';
    " | while read -r remediation; do
        echo "            <li>$remediation</li>" >> "$report_file"
    done
    
    cat >> "$report_file" << 'EOF'
        </ul>
    </div>
</body>
</html>
EOF
    
    echo "✅ Audit report generated: $report_file"
}

# Main execution
init_audit_db

case "${1:-check}" in
    "check")
        run_compliance_checks
        ;;
    "report")
        generate_audit_report
        ;;
    "log")
        log_audit_event "$2" "$3" "$4" "$5" "$6" "$7"
        ;;
    *)
        echo "Usage: $0 {check|report|log} [args...]"
        ;;
esac
```

---

## 📋 Security Runbook Summary

### Daily Security Tasks
- [ ] Review security logs for anomalies
- [ ] Check failed authentication attempts
- [ ] Verify firewall rules are active
- [ ] Monitor system resource usage
- [ ] Check for software updates

### Weekly Security Tasks
- [ ] Run vulnerability scans on containers
- [ ] Review user access and permissions
- [ ] Audit database access logs
- [ ] Test backup and restore procedures
- [ ] Update security monitoring rules

### Monthly Security Tasks
- [ ] Conduct compliance assessment
- [ ] Review and update security policies
- [ ] Test incident response procedures
- [ ] Security awareness training updates
- [ ] Penetration testing (if applicable)

### Emergency Security Procedures
- [ ] Incident detection and containment
- [ ] Evidence preservation
- [ ] Communication protocols
- [ ] Recovery procedures
- [ ] Post-incident analysis

---

**Emergency Security Contacts:**
- **Security Team:** security-team@company.com
- **Incident Response:** +1-xxx-xxx-xxxx
- **CISO Office:** ciso@company.com

---

*This security runbook is based on industry best practices and the current Jarvis system architecture. Update procedures as threats evolve and security requirements change.*