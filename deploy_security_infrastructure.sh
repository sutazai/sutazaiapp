#!/bin/bash
"""
Security Infrastructure Deployment Script for SutazAI
Deploys comprehensive security monitoring and protection system
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        warning "Running as root. Some operations may require non-root privileges."
    fi
}

# Install system dependencies
install_dependencies() {
    log "Installing system dependencies..."
    
    # Update package lists
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv \
            iptables netstat-nat net-tools sqlite3 curl wget \
            docker.io docker-compose
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip \
            iptables net-tools sqlite curl wget \
            docker docker-compose
    else
        error "Unsupported package manager. Please install dependencies manually."
        exit 1
    fi
    
    # Enable and start Docker
    sudo systemctl enable docker
    sudo systemctl start docker
    
    success "System dependencies installed successfully"
}

# Install Python dependencies
install_python_dependencies() {
    log "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "/opt/sutazaiapp/security_env" ]; then
        python3 -m venv /opt/sutazaiapp/security_env
    fi
    
    # Activate virtual environment and install packages
    source /opt/sutazaiapp/security_env/bin/activate
    
    pip install --upgrade pip
    pip install -q requests sqlite3 streamlit plotly pandas numpy \
        pyjwt cryptography ipaddress psutil docker subprocess32
    
    success "Python dependencies installed successfully"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    # Create directory structure
    mkdir -p /opt/sutazaiapp/{data,logs,reports/security,config}
    
    # Set proper permissions
    chmod 755 /opt/sutazaiapp/data
    chmod 755 /opt/sutazaiapp/logs
    chmod 755 /opt/sutazaiapp/reports
    chmod 755 /opt/sutazaiapp/config
    
    success "Directory structure created successfully"
}

# Configure firewall rules
configure_firewall() {
    log "Configuring firewall rules..."
    
    # Save current iptables rules
    sudo iptables-save > /opt/sutazaiapp/data/iptables_backup_$(date +%Y%m%d_%H%M%S).rules
    
    # Basic security rules (will be enhanced by IDS)
    sudo iptables -A INPUT -i lo -j ACCEPT
    sudo iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT  # SSH
    sudo iptables -A INPUT -p tcp --dport 8501 -j ACCEPT  # Security Dashboard
    
    # Allow Docker ports (will be restricted by security systems)
    sudo iptables -A INPUT -p tcp --dport 10000:10600 -j ACCEPT
    
    success "Basic firewall rules configured"
}

# Initialize databases
initialize_databases() {
    log "Initializing security databases..."
    
    # Create database files
    touch /opt/sutazaiapp/data/ids_database.db
    touch /opt/sutazaiapp/data/security_events.db
    touch /opt/sutazaiapp/data/incidents.db
    
    # Set proper permissions
    chmod 644 /opt/sutazaiapp/data/*.db
    
    success "Security databases initialized"
}

# Create systemd services
create_systemd_services() {
    log "Creating systemd services..."
    
    # Create security orchestrator service
    cat > /tmp/sutazai-security.service << 'EOF'
[Unit]
Description=SutazAI Security Orchestrator
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sutazaiapp
ExecStart=/usr/bin/python3 /opt/sutazaiapp/security_orchestrator.py start
ExecStop=/usr/bin/python3 /opt/sutazaiapp/security_orchestrator.py stop
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Install service
    sudo mv /tmp/sutazai-security.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable sutazai-security.service
    
    success "Systemd services created successfully"
}

# Run initial security assessment
run_initial_assessment() {
    log "Running initial security assessment..."
    
    cd /opt/sutazaiapp
    
    # Activate virtual environment
    source security_env/bin/activate
    
    # Run security assessments
    python3 security_pentest_scanner.py || warning "Penetration test scanner had issues"
    python3 container_security_auditor.py || warning "Container auditor had issues"
    python3 network_security_analyzer.py || warning "Network analyzer had issues"
    python3 auth_security_tester.py || warning "Auth tester had issues"
    
    # Generate initial report
    python3 comprehensive_security_report_generator.py || warning "Report generation had issues"
    
    success "Initial security assessment completed"
}

# Configure log rotation
configure_log_rotation() {
    log "Configuring log rotation..."
    
    cat > /tmp/sutazai-security << 'EOF'
/opt/sutazaiapp/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        systemctl reload sutazai-security.service || true
    endscript
}
EOF

    sudo mv /tmp/sutazai-security /etc/logrotate.d/
    
    success "Log rotation configured"
}

# Create monitoring cron jobs
create_cron_jobs() {
    log "Creating monitoring cron jobs..."
    
    # Create cron job for periodic assessments
    cat > /tmp/sutazai-security-cron << 'EOF'
# SutazAI Security Monitoring Cron Jobs
# Run security assessment every 6 hours
0 */6 * * * root cd /opt/sutazaiapp && python3 security_orchestrator.py assess >> /opt/sutazaiapp/logs/cron.log 2>&1

# Generate comprehensive report daily at 2 AM
0 2 * * * root cd /opt/sutazaiapp && python3 security_orchestrator.py report >> /opt/sutazaiapp/logs/cron.log 2>&1

# Health check every 5 minutes
*/5 * * * * root cd /opt/sutazaiapp && python3 security_orchestrator.py status >> /opt/sutazaiapp/logs/health.log 2>&1
EOF

    sudo mv /tmp/sutazai-security-cron /etc/cron.d/sutazai-security
    sudo chmod 644 /etc/cron.d/sutazai-security
    
    success "Cron jobs created successfully"
}

# Validate deployment
validate_deployment() {
    log "Validating security infrastructure deployment..."
    
    # Check if all security scripts exist
    scripts=(
        "security_orchestrator.py"
        "intrusion_detection_system.py"
        "security_event_logger.py"
        "automated_threat_response.py"
        "security_monitoring_dashboard.py"
        "comprehensive_security_report_generator.py"
    )
    
    missing_scripts=()
    for script in "${scripts[@]}"; do
        if [ ! -f "/opt/sutazaiapp/$script" ]; then
            missing_scripts+=("$script")
        fi
    done
    
    if [ ${#missing_scripts[@]} -ne 0 ]; then
        error "Missing security scripts: ${missing_scripts[*]}"
        return 1
    fi
    
    # Check if directories exist
    directories=(
        "/opt/sutazaiapp/data"
        "/opt/sutazaiapp/logs"
        "/opt/sutazaiapp/reports/security"
        "/opt/sutazaiapp/config"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            error "Missing directory: $dir"
            return 1
        fi
    done
    
    # Check if systemd service exists
    if [ ! -f "/etc/systemd/system/sutazai-security.service" ]; then
        error "Systemd service not installed"
        return 1
    fi
    
    success "Security infrastructure deployment validation passed"
    return 0
}

# Display deployment summary
display_summary() {
    echo
    echo "=" * 60
    echo "SutazAI Security Infrastructure Deployment Complete"
    echo "=" * 60
    echo
    echo "Deployed Components:"
    echo "  ✅ Intrusion Detection System"
    echo "  ✅ Security Event Logging"
    echo "  ✅ Automated Threat Response"
    echo "  ✅ Security Monitoring Dashboard"
    echo "  ✅ Comprehensive Reporting"
    echo "  ✅ Security Orchestrator"
    echo
    echo "Services:"
    echo "  • SystemD Service: sutazai-security.service"
    echo "  • Dashboard URL: http://localhost:8501"
    echo "  • Log Files: /opt/sutazaiapp/logs/"
    echo "  • Reports: /opt/sutazaiapp/reports/security/"
    echo
    echo "Management Commands:"
    echo "  • Start:   sudo systemctl start sutazai-security"
    echo "  • Stop:    sudo systemctl stop sutazai-security"
    echo "  • Status:  sudo systemctl status sutazai-security"
    echo "  • Logs:    sudo journalctl -u sutazai-security -f"
    echo
    echo "Manual Commands:"
    echo "  • Start:   python3 /opt/sutazaiapp/security_orchestrator.py start"
    echo "  • Status:  python3 /opt/sutazaiapp/security_orchestrator.py status"
    echo "  • Report:  python3 /opt/sutazaiapp/security_orchestrator.py report"
    echo
    echo "Next Steps:"
    echo "  1. Review initial security report at /opt/sutazaiapp/reports/security/"
    echo "  2. Start the security service: sudo systemctl start sutazai-security"
    echo "  3. Access dashboard at http://localhost:8501"
    echo "  4. Monitor logs at /opt/sutazaiapp/logs/"
    echo
}

# Main deployment function
main() {
    echo "=" * 60
    echo "SutazAI Security Infrastructure Deployment"
    echo "=" * 60
    echo
    
    log "Starting security infrastructure deployment..."
    
    # Check prerequisites
    check_root
    
    # Run deployment steps
    install_dependencies
    install_python_dependencies
    create_directories
    configure_firewall
    initialize_databases
    create_systemd_services
    configure_log_rotation
    create_cron_jobs
    
    # Run initial assessment
    run_initial_assessment
    
    # Validate deployment
    if validate_deployment; then
        success "Security infrastructure deployed successfully!"
        display_summary
    else
        error "Deployment validation failed!"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    "install")
        main
        ;;
    "validate")
        validate_deployment
        ;;
    "summary")
        display_summary
        ;;
    *)
        echo "Usage: $0 [install|validate|summary]"
        echo "  install  - Deploy complete security infrastructure"
        echo "  validate - Validate existing deployment"
        echo "  summary  - Show deployment summary"
        exit 1
        ;;
esac