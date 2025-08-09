#!/bin/bash
# SutazAI Chaos Engineering Framework Initialization
# This script sets up the chaos engineering environment

set -euo pipefail

# Configuration
CHAOS_DIR="/opt/sutazaiapp/chaos"
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/chaos_init_${TIMESTAMP}.log"

# Ensure directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$CHAOS_DIR"/{config,experiments,scripts,monitoring,reports,tools}

# Logging functions
log_info() {
    echo "[$(date +'%H:%M:%S')] INFO: $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[$(date +'%H:%M:%S')] SUCCESS: $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE"
}

log_header() {
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log_header "Checking Chaos Engineering Prerequisites"
    
    local required_commands=("docker" "docker-compose" "python3" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            log_success "$cmd is available"
        else
            log_error "$cmd is not installed"
            exit 1
        fi
    done
    
    # Check if Docker is running
    if docker info &> /dev/null; then
        log_success "Docker daemon is running"
    else
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Python dependencies
    local python_deps=("yaml" "requests" "docker" "prometheus_client")
    for dep in "${python_deps[@]}"; do
        if python3 -c "import $dep" &> /dev/null; then
            log_success "Python module $dep is available"
        else
            log_info "Installing Python module: $dep"
            pip3 install "$dep" || {
                log_error "Failed to install $dep"
                exit 1
            }
        fi
    done
}

# Initialize chaos tools
initialize_chaos_tools() {
    log_header "Initializing Chaos Engineering Tools"
    
    # Install stress-ng for resource stress testing
    if command -v stress-ng &> /dev/null; then
        log_success "stress-ng is already installed"
    else
        log_info "Installing stress-ng"
        apt-get update && apt-get install -y stress-ng || {
            log_error "Failed to install stress-ng"
            exit 1
        }
    fi
    
    # Install tc (traffic control) for network chaos
    if command -v tc &> /dev/null; then
        log_success "tc (traffic control) is available"
    else
        log_info "Installing iproute2 for traffic control"
        apt-get install -y iproute2 || {
            log_error "Failed to install iproute2"
            exit 1
        }
    fi
    
    # Install pumba for container chaos (if not using custom implementation)
    if command -v pumba &> /dev/null; then
        log_success "pumba is already installed"
    else
        log_info "Installing pumba for container chaos"
        curl -L https://github.com/alexei-led/pumba/releases/download/0.9.0/pumba_linux_amd64 \
            -o /usr/local/bin/pumba
        chmod +x /usr/local/bin/pumba
        log_success "pumba installed successfully"
    fi
}

# Setup monitoring integration
setup_monitoring() {
    log_header "Setting up Chaos Monitoring Integration"
    
    # Create Prometheus configuration for chaos metrics
    cat > "$CHAOS_DIR/monitoring/chaos-prometheus.yml" << 'EOF'
# Chaos Engineering Prometheus Configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "chaos-rules.yml"

scrape_configs:
  - job_name: 'chaos-experiments'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'chaos-targets'
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        port: 8080
    relabel_configs:
      - source_labels: [__meta_docker_container_label_chaos_target]
        action: keep
        regex: true
EOF

    # Create Grafana dashboard for chaos experiments
    cat > "$CHAOS_DIR/monitoring/chaos-dashboard.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "SutazAI Chaos Engineering",
    "tags": ["chaos", "resilience", "sutazai"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Active Experiments",
        "type": "stat",
        "targets": [
          {
            "expr": "chaos_experiments_active",
            "format": "time_series"
          }
        ]
      },
      {
        "title": "Experiment Success Rate",
        "type": "stat", 
        "targets": [
          {
            "expr": "rate(chaos_experiments_success_total[5m]) / rate(chaos_experiments_total[5m]) * 100",
            "format": "time_series"
          }
        ]
      },
      {
        "title": "Mean Recovery Time",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(chaos_recovery_time_seconds)",
            "format": "time_series"
          }
        ]
      },
      {
        "title": "Service Health Impact",
        "type": "graph",
        "targets": [
          {
            "expr": "chaos_service_health_score",
            "format": "time_series",
            "legendFormat": "{{service}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    log_success "Monitoring configuration created"
}

# Create experiment templates
create_experiment_templates() {
    log_header "Creating Chaos Experiment Templates"
    
    # Basic container chaos experiment
    cat > "$CHAOS_DIR/experiments/basic-container-chaos.yaml" << 'EOF'
apiVersion: chaos.sutazai.com/v1
kind: ChaosExperiment
metadata:
  name: basic-container-chaos
  description: "Basic container failure and recovery test"
spec:
  duration: "10m"
  schedule: "manual"
  safety:
    safe_mode: true
    max_affected_services: 2
  
  targets:
    - service: "sutazai-autogpt"
      weight: 0.3
    - service: "sutazai-crewai" 
      weight: 0.3
    - service: "sutazai-letta"
      weight: 0.2
  
  scenarios:
    - name: "container_kill"
      probability: 0.5
      actions:
        - type: "kill"
          target: "random"
          recovery_check: true
    
    - name: "container_restart"
      probability: 0.3
      actions:
        - type: "restart"
          target: "random"
          wait_for_healthy: true
    
    - name: "container_pause"
      probability: 0.2
      actions:
        - type: "pause"
          target: "random"
          duration: "60s"
  
  recovery:
    automatic: true
    timeout: "5m"
    health_check: true
  
  monitoring:
    metrics: true
    logging: true
    alerts: true
EOF

    # Network chaos experiment
    cat > "$CHAOS_DIR/experiments/network-chaos.yaml" << 'EOF'
apiVersion: chaos.sutazai.com/v1
kind: ChaosExperiment
metadata:
  name: network-chaos
  description: "Network latency and partition testing"
spec:
  duration: "15m"
  schedule: "manual"
  safety:
    safe_mode: true
    max_affected_services: 3
  
  targets:
    - service: "sutazai-backend"
      network: "sutazai-network"
    - service: "sutazai-redis"
      network: "sutazai-network"
    - service: "sutazai-chromadb"
      network: "sutazai-network"
  
  scenarios:
    - name: "network_latency"
      probability: 0.4
      actions:
        - type: "delay"
          latency: "100ms"
          jitter: "10ms"
          duration: "5m"
    
    - name: "packet_loss"
      probability: 0.3
      actions:
        - type: "loss"
          percentage: "5%"
          duration: "3m"
    
    - name: "bandwidth_limit"
      probability: 0.2
      actions:
        - type: "rate"
          bandwidth: "1mbps"
          duration: "2m"
    
    - name: "network_partition"
      probability: 0.1
      actions:
        - type: "partition"
          duration: "60s"
          targets: 2
  
  recovery:
    automatic: true
    timeout: "3m"
    network_reset: true
EOF

    # Resource stress experiment
    cat > "$CHAOS_DIR/experiments/resource-stress.yaml" << 'EOF'
apiVersion: chaos.sutazai.com/v1
kind: ChaosExperiment
metadata:
  name: resource-stress
  description: "CPU, memory, and disk stress testing"
spec:
  duration: "12m"
  schedule: "manual"
  safety:
    safe_mode: true
    max_affected_services: 2
  
  targets:
    - service: "sutazai-ollama"
      resources: ["cpu", "memory"]
    - service: "sutazai-backend"
      resources: ["cpu", "memory", "disk"]
  
  scenarios:
    - name: "cpu_stress"
      probability: 0.4
      actions:
        - type: "cpu"
          percentage: 80
          duration: "5m"
    
    - name: "memory_stress"
      probability: 0.3
      actions:
        - type: "memory"
          percentage: 75
          duration: "3m"
    
    - name: "disk_stress"
      probability: 0.2
      actions:
        - type: "disk"
          io_percentage: 70
          duration: "2m"
    
    - name: "combined_stress"
      probability: 0.1
      actions:
        - type: "combined"
          cpu: 60
          memory: 60
          duration: "4m"
  
  recovery:
    automatic: true
    timeout: "2m"
    resource_cleanup: true
EOF

    log_success "Experiment templates created"
}

# Setup chaos cron jobs
setup_scheduling() {
    log_header "Setting up Chaos Experiment Scheduling"
    
    # Create cron job for scheduled chaos experiments
    cat > "/etc/cron.d/sutazai-chaos" << 'EOF'
# SutazAI Chaos Engineering Scheduled Experiments
# Run during maintenance windows: 2-4 AM on Mon, Wed, Fri

# Basic container chaos - Monday 2:30 AM
30 2 * * 1 root /opt/sutazaiapp/chaos/scripts/run-experiment.sh --experiment basic-container-chaos --safe-mode

# Network chaos - Wednesday 2:30 AM  
30 2 * * 3 root /opt/sutazaiapp/chaos/scripts/run-experiment.sh --experiment network-chaos --safe-mode

# Resource stress - Friday 2:30 AM
30 2 * * 5 root /opt/sutazaiapp/chaos/scripts/run-experiment.sh --experiment resource-stress --safe-mode

# Cleanup old reports - Daily at 1:00 AM
0 1 * * * root /opt/sutazaiapp/chaos/scripts/cleanup-reports.sh
EOF

    log_success "Chaos scheduling configured"
}

# Set permissions
set_permissions() {
    log_header "Setting Chaos Framework Permissions"
    
    # Make scripts executable
    find "$CHAOS_DIR/scripts" -name "*.sh" -exec chmod +x {} \;
    
    # Set proper ownership
    chown -R root:docker "$CHAOS_DIR" 2>/dev/null || true
    
    # Set log directory permissions
    chmod 755 "$LOG_DIR"
    
    log_success "Permissions configured"
}

# Validate installation
validate_installation() {
    log_header "Validating Chaos Framework Installation"
    
    # Check required files
    local required_files=(
        "$CHAOS_DIR/config/chaos-config.yaml"
        "$CHAOS_DIR/experiments/basic-container-chaos.yaml"
        "$CHAOS_DIR/experiments/network-chaos.yaml"
        "$CHAOS_DIR/experiments/resource-stress.yaml"
        "$CHAOS_DIR/monitoring/chaos-prometheus.yml"
        "$CHAOS_DIR/monitoring/chaos-dashboard.json"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "File exists: $file"
        else
            log_error "Missing file: $file"
            exit 1
        fi
    done
    
    # Validate configuration
    if python3 -c "import yaml; yaml.safe_load(open('$CHAOS_DIR/config/chaos-config.yaml'))" &> /dev/null; then
        log_success "Configuration file is valid YAML"
    else
        log_error "Configuration file has invalid YAML syntax"
        exit 1
    fi
    
    log_success "Chaos framework validation completed"
}

# Main execution
main() {
    log_header "Initializing SutazAI Chaos Engineering Framework"
    
    check_prerequisites
    initialize_chaos_tools
    setup_monitoring
    create_experiment_templates
    setup_scheduling
    set_permissions
    validate_installation
    
    log_header "Chaos Engineering Framework Initialization Complete"
    log_success "Framework ready for chaos experiments"
    log_info "Run './scripts/run-experiment.sh --help' for usage information"
    log_info "View configuration: $CHAOS_DIR/config/chaos-config.yaml"
    log_info "Available experiments: $CHAOS_DIR/experiments/"
}

# Execute main function
main "$@"