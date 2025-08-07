#!/bin/bash
"""
SutazAI Monitoring System Startup Script
Purpose: Start comprehensive monitoring of SutazAI deployment
Usage: ./start_monitoring.sh [--background] [--interval SECONDS]
"""

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/opt/sutazaiapp/logs"
MONITORING_INTERVAL=10
BACKGROUND_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --background|-b)
            BACKGROUND_MODE=true
            shift
            ;;
        --interval|-i)
            MONITORING_INTERVAL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--background] [--interval SECONDS]"
            echo "Options:"
            echo "  --background, -b    Run monitoring in background"
            echo "  --interval, -i      Monitoring interval in seconds (default: 10)"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to print colored output
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

# Function to check dependencies
check_dependencies() {
    log_info "Checking monitoring dependencies..."
    
    local missing_deps=()
    
    # Check Python packages
    python3 -c "import docker, psutil, requests, prometheus_client" 2>/dev/null || missing_deps+=("python-deps")
    
    # Check Docker
    docker --version >/dev/null 2>&1 || missing_deps+=("docker")
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Installing missing Python dependencies..."
        pip3 install docker psutil requests prometheus-client psycopg2-binary redis neo4j
        
        # Verify installation
        python3 -c "import docker, psutil, requests, prometheus_client" 2>/dev/null || {
            log_error "Failed to install Python dependencies"
            exit 1
        }
    fi
    
    log_success "All dependencies are available"
}

# Function to check current deployment status
check_deployment_status() {
    log_info "Checking current deployment status..."
    
    # Check if key services are running
    local running_containers=$(docker ps --format "table {{.Names}}" | grep -E "(consul|kong|rabbitmq|neo4j|postgres|redis)" | wc -l)
    
    if [[ $running_containers -lt 5 ]]; then
        log_warning "Only $running_containers core containers are running. Expected at least 5."
        log_info "Running containers:"
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(consul|kong|rabbitmq|neo4j|postgres|redis)" || true
    else
        log_success "$running_containers core containers are running"
    fi
}

# Function to run initial health check
run_initial_health_check() {
    log_info "Running initial service health check..."
    
    python3 "$SCRIPT_DIR/service_health_checker.py" --save || {
        log_warning "Initial health check completed with some issues"
    }
}

# Function to start real-time monitoring
start_realtime_monitoring() {
    log_info "Starting real-time monitoring (interval: ${MONITORING_INTERVAL}s)..."
    
    if [[ "$BACKGROUND_MODE" == "true" ]]; then
        # Run in background
        nohup python3 "$SCRIPT_DIR/sutazai_realtime_monitor.py" \
            --interval "$MONITORING_INTERVAL" \
            > "$LOG_DIR/monitoring_output.log" 2>&1 &
        
        local pid=$!
        echo $pid > "$LOG_DIR/monitoring.pid"
        
        log_success "Real-time monitoring started in background (PID: $pid)"
        log_info "Monitor logs: tail -f $LOG_DIR/monitoring_output.log"
        log_info "Stop monitoring: kill $pid"
    else
        # Run in foreground
        log_info "Starting real-time monitoring in foreground..."
        log_info "Press Ctrl+C to stop monitoring"
        
        python3 "$SCRIPT_DIR/sutazai_realtime_monitor.py" \
            --interval "$MONITORING_INTERVAL"
    fi
}

# Function to stop existing monitoring
stop_existing_monitoring() {
    if [[ -f "$LOG_DIR/monitoring.pid" ]]; then
        local pid=$(cat "$LOG_DIR/monitoring.pid")
        if ps -p $pid > /dev/null 2>&1; then
            log_info "Stopping existing monitoring process (PID: $pid)"
            kill $pid
            sleep 2
            if ps -p $pid > /dev/null 2>&1; then
                log_warning "Force killing monitoring process"
                kill -9 $pid
            fi
        fi
        rm -f "$LOG_DIR/monitoring.pid"
    fi
}

# Function to create monitoring dashboard
create_monitoring_dashboard() {
    log_info "Creating monitoring dashboard..."
    
    cat > "$LOG_DIR/monitoring_dashboard.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SutazAI Monitoring Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .status-item { padding: 15px; border-radius: 6px; }
        .healthy { background-color: #d4edda; border-left: 4px solid #28a745; }
        .unhealthy { background-color: #f8d7da; border-left: 4px solid #dc3545; }
        .warning { background-color: #fff3cd; border-left: 4px solid #ffc107; }
        .metric { display: inline-block; margin: 5px 10px; padding: 5px 10px; background: #e9ecef; border-radius: 4px; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 SutazAI Real-Time Monitoring Dashboard</h1>
            <p>Live monitoring of all SutazAI services and infrastructure</p>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
        </div>
        
        <div class="card">
            <h2>📊 Quick Status</h2>
            <p>Last updated: <span id="timestamp">Loading...</span></p>
            <div id="status-overview">Loading monitoring data...</div>
        </div>
        
        <div class="card">
            <h2>🔗 Monitoring Links</h2>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                <a href="http://localhost:10050" target="_blank" class="metric">Grafana Dashboard</a>
                <a href="http://localhost:10006" target="_blank" class="metric">Consul UI</a>
                <a href="http://localhost:10042" target="_blank" class="metric">RabbitMQ Management</a>
                <a href="http://localhost:10002" target="_blank" class="metric">Neo4j Browser</a>
                <a href="http://localhost:10420" target="_blank" class="metric">Hygiene Backend</a>
            </div>
        </div>
        
        <div class="card">
            <h2>📝 Recent Logs</h2>
            <pre id="recent-logs">Loading logs...</pre>
        </div>
    </div>

    <script>
        // Auto-refresh every 30 seconds
        setInterval(() => {
            location.reload();
        }, 30000);
        
        // Load monitoring data
        async function loadMonitoringData() {
            try {
                const response = await fetch('/opt/sutazaiapp/logs/monitoring_report.json');
                const data = await response.json();
                
                document.getElementById('timestamp').textContent = data.timestamp;
                
                let statusHtml = '<div class="status-grid">';
                for (const [service, status] of Object.entries(data.service_mesh || {})) {
                    const statusClass = status.healthy ? 'healthy' : 'unhealthy';
                    statusHtml += `<div class="status-item ${statusClass}">
                        <strong>${service}</strong><br>
                        Status: ${status.status_message}<br>
                        Response: ${(status.response_time * 1000).toFixed(1)}ms
                    </div>`;
                }
                statusHtml += '</div>';
                
                document.getElementById('status-overview').innerHTML = statusHtml;
            } catch (error) {
                document.getElementById('status-overview').innerHTML = '<p style="color: red;">Error loading monitoring data</p>';
            }
        }
        
        // Load on page load
        loadMonitoringData();
    </script>
</body>
</html>
EOF
    
    log_success "Monitoring dashboard created at: $LOG_DIR/monitoring_dashboard.html"
}

# Main execution
main() {
    echo "🔍 SutazAI Monitoring System Startup"
    echo "====================================="
    
    # Ensure log directory exists
    mkdir -p "$LOG_DIR"
    
    # Check dependencies
    check_dependencies
    
    # Check deployment status
    check_deployment_status
    
    # Stop any existing monitoring
    if [[ "$BACKGROUND_MODE" == "true" ]]; then
        stop_existing_monitoring
    fi
    
    # Run initial health check
    run_initial_health_check
    
    # Create monitoring dashboard
    create_monitoring_dashboard
    
    # Start real-time monitoring
    start_realtime_monitoring
}

# Cleanup function
cleanup() {
    log_info "Shutting down monitoring..."
    if [[ -f "$LOG_DIR/monitoring.pid" ]]; then
        local pid=$(cat "$LOG_DIR/monitoring.pid")
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid
        fi
        rm -f "$LOG_DIR/monitoring.pid"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Run main function
main "$@"