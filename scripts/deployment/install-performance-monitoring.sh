#!/bin/bash
# SutazAI Performance Monitoring Installation Script
# =================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/performance_monitoring_install_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Logging setup
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo "=================================================="
echo "SutazAI Performance Monitoring Installation"
echo "Started: $(date)"
echo "=================================================="

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python packages
install_python_packages() {
    log "Installing Python dependencies..."
    
    # Core dependencies
    pip3 install --user --upgrade \
        psutil \
        docker \
        requests \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        pyyaml \
        asyncio \
        aiohttp \
        sqlite3 \
        jinja2 \
        flask \
        flask-socketio \
        prometheus-client
    
    # Machine learning dependencies (optional)
    log "Installing machine learning dependencies..."
    pip3 install --user --upgrade \
        scikit-learn \
        statsmodels || log "WARNING: Failed to install some ML packages"
    
    # Prophet for forecasting (optional)
    pip3 install --user --upgrade prophet || log "WARNING: Failed to install Prophet"
    
    # TensorFlow for LSTM models (optional)
    pip3 install --user --upgrade tensorflow || log "WARNING: Failed to install TensorFlow"
    
    # PDF generation (optional)
    pip3 install --user --upgrade weasyprint || log "WARNING: Failed to install WeasyPrint"
    
    log "Python dependencies installation completed"
}

# Function to setup directories
setup_directories() {
    log "Setting up directory structure..."
    
    mkdir -p "$PROJECT_ROOT/data"
    mkdir -p "$PROJECT_ROOT/reports/performance/charts"
    mkdir -p "$PROJECT_ROOT/config"
    mkdir -p "$PROJECT_ROOT/monitoring/templates"
    
    # Set proper permissions
    chmod 755 "$PROJECT_ROOT/scripts/run-performance-benchmark.sh"
    chmod 755 "$PROJECT_ROOT/monitoring/system_performance_benchmark_suite.py"
    chmod 755 "$PROJECT_ROOT/monitoring/performance_forecasting_models.py"
    chmod 755 "$PROJECT_ROOT/monitoring/continuous_performance_monitor.py"
    chmod 755 "$PROJECT_ROOT/monitoring/comprehensive_report_generator.py"
    
    log "Directory structure setup completed"
}

# Function to initialize database
initialize_database() {
    log "Initializing performance metrics database..."
    
    python3 -c "
import sqlite3
import sys
sys.path.append('$PROJECT_ROOT')

conn = sqlite3.connect('$PROJECT_ROOT/data/performance_metrics.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS benchmark_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        component TEXT,
        category TEXT,
        metric_name TEXT,
        value REAL,
        unit TEXT,
        metadata TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        cpu_percent REAL,
        memory_percent REAL,
        memory_used_gb REAL,
        memory_available_gb REAL,
        disk_io_read_mb REAL,
        disk_io_write_mb REAL,
        network_bytes_sent REAL,
        network_bytes_recv REAL,
        active_containers INTEGER,
        running_agents INTEGER
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS performance_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        component TEXT,
        metric_name TEXT,
        value REAL,
        unit TEXT,
        tags TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS performance_alerts (
        id TEXT PRIMARY KEY,
        timestamp DATETIME,
        severity TEXT,
        component TEXT,
        metric TEXT,
        current_value REAL,
        threshold_value REAL,
        message TEXT,
        acknowledged INTEGER DEFAULT 0,
        resolved INTEGER DEFAULT 0
    )
''')

conn.commit()
conn.close()
print('Database initialized successfully')
"
    
    log "Database initialization completed"
}

# Function to create systemd service
create_systemd_service() {
    log "Creating systemd service for continuous monitoring..."
    
    cat > "/tmp/sutazai-performance-monitor.service" << EOF
[Unit]
Description=SutazAI Performance Monitor
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
ExecStart=/usr/bin/python3 $PROJECT_ROOT/monitoring/continuous_performance_monitor.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    if [ "$EUID" -eq 0 ]; then
        mv "/tmp/sutazai-performance-monitor.service" "/etc/systemd/system/"
        systemctl daemon-reload
        systemctl enable sutazai-performance-monitor.service
        log "Systemd service created and enabled"
    else
        log "WARNING: Not running as root, systemd service not installed"
        log "To install service manually, run as root:"
        log "  sudo mv /tmp/sutazai-performance-monitor.service /etc/systemd/system/"
        log "  sudo systemctl daemon-reload"
        log "  sudo systemctl enable sutazai-performance-monitor.service"
    fi
}

# Function to create cron jobs
setup_cron_jobs() {
    log "Setting up cron jobs for automated reporting..."
    
    # Create cron job for daily reports
    CRON_JOB="0 2 * * * cd $PROJECT_ROOT && python3 monitoring/comprehensive_report_generator.py --type html --days 7 >> logs/cron_reports.log 2>&1"
    
    # Add cron job if it doesn't exist
    if ! crontab -l 2>/dev/null | grep -q "comprehensive_report_generator.py"; then
        (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
        log "Daily report cron job added"
    else
        log "Daily report cron job already exists"
    fi
    
    # Create cron job for weekly comprehensive benchmarks
    BENCHMARK_CRON="0 1 * * 0 cd $PROJECT_ROOT && ./scripts/run-performance-benchmark.sh --quick >> logs/cron_benchmarks.log 2>&1"
    
    if ! crontab -l 2>/dev/null | grep -q "run-performance-benchmark.sh"; then
        (crontab -l 2>/dev/null; echo "$BENCHMARK_CRON") | crontab -
        log "Weekly benchmark cron job added"
    else
        log "Weekly benchmark cron job already exists"
    fi
}

# Function to validate installation
validate_installation() {
    log "Validating installation..."
    
    # Check Python modules
    python3 -c "
import sys
sys.path.append('$PROJECT_ROOT')

try:
    from monitoring.system_performance_benchmark_suite import SystemPerformanceBenchmarkSuite
    print('✓ Benchmark suite module loaded successfully')
except ImportError as e:
    print(f'✗ Failed to load benchmark suite: {e}')

try:
    from monitoring.performance_forecasting_models import PerformanceForecastingSystem
    print('✓ Forecasting models module loaded successfully')
except ImportError as e:
    print(f'✗ Failed to load forecasting models: {e}')

try:
    from monitoring.continuous_performance_monitor import ContinuousPerformanceMonitor
    print('✓ Continuous monitor module loaded successfully')
except ImportError as e:
    print(f'✗ Failed to load continuous monitor: {e}')

try:
    from monitoring.comprehensive_report_generator import ComprehensiveReportGenerator
    print('✓ Report generator module loaded successfully')
except ImportError as e:
    print(f'✗ Failed to load report generator: {e}')
"
    
    # Check database
    if [ -f "$PROJECT_ROOT/data/performance_metrics.db" ]; then
        log "✓ Performance metrics database exists"
    else
        log "✗ Performance metrics database not found"
    fi
    
    # Check configuration
    if [ -f "$PROJECT_ROOT/config/benchmark_config.yaml" ]; then
        log "✓ Benchmark configuration exists"
    else
        log "✗ Benchmark configuration not found"
    fi
    
    # Check scripts
    if [ -x "$PROJECT_ROOT/scripts/run-performance-benchmark.sh" ]; then
        log "✓ Benchmark execution script is executable"
    else
        log "✗ Benchmark execution script not executable"
    fi
    
    log "Installation validation completed"
}

# Function to display usage information
display_usage_info() {
    log "Installation completed successfully!"
    
    cat << EOF

================================================
SutazAI Performance Monitoring - Usage Guide
================================================

1. Run One-Time Benchmark:
   ./scripts/run-performance-benchmark.sh

2. Run Quick Benchmark:
   ./scripts/run-performance-benchmark.sh --quick

3. Start Continuous Monitoring:
   sudo systemctl start sutazai-performance-monitor

4. Generate Performance Report:
   python3 monitoring/comprehensive_report_generator.py --type html --days 7

5. Access Performance Dashboard:
   Start the continuous monitor and visit http://localhost:5000

6. View Reports:
   Reports are generated in: $PROJECT_ROOT/reports/performance/

7. Check Logs:
   Installation log: $LOG_FILE
   Monitor logs: $PROJECT_ROOT/logs/
   Cron logs: $PROJECT_ROOT/logs/cron_*.log

Configuration Files:
- Main config: $PROJECT_ROOT/config/benchmark_config.yaml
- SLA thresholds: Defined in benchmark_config.yaml
- Database: $PROJECT_ROOT/data/performance_metrics.db

Automation:
- Daily reports: 02:00 AM (via cron)
- Weekly benchmarks: 01:00 AM Sunday (via cron)
- Continuous monitoring: Available as systemd service

For support and documentation, check:
$PROJECT_ROOT/docs/

================================================
EOF
}

# Main installation process
main() {
    log "Starting SutazAI Performance Monitoring installation"
    
    # Check prerequisites
    log "Checking prerequisites..."
    if ! command_exists python3; then
        log "ERROR: Python3 is required but not installed"
        exit 1
    fi
    
    if ! command_exists pip3; then
        log "ERROR: pip3 is required but not installed"
        exit 1
    fi
    
    if ! command_exists docker; then
        log "ERROR: Docker is required but not installed"
        exit 1
    fi
    
    # Installation steps
    setup_directories
    install_python_packages
    initialize_database
    create_systemd_service
    setup_cron_jobs
    validate_installation
    display_usage_info
    
    log "Installation completed successfully!"
}

# Handle script arguments
case "${1:-install}" in
    install)
        main
        ;;
    validate)
        validate_installation
        ;;
    uninstall)
        log "Uninstalling SutazAI Performance Monitoring..."
        
        # Stop and disable service
        if systemctl is-active --quiet sutazai-performance-monitor; then
            sudo systemctl stop sutazai-performance-monitor
        fi
        
        if systemctl is-enabled --quiet sutazai-performance-monitor 2>/dev/null; then
            sudo systemctl disable sutazai-performance-monitor
        fi
        
        # Remove service file
        if [ -f "/etc/systemd/system/sutazai-performance-monitor.service" ]; then
            sudo rm -f "/etc/systemd/system/sutazai-performance-monitor.service"
            sudo systemctl daemon-reload
        fi
        
        # Remove cron jobs
        crontab -l 2>/dev/null | grep -v "comprehensive_report_generator.py" | grep -v "run-performance-benchmark.sh" | crontab - || true
        
        log "Uninstallation completed"
        ;;
    *)
        echo "Usage: $0 [install|validate|uninstall]"
        echo "  install   - Install performance monitoring system (default)"
        echo "  validate  - Validate existing installation"
        echo "  uninstall - Remove performance monitoring system"
        exit 1
        ;;
esac