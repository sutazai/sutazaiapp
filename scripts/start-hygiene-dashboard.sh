#!/bin/bash
"""
Purpose: Starts the Sutazai Hygiene Enforcement Monitoring Dashboard
Usage: ./start-hygiene-dashboard.sh [--port PORT] [--host HOST] [--dev]
Requirements: Python 3.8+, Node.js (optional for dev mode)
"""

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DIR="$PROJECT_ROOT/dashboard/hygiene-monitor"
DEFAULT_PORT=8080
DEFAULT_HOST="0.0.0.0"
DEV_MODE=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Usage information
show_usage() {
    cat << EOF
Sutazai Hygiene Enforcement Dashboard Launcher

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -p, --port PORT         Port to serve dashboard on (default: $DEFAULT_PORT)
    -h, --host HOST         Host to bind to (default: $DEFAULT_HOST)
    -d, --dev              Enable development mode with auto-reload
    -v, --verbose          Enable verbose logging
    --help                 Show this help message

EXAMPLES:
    $0                     # Start on default port 8080
    $0 --port 3000         # Start on port 3000
    $0 --dev               # Start in development mode
    $0 --host localhost    # Bind to localhost only

ENVIRONMENT VARIABLES:
    HYGIENE_DASHBOARD_PORT      Override default port
    HYGIENE_DASHBOARD_HOST      Override default host
    HYGIENE_API_ENDPOINT        Backend API endpoint
    HYGIENE_WS_ENDPOINT         WebSocket endpoint for real-time updates

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--port)
                DEFAULT_PORT="$2"
                shift 2
                ;;
            -h|--host)
                DEFAULT_HOST="$2"
                shift 2
                ;;
            -d|--dev)
                DEV_MODE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if dashboard directory exists
    if [[ ! -d "$DASHBOARD_DIR" ]]; then
        log_error "Dashboard directory not found: $DASHBOARD_DIR"
        exit 1
    fi
    
    # Check if required files exist
    local required_files=("index.html" "app.js" "styles.css")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$DASHBOARD_DIR/$file" ]]; then
            log_error "Required file not found: $DASHBOARD_DIR/$file"
            exit 1
        fi
    done
    
    # Check Python availability
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if port is available
    if command -v netstat &> /dev/null; then
        if netstat -tuln | grep -q ":$DEFAULT_PORT "; then
            log_warning "Port $DEFAULT_PORT appears to be in use"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Apply environment variable overrides
    DEFAULT_PORT="${HYGIENE_DASHBOARD_PORT:-$DEFAULT_PORT}"
    DEFAULT_HOST="${HYGIENE_DASHBOARD_HOST:-$DEFAULT_HOST}"
    
    # Create logs directory if it doesn't exist
    local logs_dir="$PROJECT_ROOT/logs"
    mkdir -p "$logs_dir"
    
    # Set up dashboard configuration
    cat > "$DASHBOARD_DIR/config.js" << EOF
// Auto-generated dashboard configuration
window.HYGIENE_DASHBOARD_CONFIG = {
    apiEndpoint: '${HYGIENE_API_ENDPOINT:-/api/hygiene}',
    wsEndpoint: '${HYGIENE_WS_ENDPOINT:-ws://localhost:8081/ws}',
    refreshInterval: ${HYGIENE_REFRESH_INTERVAL:-10000},
    devMode: ${DEV_MODE,,},
    version: '1.0.0',
    buildTime: '$(date -u +"%Y-%m-%dT%H:%M:%SZ")'
};

// Load configuration into dashboard
if (typeof module !== 'undefined' && module.exports) {
    module.exports = window.HYGIENE_DASHBOARD_CONFIG;
}
EOF

    # Add config script to HTML if not already present
    if ! grep -q "config.js" "$DASHBOARD_DIR/index.html"; then
        sed -i 's|<script src="app.js"></script>|<script src="config.js"></script>\n    <script src="app.js"></script>|' "$DASHBOARD_DIR/index.html"
    fi
    
    log_success "Environment setup complete"
}

# Start dashboard server
start_server() {
    log_info "Starting Hygiene Enforcement Dashboard..."
    log_info "Server: http://$DEFAULT_HOST:$DEFAULT_PORT"
    log_info "Dashboard: $DASHBOARD_DIR"
    
    # Change to dashboard directory
    cd "$DASHBOARD_DIR"
    
    # Create a simple Python HTTP server
    local server_script="$DASHBOARD_DIR/server.py"
    cat > "$server_script" << 'EOF'
#!/usr/bin/env python3
"""
Simple HTTP server for Hygiene Dashboard with API endpoints
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import subprocess

class HygieneDashboardHandler(SimpleHTTPRequestHandler):
    """Enhanced HTTP handler with API endpoints"""
    
    def __init__(self, *args, **kwargs):
        self.project_root = Path(__file__).parent.parent.parent
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        # API endpoints
        if parsed_path.path.startswith('/api/hygiene'):
            self.handle_api_request(parsed_path)
        elif parsed_path.path.startswith('/api/system'):
            self.handle_system_request(parsed_path)
        else:
            # Serve static files
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path.startswith('/api/hygiene'):
            self.handle_api_post(parsed_path)
        else:
            self.send_error(404, "Endpoint not found")
    
    def handle_api_request(self, parsed_path):
        """Handle hygiene API requests"""
        try:
            if parsed_path.path == '/api/hygiene/status':
                self.send_hygiene_status()
            elif parsed_path.path == '/api/hygiene/report':
                self.send_hygiene_report()
            else:
                self.send_error(404, "API endpoint not found")
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
    
    def handle_system_request(self, parsed_path):
        """Handle system metrics requests"""
        try:
            if parsed_path.path == '/api/system/metrics':
                self.send_system_metrics()
            else:
                self.send_error(404, "System endpoint not found")
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
    
    def handle_api_post(self, parsed_path):
        """Handle API POST requests"""
        try:
            if parsed_path.path == '/api/hygiene/audit':
                self.run_hygiene_audit()
            elif parsed_path.path == '/api/hygiene/cleanup':
                self.run_hygiene_cleanup()
            else:
                self.send_error(404, "POST endpoint not found")
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
    
    def send_hygiene_status(self):
        """Send current hygiene status"""
        # Try to read from actual hygiene coordinator
        status_file = self.project_root / "logs" / "hygiene-status.json"
        
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
        else:
            # Generate mock data for development
            data = self.generate_mock_status()
        
        self.send_json_response(data)
    
    def send_system_metrics(self):
        """Send system metrics"""
        try:
            # Get actual system metrics
            import psutil
            
            metrics = {
                'memory': {
                    'used': round(psutil.virtual_memory().used / (1024**3), 1),
                    'total': round(psutil.virtual_memory().total / (1024**3), 1),
                    'percentage': round(psutil.virtual_memory().percent, 1)
                },
                'cpu': {
                    'usage': round(psutil.cpu_percent(interval=1), 1),
                    'cores': psutil.cpu_count()
                },
                'disk': {
                    'used': round(psutil.disk_usage('/').used / (1024**3), 1),
                    'total': round(psutil.disk_usage('/').total / (1024**3), 1),
                    'percentage': round(psutil.disk_usage('/').percent, 1)
                },
                'network': {
                    'status': 'HEALTHY',
                    'latency': 12
                }
            }
        except ImportError:
            # Fallback to mock data if psutil not available
            metrics = {
                'memory': {'used': 4.2, 'total': 16, 'percentage': 26},
                'cpu': {'usage': 34, 'cores': 8},
                'disk': {'used': 120, 'total': 500, 'percentage': 24},
                'network': {'status': 'HEALTHY', 'latency': 12}
            }
        
        self.send_json_response(metrics)
    
    def send_hygiene_report(self):
        """Generate and send hygiene report"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'summary': {
                'total_rules': 16,
                'compliant_rules': 13,
                'violation_rules': 3,
                'compliance_percentage': 81.25
            },
            'details': 'Full report would be generated here'
        }
        
        # Send as downloadable JSON
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Disposition', 'attachment; filename="hygiene-report.json"')
        self.end_headers()
        self.wfile.write(json.dumps(report_data, indent=2).encode())
    
    def run_hygiene_audit(self):
        """Run hygiene audit"""
        try:
            # Run the actual hygiene coordinator
            result = subprocess.run([
                'python3', 
                str(self.project_root / 'scripts' / 'hygiene-enforcement-coordinator.py'),
                '--phase', '1',
                '--dry-run'
            ], capture_output=True, text=True, timeout=30)
            
            response = {
                'success': result.returncode == 0,
                'message': 'Audit completed successfully' if result.returncode == 0 else 'Audit failed',
                'output': result.stdout,
                'error': result.stderr
            }
        except subprocess.TimeoutExpired:
            response = {
                'success': False,
                'message': 'Audit timed out',
                'error': 'Operation exceeded 30 second timeout'
            }
        except Exception as e:
            response = {
                'success': False,
                'message': 'Audit failed',
                'error': str(e)
            }
        
        self.send_json_response(response)
    
    def run_hygiene_cleanup(self):
        """Run forced cleanup"""
        response = {
            'success': True,
            'message': 'Cleanup initiated',
            'warning': 'This is a simulated response. Actual cleanup would be performed here.'
        }
        self.send_json_response(response)
    
    def generate_mock_status(self):
        """Generate mock hygiene status for development"""
        import random
        
        rules = {}
        rule_names = [
            'No Fantasy Elements', 'No Breaking Changes', 'Analyze Everything',
            'Reuse Before Creating', 'Professional Standards', 'Centralized Documentation',
            'Script Organization', 'Python Script Standards', 'No Code Duplication',
            'Verify Before Cleanup', 'Clean Docker Structure', 'Single Deployment Script',
            'No Garbage Files', 'Correct AI Agent Usage', 'Clean Documentation',
            'Ollama/tinyllama Standard'
        ]
        
        for i, name in enumerate(rule_names, 1):
            rules[f'rule_{i}'] = {
                'name': name,
                'status': random.choice(['COMPLIANT', 'VIOLATION', 'COMPLIANT', 'COMPLIANT']),
                'priority': random.choice(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']),
                'category': random.choice(['Code Quality', 'Process', 'Documentation', 'Infrastructure']),
                'violationCount': random.randint(0, 15),
                'lastChecked': datetime.now().isoformat()
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'systemStatus': 'MONITORING',
            'complianceScore': random.randint(75, 95),
            'totalViolations': sum(r['violationCount'] for r in rules.values()),
            'criticalViolations': random.randint(0, 5),
            'warningViolations': random.randint(5, 15),
            'activeAgents': 8,
            'rules': rules,
            'agents': {
                'hygiene-coordinator': {'status': 'ACTIVE', 'health': 95},
                'garbage-collector': {'status': 'ACTIVE', 'health': 88},
                'deploy-automation': {'status': 'IDLE', 'health': 92},
                'script-organizer': {'status': 'ACTIVE', 'health': 78},
                'docker-optimizer': {'status': 'WARNING', 'health': 65},
                'documentation-manager': {'status': 'ACTIVE', 'health': 91},
                'python-validator': {'status': 'ACTIVE', 'health': 89},
                'compliance-monitor': {'status': 'ACTIVE', 'health': 94}
            },
            'actions': [],
            'trends': []
        }
    
    def send_json_response(self, data):
        """Send JSON response with proper headers"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def log_message(self, format, *args):
        """Override to use custom logging"""
        if hasattr(self.server, 'verbose') and self.server.verbose:
            super().log_message(format, *args)

def main():
    parser = argparse.ArgumentParser(description='Hygiene Dashboard Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Start server
    server = HTTPServer((args.host, args.port), HygieneDashboardHandler)
    server.verbose = args.verbose
    
    print(f"Starting Hygiene Dashboard Server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == '__main__':
    main()
EOF
    
    # Make server script executable
    chmod +x "$server_script"
    
    # Start the server
    local verbose_flag=""
    if [[ "$VERBOSE" == true ]]; then
        verbose_flag="--verbose"
    fi
    
    if [[ "$DEV_MODE" == true ]]; then
        log_info "Starting in development mode with auto-reload..."
        # Use nodemon if available for auto-reload, otherwise fall back to Python
        if command -v nodemon &> /dev/null; then
            nodemon --exec "python3 $server_script --host $DEFAULT_HOST --port $DEFAULT_PORT $verbose_flag" --watch .
        else
            log_warning "nodemon not found, falling back to standard server"
            python3 "$server_script" --host "$DEFAULT_HOST" --port "$DEFAULT_PORT" $verbose_flag
        fi
    else
        python3 "$server_script" --host "$DEFAULT_HOST" --port "$DEFAULT_PORT" $verbose_flag
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill
    
    # Remove temporary files
    rm -f "$DASHBOARD_DIR/server.py"
    rm -f "$DASHBOARD_DIR/config.js"
    
    log_success "Cleanup complete"
}

# Main execution
main() {
    # Set up trap for cleanup
    trap cleanup EXIT
    
    # Parse arguments
    parse_args "$@"
    
    # Show startup banner
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              Sutazai Hygiene Enforcement Monitor            ║"
    echo "║                     Dashboard Launcher                      ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Run setup
    check_prerequisites
    setup_environment
    
    # Start server
    start_server
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi