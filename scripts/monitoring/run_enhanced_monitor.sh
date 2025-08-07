#!/bin/bash
"""
Enhanced Static Monitor Launcher
================================

Launches the enhanced static monitor with proper configuration and error handling.

Usage:
    ./run_enhanced_monitor.sh [config_file]
    ./run_enhanced_monitor.sh --help
    ./run_enhanced_monitor.sh --default
    ./run_enhanced_monitor.sh --debug

Examples:
    ./run_enhanced_monitor.sh
    ./run_enhanced_monitor.sh /opt/sutazaiapp/config/monitoring/enhanced_monitor.json
    ./run_enhanced_monitor.sh --debug
"""

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration paths
DEFAULT_CONFIG="$PROJECT_ROOT/config/monitoring/enhanced_monitor.json"
MONITOR_SCRIPT="$SCRIPT_DIR/static_monitor.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_debug() {
    if [[ "${DEBUG:-0}" == "1" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $*"
    fi
}

# Help function
show_help() {
    cat << EOF
Enhanced Static Monitor Launcher

USAGE:
    $0 [OPTIONS] [CONFIG_FILE]

OPTIONS:
    --help, -h      Show this help message
    --default       Use default configuration
    --debug         Enable debug mode
    --force         Force run even without TTY
    --version       Show version information

ARGUMENTS:
    CONFIG_FILE     Path to JSON configuration file
                   Default: $DEFAULT_CONFIG

EXAMPLES:
    $0                                          # Use default config
    $0 custom_config.json                       # Use custom config
    $0 --debug                                  # Debug mode with default config
    $0 --force custom_config.json              # Force run with custom config

CONFIGURATION:
    The monitor supports extensive configuration including:
    - Adaptive refresh rates based on system load
    - Custom thresholds for CPU, memory, disk alerts
    - AI agent monitoring with health checks
    - Network bandwidth monitoring
    - Logging to file with rotation
    - Visual enhancements and trends

    Example configuration is available at:
    $DEFAULT_CONFIG

REQUIREMENTS:
    - Python 3.7+
    - psutil library (pip install psutil)
    - requests library (pip install requests)
    - Terminal with ANSI color support
    - Minimum 25 rows x 80 columns terminal

EOF
}

# Version information
show_version() {
    echo "Enhanced Static Monitor v2.0"
    echo "SutazAI System Monitoring Suite"
    echo "Built for production deployment monitoring"
}

# Dependency check
check_dependencies() {
    log_debug "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)"; then
        log_error "Python 3.7+ is required, found Python $python_version"
        exit 1
    fi
    log_debug "Python $python_version found"
    
    # Check required Python modules
    for module in psutil requests; do
        if ! python3 -c "import $module" 2>/dev/null; then
            log_error "Python module '$module' is required but not installed"
            log_info "Install with: pip install $module"
            exit 1
        fi
        log_debug "Python module '$module' found"
    done
    
    # Check monitor script
    if [[ ! -f "$MONITOR_SCRIPT" ]]; then
        log_error "Monitor script not found: $MONITOR_SCRIPT"
        exit 1
    fi
    
    # Check if script is executable
    if [[ ! -x "$MONITOR_SCRIPT" ]]; then
        log_debug "Making monitor script executable"
        chmod +x "$MONITOR_SCRIPT"
    fi
    
    log_debug "All dependencies satisfied"
}

# Terminal check
check_terminal() {
    if [[ ! -t 1 ]] && [[ "${FORCE:-0}" != "1" ]]; then
        log_error "This monitor requires an interactive terminal"
        log_info "Use --force to override this check"
        exit 1
    fi
    
    # Check terminal size
    if command -v tput &> /dev/null; then
        rows=$(tput lines 2>/dev/null || echo "0")
        cols=$(tput cols 2>/dev/null || echo "0")
        
        if [[ "$rows" -lt 25 ]] || [[ "$cols" -lt 80 ]]; then
            log_warn "Terminal size ${cols}x${rows} may be too small (recommended: 80x25 minimum)"
        else
            log_debug "Terminal size: ${cols}x${rows}"
        fi
    fi
}

# Configuration validation
validate_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        exit 1
    fi
    
    # Validate JSON syntax
    if ! python3 -c "import json; json.load(open('$config_file'))" 2>/dev/null; then
        log_error "Invalid JSON in configuration file: $config_file"
        exit 1
    fi
    
    log_debug "Configuration file validated: $config_file"
}

# Create default config if it doesn't exist
ensure_default_config() {
    if [[ ! -f "$DEFAULT_CONFIG" ]]; then
        log_info "Creating default configuration file..."
        
        # Create directory if it doesn't exist
        mkdir -p "$(dirname "$DEFAULT_CONFIG")"
        
        # Note: The default config should already be created by the main script
        log_warn "Default configuration not found at: $DEFAULT_CONFIG"
        log_info "The monitor will use built-in defaults"
    fi
}

# Main execution
main() {
    local config_file=""
    local force_flag=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --version)
                show_version
                exit 0
                ;;
            --default)
                config_file="$DEFAULT_CONFIG"
                shift
                ;;
            --debug)
                export DEBUG=1
                shift
                ;;
            --force)
                export FORCE=1
                force_flag="--force"
                shift
                ;;
            --*)
                log_error "Unknown option: $1"
                log_info "Use --help for usage information"
                exit 1
                ;;
            *)
                if [[ -z "$config_file" ]]; then
                    config_file="$1"
                else
                    log_error "Multiple configuration files specified"
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Use default config if none specified
    if [[ -z "$config_file" ]]; then
        config_file="$DEFAULT_CONFIG"
    fi
    
    # Pre-flight checks
    log_info "Starting Enhanced Static Monitor..."
    log_debug "Configuration: $config_file"
    
    check_dependencies
    check_terminal
    ensure_default_config
    
    if [[ -f "$config_file" ]]; then
        validate_config "$config_file"
        log_info "Using configuration: $config_file"
    else
        log_info "Using built-in configuration (file not found: $config_file)"
        config_file=""
    fi
    
    # Set up signal handling for clean exit
    trap 'log_info "Monitor interrupted, cleaning up..."; exit 0' INT TERM
    
    # Launch monitor
    log_info "Launching enhanced monitor (Press Ctrl+C to exit)..."
    log_debug "Command: python3 $MONITOR_SCRIPT $config_file $force_flag"
    
    if [[ -n "$config_file" && -f "$config_file" ]]; then
        exec python3 "$MONITOR_SCRIPT" "$config_file" $force_flag
    else
        exec python3 "$MONITOR_SCRIPT" $force_flag
    fi
}

# Execute main function
main "$@"