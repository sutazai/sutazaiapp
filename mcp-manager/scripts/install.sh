#!/bin/bash
# MCP Manager Installation Script
# Installs and configures the Dynamic MCP Management System

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/sutazaiapp/mcp-manager"
CONFIG_DIR="$INSTALL_DIR/config"
LOG_DIR="/var/log/mcp-manager"
STATE_DIR="/var/lib/mcp-manager"
PYTHON_MIN_VERSION="3.10"

# Functions
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

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
            log_success "Python $PYTHON_VERSION is compatible"
        else
            log_error "Python $PYTHON_MIN_VERSION+ is required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 >/dev/null 2>&1; then
        log_error "pip3 is not installed"
        exit 1
    fi
    
    # Check git (for development installs)
    if ! command -v git >/dev/null 2>&1; then
        log_warning "git is not installed, some features may not work"
    fi
    
    log_success "System requirements check passed"
}

create_directories() {
    log_info "Creating directories..."
    
    # Create directories with proper permissions
    sudo mkdir -p "$LOG_DIR"
    sudo mkdir -p "$STATE_DIR"
    sudo mkdir -p "$CONFIG_DIR"
    
    # Set ownership
    sudo chown -R "$USER:$USER" "$LOG_DIR" "$STATE_DIR" 2>/dev/null || true
    
    log_success "Directories created"
}

install_dependencies() {
    log_info "Installing MCP Manager and dependencies..."
    
    # Navigate to install directory
    cd "$INSTALL_DIR"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install package in development mode
    pip install -e .
    
    # Install development dependencies if requested
    if [[ "${1:-}" == "--dev" ]]; then
        log_info "Installing development dependencies..."
        pip install -e .[dev]
    fi
    
    log_success "Dependencies installed"
}

setup_configuration() {
    log_info "Setting up configuration..."
    
    # Copy default configuration if it doesn't exist
    if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
        cp "$CONFIG_DIR/default.yaml" "$CONFIG_DIR/config.yaml"
        log_info "Created default configuration file"
    fi
    
    # Update paths in configuration
    sed -i "s|/tmp/mcp_manager_state.json|$STATE_DIR/state.json|g" "$CONFIG_DIR/config.yaml" 2>/dev/null || true
    sed -i "s|log_file: null|log_file: \"$LOG_DIR/mcp-manager.log\"|g" "$CONFIG_DIR/config.yaml" 2>/dev/null || true
    
    log_success "Configuration setup complete"
}

create_systemd_service() {
    if [[ "${1:-}" != "--systemd" ]]; then
        return 0
    fi
    
    log_info "Creating systemd service..."
    
    # Create systemd service file
    sudo tee /etc/systemd/system/mcp-manager.service >/dev/null <<EOF
[Unit]
Description=Dynamic MCP Management System
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/venv/bin:\$PATH
ExecStart=$INSTALL_DIR/venv/bin/mcp-manager start --daemon
ExecStop=$INSTALL_DIR/venv/bin/mcp-manager stop
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    log_success "Systemd service created"
    log_info "To enable the service: sudo systemctl enable mcp-manager"
    log_info "To start the service: sudo systemctl start mcp-manager"
}

setup_shell_completion() {
    log_info "Setting up shell completion..."
    
    # Bash completion
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "mcp-manager completion" "$HOME/.bashrc"; then
            echo "" >> "$HOME/.bashrc"
            echo "# MCP Manager completion" >> "$HOME/.bashrc"
            echo 'eval "$(_MCP_MANAGER_COMPLETE=source_bash mcp-manager)"' >> "$HOME/.bashrc"
            log_success "Bash completion added to ~/.bashrc"
        fi
    fi
    
    # Zsh completion
    if [ -f "$HOME/.zshrc" ] && command -v zsh >/dev/null 2>&1; then
        if ! grep -q "mcp-manager completion" "$HOME/.zshrc"; then
            echo "" >> "$HOME/.zshrc"
            echo "# MCP Manager completion" >> "$HOME/.zshrc"
            echo 'eval "$(_MCP_MANAGER_COMPLETE=source_zsh mcp-manager)"' >> "$HOME/.zshrc"
            log_success "Zsh completion added to ~/.zshrc"
        fi
    fi
}

fix_task_runner() {
    log_info "Setting up fixed task runner..."
    
    # Create symlink for easy access
    TASK_RUNNER_SCRIPT="$INSTALL_DIR/scripts/fixed-task-runner.sh"
    
    cat > "$TASK_RUNNER_SCRIPT" <<'EOF'
#!/bin/bash
# Fixed Task Runner MCP Server Wrapper

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_MANAGER_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "$MCP_MANAGER_DIR/venv/bin/activate"

# Run fixed task runner
exec python -m mcp_manager.fixed_task_runner "$@"
EOF
    
    chmod +x "$TASK_RUNNER_SCRIPT"
    
    # Update wrapper script to use the fixed version
    WRAPPER_SCRIPT="/opt/sutazaiapp/scripts/mcp/wrappers/claude-task-runner.sh"
    if [ -f "$WRAPPER_SCRIPT" ]; then
        log_info "Updating claude-task-runner wrapper to use fixed version..."
        
        # Backup original
        cp "$WRAPPER_SCRIPT" "$WRAPPER_SCRIPT.backup"
        
        # Update MCP_COMMAND to use fixed version
        sed -i "s|MCP_COMMAND=.*|MCP_COMMAND=\"$TASK_RUNNER_SCRIPT\"|g" "$WRAPPER_SCRIPT" 2>/dev/null || true
        
        log_success "Updated claude-task-runner wrapper"
    fi
}

run_tests() {
    if [[ "${1:-}" != "--test" ]]; then
        return 0
    fi
    
    log_info "Running tests..."
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    if command -v pytest >/dev/null 2>&1; then
        pytest tests/ -v
        log_success "Tests passed"
    else
        log_warning "pytest not available, skipping tests"
    fi
}

main() {
    log_info "Starting MCP Manager installation..."
    
    # Parse arguments
    INSTALL_DEV=false
    INSTALL_SYSTEMD=false
    RUN_TESTS=false
    
    for arg in "$@"; do
        case $arg in
            --dev)
                INSTALL_DEV=true
                ;;
            --systemd)
                INSTALL_SYSTEMD=true
                ;;
            --test)
                RUN_TESTS=true
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --dev      Install development dependencies"
                echo "  --systemd  Create systemd service"
                echo "  --test     Run tests after installation"
                echo "  --help     Show this help message"
                exit 0
                ;;
        esac
    done
    
    # Run installation steps
    check_requirements
    create_directories
    
    if [ "$INSTALL_DEV" = true ]; then
        install_dependencies --dev
    else
        install_dependencies
    fi
    
    setup_configuration
    fix_task_runner
    setup_shell_completion
    
    if [ "$INSTALL_SYSTEMD" = true ]; then
        create_systemd_service --systemd
    fi
    
    if [ "$RUN_TESTS" = true ]; then
        run_tests --test
    fi
    
    log_success "MCP Manager installation complete!"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Restart your shell or run: source ~/.bashrc"
    echo "2. Test installation: mcp-manager status"
    echo "3. Discover servers: mcp-manager discover"
    echo "4. Start MCP Manager: mcp-manager start"
    echo ""
    echo -e "${BLUE}Configuration:${NC} $CONFIG_DIR/config.yaml"
    echo -e "${BLUE}Logs:${NC} $LOG_DIR/"
    echo -e "${BLUE}State:${NC} $STATE_DIR/"
    echo ""
    echo -e "${YELLOW}Documentation:${NC} $INSTALL_DIR/README.md"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_error "Do not run this script as root"
    exit 1
fi

# Run main function with all arguments
main "$@"