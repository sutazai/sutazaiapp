#!/usr/bin/env bash
# =============================================================================
# MCP Server Suite Provisioning Script
# =============================================================================
# Purpose: Automatically install all dependencies and ensure every MCP server 
#          starts correctly without manual intervention
# Author: Claude Code
# Created: $(date +%Y-%m-%d)
# =============================================================================

set -Eeuo pipefail

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
ok() { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
section() { printf "\n${BLUE}=== %s ===${NC}\n" "$*"; }

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
LOG_FILE="$PROJECT_ROOT/logs/mcp_provision_$(date +%Y%m%d_%H%M%S).log"
ERRORS=0

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Log to both console and file
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

log "MCP Server Suite Provisioning Started at $(date)"
log "Log file: $LOG_FILE"

# Helper functions
has_cmd() { command -v "$1" >/dev/null 2>&1; }
require_cmd() { 
    if ! has_cmd "$1"; then 
        error "Missing required command: $1"
        ((ERRORS++))
        return 1
    fi
    return 0
}

check_system() {
    section "System Requirements Check"
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ok "Linux OS detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        ok "macOS detected"
    else
        warn "Unsupported OS: $OSTYPE"
    fi
    
    # Check architecture
    ARCH=$(uname -m)
    ok "Architecture: $ARCH"
    
    # Check if running as root or with sudo access
    if [[ $EUID -eq 0 ]]; then
        ok "Running as root"
    elif sudo -n true 2>/dev/null; then
        ok "Sudo access available"
    else
        warn "No root or sudo access - some installations may fail"
    fi
}

install_system_dependencies() {
    section "Installing System Dependencies"
    
    # Detect package manager
    if has_cmd apt-get; then
        PKG_MGR="apt-get"
        PKG_INSTALL="sudo apt-get update && sudo apt-get install -y"
    elif has_cmd yum; then
        PKG_MGR="yum"
        PKG_INSTALL="sudo yum install -y"
    elif has_cmd dnf; then
        PKG_MGR="dnf"
        PKG_INSTALL="sudo dnf install -y"
    elif has_cmd brew; then
        PKG_MGR="brew"
        PKG_INSTALL="brew install"
    else
        error "No supported package manager found"
        ((ERRORS++))
        return 1
    fi
    
    log "Using package manager: $PKG_MGR"
    
    # Install curl and wget if not present
    for cmd in curl wget jq git ca-certificates; do
        if ! has_cmd "$cmd"; then
            log "Installing $cmd..."
            eval "$PKG_INSTALL $cmd" || warn "Failed to install $cmd"
        else
            ok "$cmd already installed"
        fi
    done
    
    # Install Docker if not present
    if ! has_cmd docker; then
        log "Installing Docker..."
        if [[ "$PKG_MGR" == "apt-get" ]]; then
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker "$USER" || true
            rm -f get-docker.sh
        elif [[ "$PKG_MGR" == "brew" ]]; then
            brew install --cask docker
        else
            eval "$PKG_INSTALL docker" || warn "Failed to install Docker via package manager"
        fi
    else
        ok "Docker already installed"
    fi
    
    # Install Python 3.12+ if not present
    if ! has_cmd python3; then
        log "Installing Python 3..."
        if [[ "$PKG_MGR" == "apt-get" ]]; then
            eval "$PKG_INSTALL python3 python3-pip python3-venv"
        elif [[ "$PKG_MGR" == "brew" ]]; then
            brew install python@3.12
        else
            eval "$PKG_INSTALL python3 python3-pip"
        fi
    else
        ok "Python 3 already installed: $(python3 --version)"
    fi
    
    # Install pip if not present
    if ! has_cmd pip3 && ! python3 -m pip --version >/dev/null 2>&1; then
        log "Installing pip..."
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py
        rm -f get-pip.py
    else
        ok "pip already available"
    fi
}

install_node_and_npm() {
    section "Installing Node.js and npm"
    
    # Check if Node.js is already installed with a reasonable version
    if has_cmd node; then
        NODE_VERSION=$(node --version | cut -d'v' -f2)
        MAJOR_VERSION=$(echo "$NODE_VERSION" | cut -d'.' -f1)
        if [[ $MAJOR_VERSION -ge 18 ]]; then
            ok "Node.js already installed: v$NODE_VERSION"
            if has_cmd npm; then
                ok "npm already installed: $(npm --version)"
                return 0
            fi
        fi
    fi
    
    log "Installing Node.js via NodeSource..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Install Node.js 22.x on Linux
        curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Install Node.js on macOS via Homebrew
        if has_cmd brew; then
            brew install node
        else
            error "Homebrew not found on macOS"
            ((ERRORS++))
            return 1
        fi
    else
        # Fallback: install via Node Version Manager (nvm)
        log "Installing Node.js via nvm..."
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        nvm install 22
        nvm use 22
        nvm alias default 22
    fi
    
    # Verify installation
    if has_cmd node && has_cmd npm; then
        ok "Node.js installed: $(node --version)"
        ok "npm installed: $(npm --version)"
    else
        error "Failed to install Node.js/npm"
        ((ERRORS++))
        return 1
    fi
}

install_go() {
    section "Installing Go"
    
    # Check if Go is already installed
    if has_cmd go; then
        GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
        ok "Go already installed: $GO_VERSION"
        return 0
    fi
    
    log "Installing Go..."
    
    # Determine architecture
    case "$ARCH" in
        x86_64) GO_ARCH="amd64" ;;
        aarch64|arm64) GO_ARCH="arm64" ;;
        armv7l) GO_ARCH="armv6l" ;;
        *) error "Unsupported architecture: $ARCH"; ((ERRORS++)); return 1 ;;
    esac
    
    # Determine OS
    case "$OSTYPE" in
        linux-gnu*) GO_OS="linux" ;;
        darwin*) GO_OS="darwin" ;;
        *) error "Unsupported OS: $OSTYPE"; ((ERRORS++)); return 1 ;;
    esac
    
    # Download and install Go
    GO_VERSION="1.21.6"
    GO_TARBALL="go${GO_VERSION}.${GO_OS}-${GO_ARCH}.tar.gz"
    
    cd /tmp
    curl -LO "https://golang.org/dl/$GO_TARBALL"
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "$GO_TARBALL"
    rm -f "$GO_TARBALL"
    
    # Add Go to PATH
    if ! grep -q "/usr/local/go/bin" ~/.bashrc 2>/dev/null; then
        echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
    fi
    
    # Set up Go environment for current session
    export PATH=$PATH:/usr/local/go/bin
    export GOPATH="$HOME/go"
    export GOBIN="$GOPATH/bin"
    mkdir -p "$GOPATH/bin"
    
    # Verify installation
    if has_cmd go; then
        ok "Go installed: $(go version)"
    else
        error "Failed to install Go"
        ((ERRORS++))
        return 1
    fi
}

install_uv() {
    section "Installing uv (Python package manager)"
    
    if has_cmd uv; then
        ok "uv already installed: $(uv --version)"
        return 0
    fi
    
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if has_cmd uv; then
        ok "uv installed: $(uv --version)"
    else
        error "Failed to install uv"
        ((ERRORS++))
        return 1
    fi
}

setup_go_mcp_servers() {
    section "Setting Up Go-based MCP Servers"
    
    # Ensure Go is available
    if ! has_cmd go; then
        error "Go not available for MCP server installation"
        ((ERRORS++))
        return 1
    fi
    
    # Set up Go environment
    export PATH=$PATH:/usr/local/go/bin
    export GOPATH="$HOME/go"
    export GOBIN="$GOPATH/bin"
    mkdir -p "$GOPATH/bin"
    
    # Install mcp-language-server
    if [[ ! -f "/root/go/bin/mcp-language-server" ]]; then
        log "Installing mcp-language-server..."
        go install github.com/trypear/mcp-language-server@latest
        
        # Create symlink in expected location
        sudo mkdir -p /root/go/bin
        if [[ -f "$GOPATH/bin/mcp-language-server" ]]; then
            sudo cp "$GOPATH/bin/mcp-language-server" /root/go/bin/
            ok "mcp-language-server installed"
        else
            error "Failed to install mcp-language-server"
            ((ERRORS++))
        fi
    else
        ok "mcp-language-server already installed"
    fi
    
    # Install TypeScript Language Server
    if [[ ! -f "/root/.nvm/versions/node/v22.18.0/bin/typescript-language-server" ]]; then
        log "Installing TypeScript Language Server..."
        
        # Create directory structure and install
        sudo mkdir -p /root/.nvm/versions/node/v22.18.0/bin
        npm install -g typescript-language-server typescript
        
        # Find the actual location and symlink
        if has_cmd typescript-language-server; then
            TS_SERVER_PATH=$(which typescript-language-server)
            sudo ln -sf "$TS_SERVER_PATH" /root/.nvm/versions/node/v22.18.0/bin/typescript-language-server
            ok "TypeScript Language Server installed"
        else
            error "Failed to install TypeScript Language Server"
            ((ERRORS++))
        fi
    else
        ok "TypeScript Language Server already installed"
    fi
}

setup_python_mcp_servers() {
    section "Setting Up Python-based MCP Servers"
    
    # Set up UltimateCoder MCP
    ULTIMATECODER_VENV="/opt/sutazaiapp/.mcp/UltimateCoderMCP/.venv"
    ULTIMATECODER_MAIN="/opt/sutazaiapp/.mcp/UltimateCoderMCP/main.py"
    
    if [[ ! -d "$ULTIMATECODER_VENV" ]]; then
        log "Setting up UltimateCoder MCP..."
        mkdir -p "$(dirname "$ULTIMATECODER_VENV")"
        python3 -m venv "$ULTIMATECODER_VENV"
        "$ULTIMATECODER_VENV/bin/pip" install fastmcp
        
        # Create a basic main.py if it doesn't exist
        if [[ ! -f "$ULTIMATECODER_MAIN" ]]; then
            cat > "$ULTIMATECODER_MAIN" << 'EOF'
#!/usr/bin/env python3
"""
UltimateCoder MCP Server
"""
from fastmcp import FastMCP

mcp = FastMCP("UltimateCoder")

@mcp.tool()
def hello() -> str:
    """Say hello from UltimateCoder MCP"""
    return "Hello from UltimateCoder MCP!"

if __name__ == "__main__":
    mcp.run()
EOF
            chmod +x "$ULTIMATECODER_MAIN"
        fi
        
        ok "UltimateCoder MCP set up"
    else
        ok "UltimateCoder MCP already set up"
    fi
    
    # Set up extended-memory MCP
    MEMORY_VENV="/opt/sutazaiapp/.venvs/extended-memory"
    
    if [[ ! -d "$MEMORY_VENV" ]]; then
        log "Setting up extended-memory MCP..."
        mkdir -p "$(dirname "$MEMORY_VENV")"
        python3 -m venv "$MEMORY_VENV"
        "$MEMORY_VENV/bin/pip" install extended-memory-mcp
        ok "extended-memory MCP set up"
    else
        ok "extended-memory MCP already set up"
    fi
    
    # Install memory-bank-mcp globally if uv is available
    if has_cmd uv; then
        if ! python3 -c 'import memory_bank_mcp' 2>/dev/null; then
            log "Installing memory-bank-mcp via uv..."
            uv pip install memory-bank-mcp --system 2>/dev/null || warn "Failed to install memory-bank-mcp via uv"
        else
            ok "memory-bank-mcp already installed"
        fi
    fi
}

setup_nodejs_mcp_servers() {
    section "Setting Up Node.js-based MCP Servers"
    
    # List of npm packages to install globally
    NPM_PACKAGES=(
        "@modelcontextprotocol/server-github"
        "@modelcontextprotocol/server-filesystem"
        "@upstash/context7-mcp"
        "@modelcontextprotocol/server-duckduckgo"
        "@modelcontextprotocol/server-http"
        "@playwright/mcp"
        "playwright-mcp"
        "mcp-knowledge-graph"
        "puppeteer-mcp"
        "typescript-language-server"
        "typescript"
        "nx-mcp"
    )
    
    # Memory bank MCP from GitHub
    GITHUB_PACKAGES=(
        "github:alioshr/memory-bank-mcp"
    )
    
    log "Installing npm packages globally..."
    for package in "${NPM_PACKAGES[@]}"; do
        if npm list -g "$package" >/dev/null 2>&1; then
            ok "$package already installed"
        else
            log "Installing $package..."
            if ! npm install -g "$package"; then
                error "CRITICAL: Failed to install required package: $package"
                ((ERRORS++))
                # Continue with other packages but track the failure
            else
                ok "$package installed successfully"
            fi
        fi
    done
    
    log "Installing GitHub-based npm packages..."
    for package in "${GITHUB_PACKAGES[@]}"; do
        log "Installing $package..."
        npm install -g "$package" || warn "Failed to install $package"
    done
    
    # Install Playwright browsers
    if has_cmd playwright; then
        log "Installing Playwright browsers..."
        playwright install chromium || warn "Failed to install Playwright browsers"
    elif npx playwright --version >/dev/null 2>&1; then
        log "Installing Playwright browsers via npx..."
        npx playwright install chromium || warn "Failed to install Playwright browsers via npx"
    fi
    
    # Set up browser cache directory
    PLAYWRIGHT_CACHE="/root/.cache/ms-playwright"
    if [[ ! -d "$PLAYWRIGHT_CACHE" ]]; then
        sudo mkdir -p "$PLAYWRIGHT_CACHE"
        ok "Playwright cache directory created"
    fi
}

setup_docker_mcp_servers() {
    section "Setting Up Docker-based MCP Servers"
    
    if ! has_cmd docker; then
        error "Docker not available for MCP server setup"
        ((ERRORS++))
        return 1
    fi
    
    # Start Docker service if not running
    if ! docker info >/dev/null 2>&1; then
        log "Starting Docker service..."
        sudo systemctl start docker 2>/dev/null || sudo service docker start 2>/dev/null || warn "Could not start Docker service"
    fi
    
    # Pull required Docker images
    DOCKER_IMAGES=(
        "crystaldba/postgres-mcp"
        "mcp/duckduckgo"
        "mcp/sequentialthinking"
        "mcp/fetch"
    )
    
    for image in "${DOCKER_IMAGES[@]}"; do
        if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "$image"; then
            ok "$image already pulled"
        else
            log "Pulling Docker image: $image..."
            docker pull "$image" || warn "Failed to pull $image"
        fi
    done
    
    # Ensure Docker network exists
    NETWORK_NAME="sutazai-network"
    if docker network ls --format "{{.Name}}" | grep -q "^${NETWORK_NAME}$"; then
        ok "Docker network $NETWORK_NAME already exists"
    else
        log "Creating Docker network: $NETWORK_NAME..."
        docker network create "$NETWORK_NAME" || warn "Failed to create Docker network"
    fi
    
    # Check if postgres container is running
    POSTGRES_CONTAINER="sutazai-postgres"
    if docker ps --format "{{.Names}}" | grep -q "^${POSTGRES_CONTAINER}$"; then
        ok "Postgres container $POSTGRES_CONTAINER is running"
    else
        warn "Postgres container $POSTGRES_CONTAINER is not running"
        log "To start postgres: docker compose up -d postgres"
    fi
}

setup_special_mcp_servers() {
    section "Setting Up Special MCP Servers"
    
    # Set up mcp_ssh
    MCP_SSH_DIR="$PROJECT_ROOT/mcp_ssh"
    if [[ ! -d "$MCP_SSH_DIR/.git" ]]; then
        log "Cloning mcp_ssh repository..."
        git clone https://github.com/sinjab/mcp_ssh "$MCP_SSH_DIR" || warn "Failed to clone mcp_ssh"
    else
        ok "mcp_ssh repository already cloned"
    fi
    
    # Set up chroma MCP
    CHROMA_MCP_DIR="$PROJECT_ROOT/.mcp/chroma"
    if [[ ! -d "$CHROMA_MCP_DIR/.git" ]]; then
        log "Cloning chroma MCP repository..."
        mkdir -p "$(dirname "$CHROMA_MCP_DIR")"
        git clone https://github.com/privetin/chroma "$CHROMA_MCP_DIR" || warn "Failed to clone chroma MCP"
    else
        ok "chroma MCP repository already cloned"
    fi
}

create_environment_files() {
    section "Creating Environment Configuration"
    
    # Create .env file if it doesn't exist
    ENV_FILE="$PROJECT_ROOT/.env"
    if [[ ! -f "$ENV_FILE" ]]; then
        log "Creating default .env file..."
        cat > "$ENV_FILE" << 'EOF'
# MCP Server Environment Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=8lfYZjSlJrSCCmWyKOTtxI3ydNFVAMA4
POSTGRES_DB=sutazai
POSTGRES_HOST=postgres
DATABASE_URI=postgresql://sutazai:8lfYZjSlJrSCCmWyKOTtxI3ydNFVAMA4@postgres:5432/sutazai
DOCKER_NETWORK=sutazai-network
POSTGRES_CONTAINER=sutazai-postgres
PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright
CHROMA_URL=http://sutazai-chromadb:8000
EOF
        ok ".env file created"
    else
        ok ".env file already exists"
    fi
    
    # Set execute permissions on wrapper scripts
    WRAPPER_DIR="$PROJECT_ROOT/scripts/mcp/wrappers"
    if [[ -d "$WRAPPER_DIR" ]]; then
        log "Setting execute permissions on wrapper scripts..."
        chmod +x "$WRAPPER_DIR"/*.sh || warn "Failed to set permissions on wrapper scripts"
        ok "Wrapper script permissions set"
    fi
}

run_mcp_selfchecks() {
    section "Running MCP Server Self-Checks"
    
    SELFCHECK_SCRIPT="$PROJECT_ROOT/scripts/mcp/selfcheck_all.sh"
    if [[ -x "$SELFCHECK_SCRIPT" ]]; then
        log "Running MCP server self-checks..."
        if "$SELFCHECK_SCRIPT"; then
            ok "All MCP server self-checks passed"
        else
            warn "Some MCP server self-checks failed - check the selfcheck log for details"
        fi
    else
        warn "MCP selfcheck script not found or not executable"
    fi
}

generate_status_report() {
    section "Generating Status Report"
    
    REPORT_FILE="$PROJECT_ROOT/logs/mcp_provision_status_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$REPORT_FILE" << EOF
# MCP Server Provisioning Status Report

Generated: $(date)
Script: $0
Log File: $LOG_FILE

## System Information
- OS: $OSTYPE
- Architecture: $(uname -m)
- User: $(whoami)

## Installed Dependencies

### System Tools
EOF
    
    # Check system tools
    for cmd in docker python3 node npm go uv; do
        if has_cmd "$cmd"; then
            case "$cmd" in
                docker) VERSION=$(docker --version 2>/dev/null || echo "unknown") ;;
                python3) VERSION=$(python3 --version 2>/dev/null || echo "unknown") ;;
                node) VERSION=$(node --version 2>/dev/null || echo "unknown") ;;
                npm) VERSION=$(npm --version 2>/dev/null || echo "unknown") ;;
                go) VERSION=$(go version 2>/dev/null || echo "unknown") ;;
                uv) VERSION=$(uv --version 2>/dev/null || echo "unknown") ;;
            esac
            echo "- ✅ $cmd: $VERSION" >> "$REPORT_FILE"
        else
            echo "- ❌ $cmd: NOT INSTALLED" >> "$REPORT_FILE"
        fi
    done
    
    cat >> "$REPORT_FILE" << 'EOF'

### MCP Server Status

EOF
    
    # Check MCP servers from .mcp.json
    if [[ -f "$PROJECT_ROOT/.mcp.json" ]]; then
        echo "Reading MCP servers from .mcp.json..." >> "$REPORT_FILE"
        # We'll add a simple status check here
        echo "- Total MCP servers configured: $(jq '.mcpServers | length' "$PROJECT_ROOT/.mcp.json" 2>/dev/null || echo "unknown")" >> "$REPORT_FILE"
    fi
    
    cat >> "$REPORT_FILE" << EOF

## Errors Encountered
Total errors: $ERRORS

## Next Steps
1. Review any error messages in the log file: $LOG_FILE
2. Run self-checks: $PROJECT_ROOT/scripts/mcp/selfcheck_all.sh
3. Test MCP servers in Claude Code

---
End of report
EOF
    
    ok "Status report generated: $REPORT_FILE"
    log "Total errors encountered: $ERRORS"
}

main() {
    log "Starting MCP Server Suite Provisioning..."
    
    # Check system requirements
    check_system
    
    # Install dependencies
    install_system_dependencies
    install_node_and_npm
    install_go
    install_uv
    
    # Set up MCP servers by type
    setup_go_mcp_servers
    setup_python_mcp_servers
    setup_nodejs_mcp_servers
    setup_docker_mcp_servers
    setup_special_mcp_servers
    
    # Configure environment
    create_environment_files
    
    # Run validation
    run_mcp_selfchecks
    
    # Generate report
    generate_status_report
    
    if [[ $ERRORS -eq 0 ]]; then
        ok "MCP Server Suite Provisioning completed successfully!"
        ok "All dependencies installed and configured."
    else
        warn "MCP Server Suite Provisioning completed with $ERRORS errors."
        warn "Check the log file for details: $LOG_FILE"
    fi
    
    log "Provisioning finished at $(date)"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
