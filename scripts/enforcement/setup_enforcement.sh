#!/bin/bash
# 🔧 RULE ENFORCEMENT SYSTEM SETUP
# Installs and configures the comprehensive rule enforcement infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="${1:-/opt/sutazaiapp}"
ENFORCEMENT_DIR="$REPO_ROOT/scripts/enforcement"

# Functions
log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅${NC} $1"; }
warning() { echo -e "${YELLOW}⚠️${NC} $1"; }
error() { echo -e "${RED}❌${NC} $1" >&2; exit 1; }

# Header
echo "======================================================================"
echo "🔧 SUPREME VALIDATOR - Rule Enforcement System Setup"
echo "======================================================================"
echo ""

# Check if running in correct directory
if [ ! -f "$REPO_ROOT/CLAUDE.md" ]; then
    error "Not in SutazAI repository root. Please run from /opt/sutazaiapp"
fi

log "Setting up rule enforcement system in $REPO_ROOT"

# Step 1: Install Python dependencies
log "Installing Python dependencies..."
pip install watchdog >/dev/null 2>&1 || warning "watchdog already installed"
pip install ripgrep-py >/dev/null 2>&1 || warning "ripgrep-py optional - using grep fallback"
success "Python dependencies installed"

# Step 2: Install ripgrep for faster scanning (optional but recommended)
log "Checking for ripgrep installation..."
if ! command -v rg &> /dev/null; then
    warning "ripgrep not found - installing for better performance..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ripgrep
    elif command -v brew &> /dev/null; then
        brew install ripgrep
    else
        warning "Could not install ripgrep - will use grep fallback"
    fi
else
    success "ripgrep already installed"
fi

# Step 3: Make enforcement scripts executable
log "Setting script permissions..."
chmod +x "$ENFORCEMENT_DIR"/*.py 2>/dev/null || true
success "Script permissions set"

# Step 4: Create required directories
log "Creating required directories..."
mkdir -p "$REPO_ROOT/reports/enforcement"
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/IMPORTANT"
mkdir -p "$REPO_ROOT/docs"
success "Directories created"

# Step 5: Install pre-commit hook
log "Installing pre-commit hook..."
if [ -d "$REPO_ROOT/.git" ]; then
    # Create hooks directory if it doesn't exist
    mkdir -p "$REPO_ROOT/.git/hooks"
    
    # Create pre-commit hook
    cat > "$REPO_ROOT/.git/hooks/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook for rule enforcement
python /opt/sutazaiapp/scripts/enforcement/pre_commit_hook.py
EOF
    
    chmod +x "$REPO_ROOT/.git/hooks/pre-commit"
    success "Pre-commit hook installed"
else
    warning "Git repository not found - skipping pre-commit hook"
fi

# Step 6: Integrate with Makefile
log "Integrating with Makefile..."
if [ -f "$REPO_ROOT/Makefile" ]; then
    # Check if already integrated
    if ! grep -q "include scripts/enforcement/Makefile.enforcement" "$REPO_ROOT/Makefile"; then
        echo "" >> "$REPO_ROOT/Makefile"
        echo "# Rule Enforcement Integration" >> "$REPO_ROOT/Makefile"
        echo "-include scripts/enforcement/Makefile.enforcement" >> "$REPO_ROOT/Makefile"
        success "Makefile integration added"
    else
        success "Makefile already integrated"
    fi
else
    warning "Makefile not found - manual integration required"
fi

# Step 7: Create systemd service for continuous monitoring (optional)
log "Creating systemd service configuration..."
cat > "$ENFORCEMENT_DIR/rule-monitor.service" << EOF
[Unit]
Description=SutazAI Rule Enforcement Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_ROOT
ExecStart=/usr/bin/python3 $ENFORCEMENT_DIR/continuous_rule_monitor.py --root $REPO_ROOT --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
success "Systemd service configuration created"

# Step 8: Run initial validation
log "Running initial compliance validation..."
echo ""
python3 "$ENFORCEMENT_DIR/comprehensive_rule_enforcer.py" --root "$REPO_ROOT" --summary

# Step 9: Create enforcement status file
cat > "$REPO_ROOT/reports/enforcement/setup_status.json" << EOF
{
    "setup_timestamp": "$(date -u '+%Y-%m-%d %H:%M:%S UTC')",
    "repo_root": "$REPO_ROOT",
    "enforcement_dir": "$ENFORCEMENT_DIR",
    "pre_commit_hook": $([ -f "$REPO_ROOT/.git/hooks/pre-commit" ] && echo "true" || echo "false"),
    "makefile_integrated": $(grep -q "Makefile.enforcement" "$REPO_ROOT/Makefile" 2>/dev/null && echo "true" || echo "false"),
    "ripgrep_available": $(command -v rg &> /dev/null && echo "true" || echo "false"),
    "status": "ready"
}
EOF

# Summary
echo ""
echo "======================================================================"
echo "🔧 RULE ENFORCEMENT SETUP COMPLETE"
echo "======================================================================"
echo ""
success "Setup completed successfully!"
echo ""
echo "Available commands:"
echo "  make validate           - Quick validation with summary"
echo "  make validate-all       - Full validation with report"
echo "  make validate-fix       - Auto-fix violations"
echo "  make validate-monitor   - Start continuous monitoring"
echo "  make enforcement-dashboard - View compliance dashboard"
echo ""
echo "Pre-commit hook: $([ -f "$REPO_ROOT/.git/hooks/pre-commit" ] && echo "✅ Installed" || echo "❌ Not installed")"
echo "Continuous monitoring: Ready to start with 'make validate-monitor'"
echo ""
echo "To start continuous monitoring as a service:"
echo "  sudo cp $ENFORCEMENT_DIR/rule-monitor.service /etc/systemd/system/"
echo "  sudo systemctl enable rule-monitor"
echo "  sudo systemctl start rule-monitor"
echo ""
warning "Review the compliance report above and address any violations"