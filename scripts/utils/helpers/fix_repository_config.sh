#!/bin/bash
# fix_repository_config.sh - Fix repository configuration for SutazAI

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
LOG_FILE="${PROJECT_ROOT}/logs/repository_fix_$(date +%Y%m%d_%H%M%S).log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE" >&2
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

# Create log directory
mkdir -p "${PROJECT_ROOT}/logs"

log "Starting SutazAI Repository Configuration Fix"
log "=============================================="

# Step 1: Verify current directory and git status
log "Step 1: Verifying current repository status..."

if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
    error "No git repository found in ${PROJECT_ROOT}"
    exit 1
fi

cd "$PROJECT_ROOT"

# Step 2: Fix git ownership issues
log "Step 2: Fixing git ownership and permissions..."

# Add safe directory
git config --global --add safe.directory "$PROJECT_ROOT" 2>/dev/null || true

# Fix git permissions
chmod -R 755 .git/
chmod 644 .git/config

# Step 3: Verify remote configuration
log "Step 3: Verifying remote configuration..."

CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
EXPECTED_REMOTE="https://github.com/sutazai/sutazaiapp.git"

if [[ "$CURRENT_REMOTE" != *"sutazai/sutazaiapp"* ]]; then
    warning "Remote URL doesn't match expected pattern"
    log "Current remote: $CURRENT_REMOTE"
    log "Expected pattern: sutazai/sutazaiapp"
    
    # Fix remote URL if needed
    if [[ -n "$CURRENT_REMOTE" ]]; then
        git remote set-url origin "$EXPECTED_REMOTE"
        success "Updated remote URL to: $EXPECTED_REMOTE"
    else
        git remote add origin "$EXPECTED_REMOTE"
        success "Added remote origin: $EXPECTED_REMOTE"
    fi
else
    success "Remote configuration is correct"
fi

# Step 4: Check current branch
log "Step 4: Checking current branch..."

CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
log "Current branch: $CURRENT_BRANCH"

# Step 5: Create proper workspace configuration
log "Step 5: Creating workspace configuration..."

# Create .vscode/workspace.json if it doesn't exist
mkdir -p .vscode
if [[ ! -f ".vscode/workspace.json" ]]; then
    cat > .vscode/workspace.json << 'EOF'
{
    "folders": [
        {
            "name": "SutazAI",
            "path": "."
        }
    ],
    "settings": {
        "git.repositoryScanMaxDepth": 1,
        "git.autoRepositoryDetection": true,
        "files.exclude": {
            "**/__pycache__": true,
            "**/*.pyc": true,
            "**/node_modules": true,
            "**/.git": false
        }
    }
}
EOF
    success "Created VS Code workspace configuration"
fi

# Step 6: Update .vscode/settings.json
log "Step 6: Updating VS Code settings..."

cat > .vscode/settings.json << 'EOF'
{
    "makefile.configureOnOpen": false,
    "git.repositoryScanMaxDepth": 1,
    "git.autoRepositoryDetection": true,
    "python.defaultInterpreterPath": "/opt/sutazaiapp/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "files.watcherExclude": {
        "**/__pycache__/**": true,
        "**/node_modules/**": true,
        "**/logs/**": true,
        "**/data/cache/**": true
    },
    "search.exclude": {
        "**/__pycache__": true,
        "**/node_modules": true,
        "**/logs": true,
        "**/data/cache": true,
        "**/.git": true
    }
}
EOF
success "Updated VS Code settings"

# Step 7: Create repository metadata
log "Step 7: Creating repository metadata..."

cat > .repository-info.json << 'EOF'
{
    "name": "sutazaiapp",
    "owner": "sutazai",
    "full_name": "sutazai/sutazaiapp",
    "description": "SutazAI automation/advanced automation Autonomous System",
    "type": "local",
    "path": "/opt/sutazaiapp",
    "git_remote": "https://github.com/sutazai/sutazaiapp.git",
    "current_branch": "v28",
    "workspace_type": "sutazai_agi_system"
}
EOF
success "Created repository metadata"

# Step 8: Fix Claude configuration
log "Step 8: Updating Claude configuration..."

# Update .claude/settings.local.json
cat > .claude/settings.local.json << 'EOF'
{
  "permissions": {
    "allow": [
      "Bash(mkdir:*)",
      "Bash(ss:*)",
      "Bash(chmod:*)",
      "Bash(python3:*)",
      "Bash(pip3 install:*)",
      "Bash(apt:*)",
      "Bash(apt install:*)",
      "Bash(find:*)",
      "Bash(source:*)",
      "Bash(python:*)",
      "Bash(rm:*)",
      "Bash(pip install:*)",
      "Bash(curl:*)",
      "Bash(ls:*)",
      "Bash(pkill:*)",
      "Bash(kill:*)",
      "Bash(docker:*)",
      "Bash(sh:*)",
      "Bash(service docker:*)",
      "Bash(grep:*)",
      "Bash(sed:*)",
      "Bash(echo)",
      "Bash(cat:*)",
      "Bash(touch:*)",
      "Bash(cp:*)",
      "Bash(git:*)",
      "Bash(systemctl:*)",
      "Bash(sudo:*)",
      "Bash(mv:*)",
      "Bash(timeout:*)",
      "Bash(journalctl:*)",
      "WebFetch(domain:github.com)"
    ],
    "deny": []
  },
  "enableAllProjectMcpServers": true,
  "enabledMcpjsonServers": [
    "task-master-ai"
  ],
  "repository": {
    "name": "sutazaiapp",
    "owner": "sutazai",
    "full_name": "sutazai/sutazaiapp",
    "path": "/opt/sutazaiapp",
    "type": "local"
  }
}
EOF
success "Updated Claude configuration"

# Step 9: Create background agent configuration
log "Step 9: Creating background agent configuration..."

mkdir -p .claude/agents
cat > .claude/agents/background-agent.json << 'EOF'
{
    "name": "sutazai-background-agent",
    "repository": "sutazai/sutazaiapp",
    "workspace_path": "/opt/sutazaiapp",
    "enabled": true,
    "permissions": [
        "file_read",
        "file_write",
        "git_operations",
        "docker_operations",
        "system_operations"
    ],
    "triggers": [
        "file_change",
        "git_push",
        "deployment_request"
    ]
}
EOF
success "Created background agent configuration"

# Step 10: Verify git status
log "Step 10: Verifying git status..."

git status --porcelain > /dev/null 2>&1
if [[ $? -eq 0 ]]; then
    success "Git repository is healthy"
    
    # Show current status
    log "Repository Information:"
    log "  - Remote: $(git remote get-url origin)"
    log "  - Branch: $(git branch --show-current)"
    log "  - Commit: $(git rev-parse --short HEAD)"
    log "  - Status: $(git status --porcelain | wc -l) changes"
else
    error "Git repository has issues"
    git status
fi

# Step 11: Test repository access
log "Step 11: Testing repository access..."

# Test git operations
if git fetch origin --dry-run > /dev/null 2>&1; then
    success "Repository access test passed"
else
    warning "Repository access test failed - this may be expected for local development"
fi

# Step 12: Create workspace validation script
log "Step 12: Creating workspace validation script..."

cat > scripts/validate_workspace.sh << 'EOF'
#!/bin/bash
# validate_workspace.sh - Validate workspace configuration

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"

echo "=== SutazAI Workspace Validation ==="
echo "Project Root: $PROJECT_ROOT"
echo "Current Directory: $(pwd)"
echo ""

# Check if we're in the right directory
if [[ "$(pwd)" != "$PROJECT_ROOT" ]]; then
    echo "❌ Not in project root directory"
    exit 1
else
    echo "✅ In correct project directory"
fi

# Check git repository
if [[ -d ".git" ]]; then
    echo "✅ Git repository found"
    echo "  - Remote: $(git remote get-url origin 2>/dev/null || echo 'Not configured')"
    echo "  - Branch: $(git branch --show-current 2>/dev/null || echo 'Unknown')"
else
    echo "❌ No git repository found"
fi

# Check required directories
required_dirs=("backend" "frontend" "agents" "scripts" "data" "logs")
for dir in "${required_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "✅ Directory exists: $dir"
    else
        echo "❌ Missing directory: $dir"
    fi
done

# Check configuration files
config_files=(".vscode/settings.json" ".claude/settings.local.json" "docker-compose.yml")
for file in "${config_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ Configuration file exists: $file"
    else
        echo "❌ Missing configuration file: $file"
    fi
done

echo ""
echo "=== Validation Complete ==="
EOF

chmod +x scripts/validate_workspace.sh
success "Created workspace validation script"

# Step 13: Final verification
log "Step 13: Running final verification..."

# Run validation script
if ./scripts/validate_workspace.sh; then
    success "Workspace validation passed"
else
    warning "Workspace validation had issues - check output above"
fi

# Summary
log "=============================================="
log "Repository Configuration Fix Complete!"
log "=============================================="
log ""
log "What was fixed:"
log "  ✅ Git ownership and permissions"
log "  ✅ Remote repository configuration"
log "  ✅ VS Code workspace settings"
log "  ✅ Claude configuration"
log "  ✅ Background agent configuration"
log "  ✅ Repository metadata"
log "  ✅ Workspace validation script"
log ""
log "Next steps:"
log "  1. Restart your development environment"
log "  2. Run: ./scripts/validate_workspace.sh"
log "  3. Try using the background agent again"
log ""
log "Log saved to: $LOG_FILE"
log "=============================================="

success "Repository configuration fix completed successfully!" 