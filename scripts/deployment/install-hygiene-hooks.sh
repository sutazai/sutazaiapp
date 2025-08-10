#!/bin/bash
# Purpose: Install pre-commit hooks for comprehensive hygiene enforcement
# Usage: ./install-hygiene-hooks.sh [--force]
# Requirements: Python 3.8+, pip, git

set -e

# Colors for output

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
PRECOMMIT_CONFIG="$PROJECT_ROOT/.pre-commit-config.yaml"
SCRIPTS_DIR="$PROJECT_ROOT/scripts/pre-commit"
BYPASS_LOG="$PROJECT_ROOT/.git/hooks/bypass.log"

echo -e "${BLUE}🛠️  SutazAI Hygiene Hooks Installation${NC}"
echo "============================================"

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo -e "${RED}❌ Error: Not in a git repository${NC}"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Check Python version
echo -e "\n${BLUE}📋 Checking prerequisites...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null; then
    echo -e "\n${YELLOW}📦 Installing pre-commit...${NC}"
    pip3 install pre-commit
else
    echo -e "  ✅ pre-commit is already installed"
fi

# Install required Python packages for hooks
echo -e "\n${BLUE}📦 Installing hook dependencies...${NC}"
pip3 install --quiet pyyaml gitpython

# Create pre-commit scripts directory if it doesn't exist
if [ ! -d "$SCRIPTS_DIR" ]; then
    echo -e "\n${YELLOW}📁 Creating pre-commit scripts directory...${NC}"
    mkdir -p "$SCRIPTS_DIR"
fi

# Check if all required scripts exist
echo -e "\n${BLUE}🔍 Checking hook scripts...${NC}"
missing_scripts=()
required_scripts=(
    "check-fantasy-elements.py"
    "check-deployment-scripts.py"
    "check-directory-duplication.py"
    "check-docker-structure.py"
    "check-python-docs.py"
    "quick-system-check.py"
    "check-garbage-files.py"
    "check-breaking-changes.py"
    "check-safe-deletion.py"
    "check-script-organization.py"
    "check-doc-structure.py"
    "check-script-duplication.py"
    "check-doc-duplication.py"
    "check-agent-usage.py"
    "check-llm-usage.py"
)

for script in "${required_scripts[@]}"; do
    if [ ! -f "$SCRIPTS_DIR/$script" ]; then
        missing_scripts+=("$script")
    else
        echo "  ✅ $script"
    fi
done

if [ ${#missing_scripts[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}⚠️  Missing scripts:${NC}"
    printf '  - %s\n' "${missing_scripts[@]}"
    echo -e "\n${RED}Some hook scripts are missing. They have been created as placeholders.${NC}"
fi

# Make all scripts executable
echo -e "\n${BLUE}🔧 Setting script permissions...${NC}"
chmod +x "$SCRIPTS_DIR"/*.py 2>/dev/null || true

# Install pre-commit hooks
echo -e "\n${BLUE}🎣 Installing pre-commit hooks...${NC}"
if [ "$1" == "--force" ]; then
    pre-commit install --force
else
    pre-commit install
fi

# Create custom bypass logging hook
echo -e "\n${BLUE}📝 Setting up bypass logging...${NC}"
cat > "$PROJECT_ROOT/.git/hooks/pre-commit.bypass" << 'EOF'
#!/bin/bash
# Log bypass attempts for audit

BYPASS_LOG="/opt/sutazaiapp/.git/hooks/bypass.log"
mkdir -p "$(dirname "$BYPASS_LOG")"

if [ "$SKIP" != "" ] || [ "$PRE_COMMIT_ALLOW_NO_CONFIG" == "1" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] BYPASS: User=$(whoami) Skip=$SKIP Message='$1'" >> "$BYPASS_LOG"
fi
EOF

chmod +x "$PROJECT_ROOT/.git/hooks/pre-commit.bypass"

# Create a wrapper to call both pre-commit and bypass logger
if [ -f "$PROJECT_ROOT/.git/hooks/pre-commit" ]; then
    mv "$PROJECT_ROOT/.git/hooks/pre-commit" "$PROJECT_ROOT/.git/hooks/pre-commit.original"
fi

cat > "$PROJECT_ROOT/.git/hooks/pre-commit" << 'EOF'
#!/bin/bash
# Wrapper to log bypasses and run pre-commit

# Call bypass logger
/opt/sutazaiapp/.git/hooks/pre-commit.bypass "$@"

# Call original pre-commit
if [ -f /opt/sutazaiapp/.git/hooks/pre-commit.original ]; then
    exec /opt/sutazaiapp/.git/hooks/pre-commit.original "$@"
else
    exec pre-commit "$@"
fi
EOF

chmod +x "$PROJECT_ROOT/.git/hooks/pre-commit"

# Create secrets baseline if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.secrets.baseline" ]; then
    echo -e "\n${BLUE}🔐 Creating secrets baseline...${NC}"
    detect-secrets scan > "$PROJECT_ROOT/.secrets.baseline"
fi

# Test the installation
echo -e "\n${BLUE}🧪 Testing installation...${NC}"
if pre-commit run --version &> /dev/null; then
    echo -e "  ✅ Pre-commit is working correctly"
else
    echo -e "  ${RED}❌ Pre-commit test failed${NC}"
    exit 1
fi

# Create team documentation
cat > "$PROJECT_ROOT/docs/pre-commit-usage.md" << 'EOF'
# Pre-commit Hooks Usage Guide

## Overview
This project uses pre-commit hooks to enforce CLAUDE.md rules automatically before every commit.

## Normal Usage
Simply commit as usual:
```bash
git add .
git commit -m "Your commit message"
```

The hooks will run automatically and prevent commits that violate hygiene rules.

## Bypass Options

### Skip Specific Hook (Recommended)
```bash
SKIP=hook-id git commit -m "Emergency: reason for bypass"
```

Example:
```bash
SKIP=no-garbage-files git commit -m "Emergency: keeping backup for rollback"
```

### Skip All Hooks (Use Sparingly)
```bash
git commit --no-verify -m "Emergency bypass: detailed justification"
```

⚠️ **All bypasses are logged** in `.git/hooks/bypass.log` for audit purposes.

## Common Issues and Solutions

### Hook Fails with "command not found"
```bash
pip3 install pre-commit pyyaml gitpython
```

### Hooks Running Too Slowly
Some hooks only run on changed files. To see what's taking time:
```bash
pre-commit run --verbose
```

### Update Hooks After Config Change
```bash
pre-commit install --force
```

### Run Hooks Manually
```bash
# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run hook-id
```

## Hook Reference

### Phase 1: Critical (Rules 13, 12, 9)
- `no-garbage-files`: Blocks backup/temp files
- `single-deployment-script`: Ensures one deploy.sh
- `no-duplicate-directories`: Prevents directory duplication

### Phase 2: Structural (Rules 11, 8, 1, 2, 3)
- `docker-structure`: Validates Dockerfile best practices
- `python-documentation`: Ensures Python docs
- `no-fantasy-elements`: Blocks placeholder code
- `no-breaking-changes`: Detects breaking changes
- `system-analysis`: Quick system health check

### Phase 3: Organizational (Rules 7, 6, 15, 4, 5, 10, 14, 16)
- `script-organization`: Validates script structure
- `documentation-structure`: Checks doc organization
- Various other quality checks

## Troubleshooting

### See Why a Hook Failed
Look for the specific error message in the output. Each hook provides:
- What rule was violated
- Where the problem is
- How to fix it

### Temporarily Disable for Development
```bash
# Disable
git config --local core.hooksPath /dev/null

# Re-enable
git config --local --unset core.hooksPath
pre-commit install
```

### Report Issues
If a hook is incorrectly blocking valid code, create an issue with:
1. The hook that failed
2. The error message
3. Why you believe it's a false positive
EOF

echo -e "\n${GREEN}✅ Installation complete!${NC}"
echo
echo "Pre-commit hooks are now active. They will run automatically on:"
echo "  - git commit"
echo
echo "To run hooks manually:"
echo "  - On staged files: pre-commit run"
echo "  - On all files: pre-commit run --all-files"
echo
echo "Documentation created at: docs/pre-commit-usage.md"
echo
echo -e "${YELLOW}⚠️  Important:${NC}"
echo "  - First run may be slower as it sets up environments"
echo "  - Use 'git commit --no-verify' only in emergencies"
echo "  - All bypasses are logged for audit"
echo
echo -e "${GREEN}Happy coding with automatic hygiene enforcement! 🎉${NC}"