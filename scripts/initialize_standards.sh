#!/bin/bash
# Purpose: Initialize all codebase standards tooling and enforcement
# Usage: ./scripts/initialize_standards.sh
# Requires: npm, pip, docker

set -euo pipefail

echo "🚀 Initializing Codebase Standards Enforcement System"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    local missing_deps=()
    
    command -v npm >/dev/null 2>&1 || missing_deps+=("npm")
    command -v python3 >/dev/null 2>&1 || missing_deps+=("python3")
    command -v pip >/dev/null 2>&1 || missing_deps+=("pip")
    command -v docker >/dev/null 2>&1 || missing_deps+=("docker")
    command -v git >/dev/null 2>&1 || missing_deps+=("git")
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    print_status "All prerequisites installed"
}

# Install JavaScript tooling
install_js_tools() {
    echo -e "\n📦 Installing JavaScript quality tools..."
    
    npm install -D \
        eslint \
        prettier \
        @typescript-eslint/parser \
        @typescript-eslint/eslint-plugin \
        husky \
        lint-staged \
        @commitlint/cli \
        @commitlint/config-conventional \
        jest \
        @testing-library/react \
        @testing-library/jest-dom \
        nyc \
        jscpd \
        || print_error "Failed to install JS tools"
    
    print_status "JavaScript tools installed"
}

# Install Python tooling
install_python_tools() {
    echo -e "\n🐍 Installing Python quality tools..."
    
    pip install --upgrade \
        black \
        flake8 \
        mypy \
        bandit \
        safety \
        pytest \
        pytest-cov \
        pytest-asyncio \
        vulture \
        radon \
        pylint \
        || print_error "Failed to install Python tools"
    
    print_status "Python tools installed"
}

# Install security scanning tools
install_security_tools() {
    echo -e "\n🔒 Installing security tools..."
    
    # Install Trivy for container scanning
    if ! command -v trivy &> /dev/null; then
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    fi
    
    # Install semgrep
    pip install semgrep || print_error "Failed to install semgrep"
    
    print_status "Security tools installed"
}

# Configure pre-commit hooks
setup_precommit() {
    echo -e "\n🪝 Setting up pre-commit hooks..."
    
    pip install pre-commit
    
    cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key
      
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3
        
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203']
        
  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.56.0
    hooks:
      - id: eslint
        files: \.(js|jsx|ts|tsx)$
        additional_dependencies:
          - eslint@8.56.0
          - eslint-config-prettier@9.1.0
          
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.1.0
    hooks:
      - id: prettier
        files: \.(js|jsx|ts|tsx|css|scss|json|md)$
EOF

    pre-commit install
    print_status "Pre-commit hooks configured"
}

# Create ESLint configuration
setup_eslint() {
    echo -e "\n📝 Configuring ESLint..."
    
    cat > .eslintrc.json << 'EOF'
{
  "parser": "@typescript-eslint/parser",
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "prettier"
  ],
  "plugins": ["@typescript-eslint"],
  "env": {
    "browser": true,
    "node": true,
    "es2021": true
  },
  "rules": {
    "no-console": "warn",
    "no-unused-vars": "off",
    "@typescript-eslint/no-unused-vars": "error",
    "complexity": ["error", 10],
    "max-lines": ["warn", 300],
    "max-lines-per-function": ["warn", 50]
  }
}
EOF
    
    print_status "ESLint configured"
}

# Create Prettier configuration
setup_prettier() {
    echo -e "\n💅 Configuring Prettier..."
    
    cat > .prettierrc.json << 'EOF'
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 88,
  "tabWidth": 2,
  "useTabs": false,
  "arrowParens": "always",
  "endOfLine": "lf"
}
EOF
    
    print_status "Prettier configured"
}

# Setup Python configurations
setup_python_configs() {
    echo -e "\n🐍 Configuring Python tools..."
    
    # Black configuration
    cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --cov=backend --cov-report=html --cov-report=term"
testpaths = ["tests", "backend/tests"]

[tool.coverage.run]
source = ["backend"]
omit = ["*/tests/*", "*/migrations/*", "*/__pycache__/*"]

[tool.coverage.report]
fail_under = 80
EOF
    
    # Flake8 configuration
    cat > .flake8 << 'EOF'
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist
max-complexity = 10
EOF
    
    print_status "Python tools configured"
}

# Create commit message template
setup_commit_template() {
    echo -e "\n📝 Setting up commit standards..."
    
    cat > .gitmessage << 'EOF'
# <type>(<scope>): <subject>
#
# <body>
#
# <footer>
#
# Type: feat, fix, docs, style, refactor, test, chore
# Scope: backend, frontend, docker, scripts, etc.
# Subject: imperative mood, max 50 chars
# Body: explain what and why, not how
# Footer: issues closed, breaking changes
EOF

    git config commit.template .gitmessage
    
    print_status "Commit template configured"
}

# Create documentation check script
create_doc_checker() {
    echo -e "\n📚 Creating documentation checker..."
    
    cat > scripts/check_documentation.py << 'EOF'
#!/usr/bin/env python3
"""
Purpose: Verify documentation standards compliance
Usage: python scripts/check_documentation.py
Requirements: Python 3.8+
"""

import os
import sys
from pathlib import Path

def check_docs():
    """Check if all required documentation exists and is up to date."""
    required_docs = [
        "README.md",
        "docs/overview.md",
        "docs/setup/local_dev.md",
        "docs/api/endpoints.md",
        "CHANGELOG.md"
    ]
    
    missing = []
    for doc in required_docs:
        if not Path(doc).exists():
            missing.append(doc)
    
    if missing:
        print(f"❌ Missing required documentation: {missing}")
        return False
    
    print("✅ All required documentation present")
    return True

def check_script_headers():
    """Verify all scripts have proper documentation headers."""
    scripts_dir = Path("scripts")
    undocumented = []
    
    for script in scripts_dir.rglob("*.py"):
        with open(script, 'r') as f:
            content = f.read()
            if not ("Purpose:" in content and "Usage:" in content):
                undocumented.append(str(script))
    
    if undocumented:
        print(f"❌ Scripts without proper headers: {undocumented}")
        return False
    
    print("✅ All scripts properly documented")
    return True

if __name__ == "__main__":
    if not (check_docs() and check_script_headers()):
        sys.exit(1)
EOF

    chmod +x scripts/check_documentation.py
    print_status "Documentation checker created"
}

# Create cleanup validation script
create_cleanup_validator() {
    echo -e "\n🧹 Creating cleanup validator..."
    
    cat > scripts/validate_cleanup.py << 'EOF'
#!/usr/bin/env python3
"""
Purpose: Validate file deletions and cleanups are safe
Usage: python scripts/validate_cleanup.py <file_to_delete>
Requirements: Python 3.8+, grep
"""

import sys
import subprocess
from pathlib import Path

def check_references(filepath):
    """Check if file is referenced anywhere in codebase."""
    filename = Path(filepath).name
    
    # Search for imports or references
    cmd = f"grep -r '{filename}' . --exclude-dir=.git --exclude-dir=node_modules"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(f"⚠️  File '{filepath}' is referenced in:")
        print(result.stdout)
        return False
    
    print(f"✅ No references found for '{filepath}'")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_cleanup.py <file>")
        sys.exit(1)
    
    if not check_references(sys.argv[1]):
        print("❌ Unsafe to delete - file has references")
        sys.exit(1)
EOF

    chmod +x scripts/validate_cleanup.py
    print_status "Cleanup validator created"
}

# Setup GitHub Actions workflow
setup_github_actions() {
    echo -e "\n🤖 Setting up GitHub Actions..."
    
    mkdir -p .github/workflows
    
    cat > .github/workflows/standards-enforcement.yml << 'EOF'
name: Codebase Standards Enforcement

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  quality-gates:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install black flake8 mypy bandit safety pytest pytest-cov
        npm ci
        
    - name: Run security checks
      run: |
        bandit -r backend/ -f json -o bandit-report.json
        safety check --json
        
    - name: Check code quality
      run: |
        black --check backend/
        flake8 backend/
        mypy backend/
        
    - name: Run tests with coverage
      run: |
        pytest --cov=backend --cov-fail-under=80
        
    - name: Check documentation
      run: python scripts/check_documentation.py
      
    - name: Build Docker images
      run: docker-compose build --parallel
      
    - name: Run container security scan
      run: |
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          aquasec/trivy image --severity HIGH,CRITICAL \
          --exit-code 1 sutazai-backend:latest
EOF
    
    print_status "GitHub Actions configured"
}

# Main execution
main() {
    echo "==========================================
    CODEBASE STANDARDS INITIALIZATION
=========================================="
    
    check_prerequisites
    
    # Install all tools
    install_js_tools
    install_python_tools
    install_security_tools
    
    # Configure tools
    setup_precommit
    setup_eslint
    setup_prettier
    setup_python_configs
    setup_commit_template
    
    # Create helper scripts
    create_doc_checker
    create_cleanup_validator
    
    # Setup CI/CD
    setup_github_actions
    
    echo -e "\n✨ ${GREEN}Codebase standards initialization complete!${NC}"
    echo -e "\n📋 Next steps:"
    echo "  1. Review and adjust configurations as needed"
    echo "  2. Run 'pre-commit run --all-files' to check existing code"
    echo "  3. Deploy AI agents using 'python scripts/deploy_standards_agents.py'"
    echo "  4. Monitor compliance dashboard at http://localhost:3000/standards"
}

# Run main function
main "$@"