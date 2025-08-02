#!/bin/bash

echo "🧹 Organizing root directory files..."
echo "===================================="

# Create necessary directories
mkdir -p build/make
mkdir -p config/{docker,project,security}
mkdir -p reports/security

# Move Makefile to build directory
echo "📁 Moving build files..."
mv Makefile build/make/ 2>/dev/null

# Move Docker Compose files to config/docker
echo "🐳 Moving Docker configurations..."
mv docker-compose.yml config/docker/ 2>/dev/null
mv docker-compose.tinyllama.yml config/docker/ 2>/dev/null

# Move project configuration files
echo "📋 Moving project configurations..."
mv package.json config/project/ 2>/dev/null
mv pyproject.toml config/project/ 2>/dev/null

# Move security reports
echo "🔒 Moving security reports..."
mv *_audit.json reports/security/ 2>/dev/null
mv *_safety.json reports/security/ 2>/dev/null
mv agent_validation_report.json reports/security/ 2>/dev/null
mv semgrep_custom_rules.yaml config/security/ 2>/dev/null

# Move any remaining status/report files
echo "📊 Moving status files..."
mv TINYLLAMA_CONFIGURATION_COMPLETE.md docs/system/ 2>/dev/null
mv COMPLETE_TINYLLAMA_LITELLM_REMOVAL.md docs/system/ 2>/dev/null
mv PROJECT_ORGANIZATION_COMPLETE.md docs/system/ 2>/dev/null

# Create symlinks for commonly used files
echo "🔗 Creating convenient symlinks..."
ln -sf config/docker/docker-compose.yml docker-compose.yml
ln -sf config/docker/docker-compose.tinyllama.yml docker-compose.tinyllama.yml
ln -sf build/make/Makefile Makefile

# Create a root README for navigation
cat > README.md << 'EOF'
# SutazAI AGI/ASI Autonomous System

A lightweight, fully local AGI system running on TinyLlama.

## 🚀 Quick Start

```bash
# Using Make commands (symlinked for convenience)
make setup      # Initial setup
make deploy     # Deploy the system
make status     # Check status
make verify     # Verify configuration
```

## 📁 Project Structure

```
.
├── agents/         # Agent configurations and definitions
├── archive/        # Archived files (scripts, docs, configs)
├── backend/        # Backend API service
├── brain/          # AGI brain components
├── build/          # Build tools and Makefile
├── config/         # All configuration files
│   ├── docker/     # Docker compose files
│   ├── project/    # Project configs (package.json, pyproject.toml)
│   └── security/   # Security configurations
├── data/           # Application data
├── docs/           # Documentation
├── frontend/       # Frontend UI
├── localagi/       # LocalAGI components
├── logs/           # Application logs
├── models/         # AI models
├── monitoring/     # Prometheus, Grafana configs
├── ollama/         # Ollama model definitions
├── reports/        # Generated reports
├── scripts/        # Organized scripts
└── tests/          # Test suites
```

## 🔧 Configuration Files

- **Docker Compose**: `config/docker/docker-compose.tinyllama.yml`
- **Makefile**: `build/make/Makefile`
- **Environment**: `.env`, `.env.tinyllama`

## 📚 Documentation

- Quick Start: `docs/guides/quickstart/`
- Architecture: `docs/system/architecture/`
- Agent Specs: `.claude/agents/` (preserved)

## 🎯 Key Commands

```bash
# Deployment
./scripts/deployment/system/start_tinyllama.sh

# Verification
./scripts/utils/verification/verify_tinyllama_config.sh
./scripts/utils/verification/verify_litellm_removal.sh

# Using Make (recommended)
make help       # Show all available commands
```

## 🤖 System Status

- **Model**: TinyLlama (637MB)
- **API**: Native Ollama
- **Agents**: 37 specialized AI agents
- **External APIs**: None (100% local)

---

For detailed documentation, see the `docs/` directory.
EOF

echo ""
echo "✅ Root directory organized!"
echo ""
echo "📊 Changes made:"
echo "  - Makefile → build/make/"
echo "  - docker-compose files → config/docker/"
echo "  - Project configs → config/project/"
echo "  - Security reports → reports/security/"
echo "  - Status docs → docs/system/"
echo ""
echo "🔗 Symlinks created for convenience:"
echo "  - Makefile (→ build/make/Makefile)"
echo "  - docker-compose.yml (→ config/docker/docker-compose.yml)"
echo "  - docker-compose.tinyllama.yml (→ config/docker/docker-compose.tinyllama.yml)"
echo ""
echo "📝 New README.md created with project overview"