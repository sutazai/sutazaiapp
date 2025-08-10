#!/bin/bash

# Strict error handling
set -euo pipefail



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

echo "🗂️  Organizing SutazAI Project Structure..."
echo "=========================================="

# Create organized structure for scripts
echo "📁 Creating organized script directories..."

# Main script categories
mkdir -p scripts/{deployment,agents,models,utils,monitoring,docker,testing,demos}
mkdir -p scripts/deployment/{system,agents,infrastructure}
mkdir -p scripts/agents/{configuration,management,templates}
mkdir -p scripts/models/{ollama,training,optimization}
mkdir -p scripts/utils/{cleanup,verification,helpers}
mkdir -p scripts/monitoring/{health,logs,metrics}
mkdir -p scripts/docker/{build,compose,services}

# Create central archive location
echo "📦 Creating central archive directory..."
mkdir -p archive/{scripts,docs,backend,configs,old_deployments}
mkdir -p archive/scripts/{2024,2025}
mkdir -p archive/docs/{old_versions,project_history}

# Documentation categories
echo "📁 Creating organized documentation structure..."
mkdir -p docs/{system,agents,deployment,api,guides}
mkdir -p docs/system/{architecture,configuration,requirements}
mkdir -p docs/agents/{specifications,integrations,workflows}
mkdir -p docs/deployment/{docker,kubernetes,manual}
mkdir -p docs/api/{backend,frontend,mcp}
mkdir -p docs/guides/{quickstart,advanced,troubleshooting}

# Move scripts to appropriate locations
echo "🚚 Moving scripts to organized locations..."

# Deployment scripts
mv scripts/deploy_*.sh scripts/deployment/system/ 2>/dev/null
mv scripts/start_*.sh scripts/deployment/system/ 2>/dev/null
mv scripts/configure_all_agents.sh scripts/agents/configuration/ 2>/dev/null
mv scripts/update_agents_*.py scripts/agents/management/ 2>/dev/null

# Model-related scripts
mv scripts/ollama_*.sh scripts/models/ollama/ 2>/dev/null
mv scripts/update_*_to_tinyllama.* scripts/models/ollama/ 2>/dev/null

# Cleanup and verification scripts
mv scripts/remove_*.sh scripts/utils/cleanup/ 2>/dev/null
mv scripts/verify_*.sh scripts/utils/verification/ 2>/dev/null
mv scripts/fix_*.sh scripts/utils/helpers/ 2>/dev/null

# Docker-related scripts
mv scripts/docker_*.py scripts/docker/services/ 2>/dev/null

# Demo scripts
mv scripts/demos_*.* scripts/demos/ 2>/dev/null

# Consolidate ALL archives to central location
echo "📦 Consolidating all archives..."

# Move script archives
mv scripts/archive_*.* archive/scripts/2025/ 2>/dev/null
mv scripts/backend_archive_*.py archive/scripts/2025/ 2>/dev/null
mv scripts/archive/* archive/scripts/ 2>/dev/null
mv scripts/*backup*.* archive/scripts/2025/ 2>/dev/null
mv scripts/*old*.* archive/scripts/2025/ 2>/dev/null

# Move backend archives
if [ -d "backend/backend_archive" ]; then
    mv backend/backend_archive/* archive/backend/ 2>/dev/null
    rmdir backend/backend_archive 2>/dev/null
fi

# Move any archive directories from scripts
if [ -d "scripts/archive" ]; then
    mv scripts/archive/* archive/scripts/ 2>/dev/null
    rmdir scripts/archive 2>/dev/null
fi

# Move documentation archives
if [ -d "docs/archive" ]; then
    mv docs/archive/* archive/docs/old_versions/ 2>/dev/null
    rmdir docs/archive 2>/dev/null
fi

# Move any backup files
find . -name "*.backup" -o -name "*.bak" -o -name "*_backup_*" | while read file; do
    mv "$file" archive/old_deployments/ 2>/dev/null
done

# Move old deployment scripts with dates
find scripts -name "*_20[0-9][0-9]*" -type f | while read file; do
    mv "$file" archive/scripts/$(date +%Y)/ 2>/dev/null
done

# Move main verification scripts to root of utils
mv verify_*.sh scripts/utils/verification/ 2>/dev/null
mv final_litellm_cleanup.sh scripts/utils/cleanup/ 2>/dev/null 2>/dev/null
mv organize_project_structure.sh scripts/utils/ 2>/dev/null

echo "📚 Organizing documentation..."

# System documentation
mv docs/*ARCHITECTURE*.md docs/system/architecture/ 2>/dev/null
mv docs/*automation*.md docs/system/architecture/ 2>/dev/null
mv docs/*SYSTEM*.md docs/system/ 2>/dev/null

# Deployment docs
mv docs/*DEPLOYMENT*.md docs/deployment/ 2>/dev/null
mv docs/*DOCKER*.md docs/deployment/docker/ 2>/dev/null

# Guide documents
mv docs/*GUIDE*.md docs/guides/ 2>/dev/null
mv docs/*USAGE*.md docs/guides/ 2>/dev/null
mv docs/QUICK_START*.md docs/guides/quickstart/ 2>/dev/null

# Archive old docs (but skip .claude directory)
mv docs/archive/*.md archive/docs/old_versions/ 2>/dev/null

# Create index files
echo "📝 Creating index files..."

# Scripts index
cat > scripts/README.md << 'EOF'
# SutazAI Scripts Organization

## Directory Structure

### 🚀 deployment/
- **system/** - Main system deployment scripts
- **agents/** - Agent-specific deployment scripts
- **infrastructure/** - Infrastructure setup scripts

### 🤖 agents/
- **configuration/** - Agent configuration scripts
- **management/** - Agent lifecycle management
- **templates/** - Agent creation templates

### 🧠 models/
- **ollama/** - Ollama model management
- **training/** - Model training scripts
- **optimization/** - Model optimization tools

### 🛠️ utils/
- **cleanup/** - System cleanup utilities
- **verification/** - Verification and validation scripts
- **helpers/** - Helper utilities and fixes

### 📊 monitoring/
- **health/** - Health check scripts
- **logs/** - Log management tools
- **metrics/** - Metrics collection scripts

### 🐳 docker/
- **build/** - Docker build scripts
- **compose/** - Docker Compose utilities
- **services/** - Service-specific scripts

### 🧪 testing/
- Test scripts and automation

### 📦 archive/
- Archived and deprecated scripts

### 🎮 demos/
- Demo and example scripts

## Key Scripts

### System Deployment
- `deployment/system/deploy_complete_system.sh` - Main deployment script
- `deployment/system/start_tinyllama.sh` - Start system with tinyllama

### Agent Management
- `agents/configuration/configure_all_agents.sh` - Configure all agents
- `agents/management/update_agents_to_tinyllama.py` - Update agent models

### Verification
- `utils/verification/verify_tinyllama_config.sh` - Verify tinyllama setup
- `utils/verification/verify_litellm_removal.sh` - Verify LiteLLM removal

### Model Management
- `models/ollama/ollama_models_build_all_models.sh` - Build all Ollama models
EOF

# Documentation index
cat > docs/README.md << 'EOF'
# SutazAI Documentation

## Directory Structure

### 📐 system/
- **architecture/** - System architecture documentation
- **configuration/** - Configuration guides
- **requirements/** - System requirements

### 🤖 agents/
- **specifications/** - Individual agent specifications
- **integrations/** - Agent integration guides
- **workflows/** - Agent workflow documentation

### 🚀 deployment/
- **docker/** - Docker deployment guides
- **kubernetes/** - Kubernetes deployment (if applicable)
- **manual/** - Manual deployment instructions

### 🔌 api/
- **backend/** - Backend API documentation
- **frontend/** - Frontend API documentation
- **mcp/** - MCP server documentation

### 📚 guides/
- **quickstart/** - Quick start guides
- **advanced/** - Advanced usage guides
- **troubleshooting/** - Troubleshooting guides

### 📦 archive/
- Archived documentation

## Key Documents

### Getting Started
- `guides/quickstart/QUICK_START_ENHANCED.md` - Enhanced quick start guide
- `GPT-OSS_CONFIGURATION_COMPLETE.md` - tinyllama setup complete

### System Overview
- `system/architecture/OPTIMIZED_AGI_ARCHITECTURE_PLAN.md` - automation architecture
- `system/SUTAZAI_AGI_ASI_PROJECT_DOCUMENTATION.md` - Complete project docs

### Deployment
- `deployment/DEPLOYMENT_SUCCESS.md` - Deployment success guide
- `deployment/docker/DOCKER_DEPLOYMENT_GUIDE.md` - Docker deployment

### Agent Information
- `.claude/agents/` - Individual agent specifications
- `agents/AI_AGENT_CONTAINER_MAPPING.md` - Agent container mapping
EOF

# Create quick navigation script
cat > scripts/utils/navigate.sh << 'EOF'
#!/bin/bash
# Quick navigation helper for SutazAI project

case "$1" in
    deploy)
        cd scripts/deployment/system
        ;;
    agents)
        cd scripts/agents
        ;;
    models)
        cd scripts/models
        ;;
    docs)
        cd docs
        ;;
    *)
        echo "Usage: ./navigate.sh [deploy|agents|models|docs]"
        echo "Quick navigation to common directories"
        ;;
esac
EOF

chmod +x scripts/utils/navigate.sh

# Create archive index
cat > archive/README.md << 'EOF'
# SutazAI Archives

## Central Archive Repository

This directory contains all archived files from the SutazAI project, organized by type and date.

### 📁 Structure

- **scripts/** - Archived scripts
  - **2024/** - Scripts from 2024
  - **2025/** - Scripts from 2025
- **docs/** - Archived documentation
  - **old_versions/** - Previous documentation versions
  - **project_history/** - Historical project documents
- **backend/** - Archived backend code
- **configs/** - Old configuration files
- **old_deployments/** - Previous deployment scripts and backups

### ⚠️ Note
These files are archived for reference only. Use current versions in the main directories.

### 🔍 To restore a file:
```bash
cp archive/[path-to-file] [destination]
```
EOF

echo ""
echo "✅ Organization complete!"
echo ""
echo "📊 Summary:"
echo "  - Scripts organized into logical categories"
echo "  - Documentation structured by topic"
echo "  - Index files created for easy navigation"
echo "  - ALL archives consolidated in /archive directory"
echo "  - .claude directory preserved (not touched)"
echo ""
echo "📍 Quick access:"
echo "  - Main deployment: scripts/deployment/system/"
echo "  - Agent configs: scripts/agents/"
echo "  - Documentation: docs/"
echo "  - Utilities: scripts/utils/"
echo "  - Archives: archive/"
echo ""
echo "⚠️  Note: The .claude directory remains untouched as requested"