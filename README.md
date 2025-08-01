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
