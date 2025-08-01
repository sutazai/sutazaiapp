# SutazAI Quick Reference

## 🎯 Common Commands

### Using Make (Recommended)
```bash
make help       # Show all commands
make deploy     # Full deployment
make up         # Start services
make down       # Stop services
make status     # Check status
make logs       # View logs
make verify     # Verify setup
```

### Direct Commands
```bash
# Start TinyLlama system
./scripts/deployment/system/start_tinyllama.sh

# Verify configuration
./scripts/utils/verification/verify_tinyllama_config.sh
./scripts/utils/verification/verify_litellm_removal.sh
```

## 📁 Where Things Are

### Actual Locations
- **Makefile**: `build/make/Makefile`
- **Docker Compose**: `config/docker/docker-compose.tinyllama.yml`
- **Deployment Scripts**: `scripts/deployment/system/`
- **Agent Configs**: `agents/configs/`
- **Documentation**: `docs/`
- **Archives**: `archive/`

### Symlinks in Root (for convenience)
- `Makefile` → `build/make/Makefile`
- `docker-compose.yml` → `config/docker/docker-compose.yml`
- `docker-compose.tinyllama.yml` → `config/docker/docker-compose.tinyllama.yml`

## 🚀 Quick Start
1. `make setup` - Create directories
2. `make deploy` - Deploy system
3. `make status` - Check if running

## 🔍 Quick Navigation
```bash
cd scripts/deployment/system    # Deployment scripts
cd scripts/agents              # Agent scripts
cd docs/guides/quickstart      # Quick start guides
cd config/docker               # Docker configs
```

## 💡 Tips
- Use `make help` to see all available commands
- Symlinks allow you to use familiar commands from root
- All archives are in `/archive` directory
- The `.claude` directory is preserved and untouched