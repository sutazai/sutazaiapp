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

### 🔍 verification/
- **verify-hygiene-monitoring-system.py** - Comprehensive hygiene monitoring system verification
- **README_HYGIENE_VERIFICATION.md** - Hygiene verification documentation

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
- `deployment/system/start_tinyllama.sh` - Start system with TinyLlama

### Agent Management
- `agents/configuration/configure_all_agents.sh` - Configure all agents
- `agents/management/update_agents_to_tinyllama.py` - Update agent models

### Verification
- `utils/verification/verify_tinyllama_config.sh` - Verify TinyLlama setup
- `utils/verification/verify_litellm_removal.sh` - Verify LiteLLM removal

### Model Management
- `models/ollama/ollama_models_build_all_models.sh` - Build all Ollama models
