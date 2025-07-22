# SutazAI AGI/ASI System - Automated Real Agent Deployment

## 🎯 Quick Start (One Command)

```bash
./deploy_automated_sutazai_system.sh
```

**That's it!** This single command will:
- ✅ Deploy ALL real AI agents (no mocks)
- ✅ Set up AI models (DeepSeek R1, Qwen3, Llama 3.2)
- ✅ Configure infrastructure (PostgreSQL, Redis, Qdrant, ChromaDB)
- ✅ Start monitoring and health checks
- ✅ Verify everything works

## 🚀 Daily Usage

```bash
# Start the system (with real agents)
./start_sutazai_with_real_agents.sh

# Access the system
# Main UI: http://localhost:8501
# OpenWebUI: http://localhost:8089
# Backend API: http://localhost:8000
```

## 🤖 Real Agents Included

All agents are **REAL implementations** from official repositories:

- **OpenWebUI** - Advanced chat interface with real models
- **TabbyML** - Real code completion and suggestions  
- **LangFlow** - Visual workflow orchestration
- **Dify** - App development platform
- **Browserless** - Real web automation
- **Enhanced Orchestrator** - Coordinates all agents

## 🧠 AI Models

- **DeepSeek R1 8B** - Advanced reasoning
- **Qwen3 8B** - Multilingual capabilities
- **Llama 3.2 1B** - Fast general-purpose

## 📋 Requirements

- Docker & Docker Compose
- 8GB+ RAM (16GB recommended)
- 50GB+ disk space

## 🔧 Management Commands

```bash
# Start system
./start_sutazai_with_real_agents.sh

# Check status
docker-compose -f docker-compose-real-agents.yml ps

# View logs
docker-compose -f docker-compose-real-agents.yml logs -f

# Stop system
docker-compose -f docker-compose-real-agents.yml down

# Full redeploy
./deploy_automated_sutazai_system.sh
```

## 📚 Full Documentation

See `REAL_AGENTS_DEPLOYMENT.md` for complete documentation including:
- Manual deployment steps
- Troubleshooting guide
- API reference
- Security configuration
- Advanced customization

## 🎉 Key Features

- **100% Automated** - One command deployment
- **Real Agents Only** - No mock implementations
- **Production Ready** - Health checks, monitoring, logging
- **Reproducible** - Same deployment every time
- **Self-Documenting** - Generates deployment reports
- **Recovery Built-in** - Automatic error handling

---

**No configuration required!** The automation handles everything.