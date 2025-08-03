# SutazAI automation system/advanced automation System - Deployment Instructions

## 🚀 Complete System Deployment

### Prerequisites
- Docker and Docker Compose installed
- At least 32GB RAM recommended
- 100GB+ free disk space
- Existing SutazAI application running at http://192.168.131.128:8501/

### Step 1: Deploy the Complete System

```bash
cd /opt/sutazaiapp
./deploy_complete_sutazai_system_improved.sh
```

This will:
- Deploy 30+ AI agents
- Pull and configure 11 Ollama models
- Set up vector databases (ChromaDB, Qdrant, FAISS)
- Configure LiteLLM proxy for OpenAI API compatibility
- Initialize monitoring stack (Prometheus, Grafana, Loki)
- Set up autonomous code improvement system
- Create service communication hub
- Generate Ollama configuration guide

### Step 2: Quick Verification

After deployment starts (wait ~5 minutes for services to stabilize):

```bash
./verify_deployment.sh
```

This provides a quick health check of core services.

### Step 3: Full System Test

For comprehensive testing:

```bash
python test_complete_agi_system.py
```

This will:
- Test all core services
- Verify all AI agents are running
- Test Ollama model availability
- Verify LiteLLM proxy functionality
- Test service orchestration
- Check vector databases
- Monitor stack validation

### Step 4: Access Services

#### Main Applications
- **SutazAI Frontend**: http://192.168.131.128:8501
- **Backend API**: http://192.168.131.128:8000
- **Service Hub**: http://192.168.131.128:8114

#### AI Agents
- **AutoGPT**: http://192.168.131.128:8080
- **CrewAI**: http://192.168.131.128:8096
- **Aider**: http://192.168.131.128:8095
- **GPT-Engineer**: http://192.168.131.128:8097
- **BigAGI**: http://192.168.131.128:8106
- **Dify**: http://192.168.131.128:8107
- **LangFlow**: http://192.168.131.128:8090
- **Flowise**: http://192.168.131.128:8099
- **n8n**: http://192.168.131.128:5678

#### Monitoring
- **Prometheus**: http://192.168.131.128:9090
- **Grafana**: http://192.168.131.128:3000
- **Service Health**: http://192.168.131.128:8114/health

### Step 5: Configure AI Agents

Each agent is pre-configured to use Ollama. For manual configuration:

1. **Dify**: 
   - Access http://192.168.131.128:8107
   - Add Ollama provider with URL: `http://ollama:11434`

2. **BigAGI**:
   - Access http://192.168.131.128:8106
   - Already configured for both LiteLLM proxy and direct Ollama

3. **n8n**:
   - Access http://192.168.131.128:5678
   - Use HTTP Request node with Ollama API

### Troubleshooting

1. **Services not starting**:
   ```bash
   docker-compose logs <service-name>
   docker ps -a | grep sutazai
   ```

2. **Ollama models missing**:
   ```bash
   docker exec sutazai-ollama ollama list
   docker exec sutazai-ollama ollama pull <model-name>
   ```

3. **Port conflicts**:
   Check `docker-compose.yml` and adjust port mappings if needed

4. **Memory issues**:
   ```bash
   docker stats
   # Reduce number of concurrent services if needed
   ```

### Documentation

- **Ollama Configuration**: `./docs/OLLAMA_AGENT_CONFIGURATION.md`
- **Deployment Summary**: `./DEPLOYMENT_SUMMARY.md`
- **Architecture**: `./docs/ARCHITECTURE.md`

### Notes

- All services run locally without external API dependencies
- OpenWebUI has been removed as requested
- All agents configured to use Ollama via direct connection or LiteLLM proxy
- Autonomous code improvement runs every 6 hours (configurable)
- Service hub provides centralized orchestration for all agents

### Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Run diagnostics: `./verify_deployment.sh`
3. Review test results: `test_results_detailed.json`