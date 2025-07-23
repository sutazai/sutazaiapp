# SutazAI AGI/ASI Autonomous System - Complete Implementation

## 🚀 Overview

SutazAI is a comprehensive AGI/ASI autonomous system that integrates multiple AI agents, models, and services to create a self-improving, intelligent platform. The system features over 20 specialized AI agents working together through a central orchestrator.

## 🏗️ System Architecture

### Core Components

1. **Model Management**
   - Ollama (Local LLM hosting)
   - DeepSeek R1 (Advanced reasoning)
   - Qwen 2.5 (General AI tasks)
   - CodeLlama (Code generation)
   - Multiple specialized models

2. **AI Agents**
   - **AutoGPT** - Task automation and planning
   - **LocalAGI** - Autonomous AI orchestration
   - **TabbyML** - Code completion
   - **Semgrep** - Code security analysis
   - **BrowserUse** - Web automation
   - **Skyvern** - Advanced web scraping
   - **Documind** - Document processing
   - **FinRobot** - Financial analysis
   - **GPT-Engineer** - Code generation
   - **Aider** - AI pair programming
   - **BigAGI** - Advanced conversational AI
   - **AgentZero** - Autonomous agent
   - **LangFlow** - Visual workflow builder
   - **Dify** - App builder
   - **AutoGen** - Multi-agent collaboration
   - **CrewAI** - Team collaboration
   - **AgentGPT** - Web-based autonomous agent
   - **PrivateGPT** - Private LLM interface
   - **LlamaIndex** - Data indexing and RAG
   - **FlowiseAI** - Chatflow builder

3. **Infrastructure**
   - PostgreSQL (Main database)
   - Redis (Caching and queuing)
   - Qdrant (Vector database)
   - ChromaDB (Vector storage)
   - FAISS (Fast similarity search)
   - Prometheus & Grafana (Monitoring)
   - Nginx (Reverse proxy)

4. **Self-Improvement System**
   - Automatic code analysis
   - AI-powered code generation
   - Continuous improvement loop
   - Version control integration

## 📋 Prerequisites

- **Hardware Requirements:**
  - CPU: 8+ cores recommended
  - RAM: 32GB minimum (64GB recommended)
  - Storage: 100GB+ available space
  - GPU: Optional but recommended for faster inference

- **Software Requirements:**
  - Ubuntu 20.04+ or similar Linux distribution
  - Docker 20.10+
  - Docker Compose 2.0+
  - Git
  - curl

## 🛠️ Installation

### Quick Start

```bash
# Clone the repository
cd /opt/sutazaiapp

# Run the complete deployment script
sudo ./deploy_complete_sutazai_system_v11.sh

# Setup AI models (optional - for additional models)
./setup_models.sh
```

### Manual Installation

1. **Install Prerequisites:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

2. **Deploy Services:**
```bash
# Build and start all services
docker-compose up -d --build

# Check service status
docker-compose ps
```

## 🎯 Usage

### Access Points

- **Main Application:** http://localhost:8501
- **API Backend:** http://localhost:8000
- **Grafana Dashboard:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090

### AI Agent Interfaces

- **AutoGPT:** http://localhost:8080
- **BigAGI:** http://localhost:8090
- **LangFlow:** http://localhost:7860
- **Dify:** http://localhost:5001
- **AgentGPT:** http://localhost:8103
- **FlowiseAI:** http://localhost:8106

### API Examples

```python
# Chat with the system
import requests

response = requests.post("http://localhost:8000/api/chat", json={
    "message": "Generate a Python web scraper",
    "model": "codellama:7b"
})
print(response.json())

# Execute agent task
response = requests.post("http://localhost:8000/api/agents/execute", json={
    "task_type": "code_generation",
    "task_data": {
        "description": "Create a REST API with FastAPI",
        "language": "python"
    }
})
print(response.json())
```

### WebSocket Chat

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
    ws.send(JSON.stringify({
        message: "Hello, SutazAI!",
        model: "llama3.2:3b"
    }));
};

ws.onmessage = (event) => {
    console.log('Response:', JSON.parse(event.data));
};
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Database
DATABASE_URL=postgresql://sutazai:sutazai_password@postgres:5432/sutazai
REDIS_URL=redis://redis:6379

# Vector Databases
QDRANT_URL=http://qdrant:6333
CHROMADB_URL=http://chromadb:8000

# Model Management
OLLAMA_URL=http://ollama:11434
AUTO_PULL_MODELS=true

# API Keys (if using external services)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

### Model Configuration

Edit `docker/enhanced-model-manager/config.yaml`:

```yaml
models:
  default: llama3.2:3b
  code: codellama:7b
  reasoning: deepseek-r1:8b
  embeddings: nomic-embed-text

parameters:
  temperature: 0.7
  max_tokens: 2000
  top_p: 0.9
```

## 🤖 Self-Improvement System

The system includes an autonomous self-improvement capability:

1. **Automatic Code Analysis**
   - Scans codebase for improvement opportunities
   - Identifies complexity and quality issues
   - Suggests optimizations

2. **AI-Powered Code Generation**
   - Generates improved code versions
   - Creates documentation
   - Writes test cases

3. **Continuous Improvement Loop**
   - Runs daily analysis
   - Creates improvement branches
   - Requires approval for changes

Enable self-improvement:
```bash
docker exec sutazai-backend python -c "
from backend.self_improvement_system import get_self_improvement_system
system = get_self_improvement_system()
asyncio.run(system.continuous_improvement_loop())
"
```

## 📊 Monitoring

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Login: admin/admin
3. Available dashboards:
   - System Overview
   - Agent Performance
   - Model Usage
   - API Metrics

### Health Checks

```bash
# Check all services
curl http://localhost:8000/api/health/all

# Check specific agent
curl http://localhost:8000/api/agents/autogpt/health

# System metrics
curl http://localhost:9090/metrics
```

## 🛡️ Security

### Best Practices

1. **Change default passwords**
   ```bash
   # Update PostgreSQL password
   docker exec -it sutazai-postgres psql -U sutazai -c "ALTER USER sutazai PASSWORD 'new_password';"
   ```

2. **Enable SSL/TLS**
   - Configure Nginx with SSL certificates
   - Use HTTPS for all external communications

3. **API Authentication**
   - Enable API key authentication
   - Use JWT tokens for session management

4. **Network Security**
   - Use Docker networks for isolation
   - Configure firewall rules
   - Limit exposed ports

## 🔨 Development

### Adding New Agents

1. Create Dockerfile:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "agent_server.py"]
```

2. Add to docker-compose.yml:
```yaml
new-agent:
  build: ./docker/new-agent
  container_name: sutazai-new-agent
  ports:
    - "8200:8080"
  networks:
    - sutazai-network
```

3. Register in agent integration:
```python
AgentType.NEW_AGENT: AgentConfig(
    name="NewAgent",
    type=AgentType.NEW_AGENT,
    url="http://new-agent:8080",
    port=8200,
    capabilities=["new_capability"],
    health_endpoint="/health"
)
```

### Testing

```bash
# Run unit tests
docker exec sutazai-backend pytest tests/

# Integration tests
./scripts/run_integration_tests.sh

# Load testing
docker run -v $PWD:/scripts -it loadimpact/k6 run /scripts/load_test.js
```

## 🚨 Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check logs
   docker-compose logs [service-name]
   
   # Restart service
   docker-compose restart [service-name]
   ```

2. **Model download fails**
   ```bash
   # Manual model pull
   docker exec sutazai-ollama ollama pull model-name
   ```

3. **Database connection issues**
   ```bash
   # Check PostgreSQL
   docker exec -it sutazai-postgres psql -U sutazai -d sutazai
   ```

4. **Memory issues**
   ```bash
   # Check resource usage
   docker stats
   
   # Limit service resources in docker-compose.yml
   ```

### Logs

```bash
# View all logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f sutazai-backend

# Save logs to file
docker-compose logs > sutazai_logs_$(date +%Y%m%d).txt
```

## 📦 Backup & Restore

### Backup

```bash
# Backup script
./backup_system.sh

# Manual backup
docker exec sutazai-postgres pg_dump -U sutazai sutazai > backup.sql
docker run --rm -v sutazai_models-data:/data -v $PWD:/backup alpine tar czf /backup/models_backup.tar.gz /data
```

### Restore

```bash
# Restore database
docker exec -i sutazai-postgres psql -U sutazai sutazai < backup.sql

# Restore models
docker run --rm -v sutazai_models-data:/data -v $PWD:/backup alpine tar xzf /backup/models_backup.tar.gz -C /
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude
- Meta for Llama models
- All open-source AI projects integrated

---

**Note:** This is an advanced AI system. Use responsibly and ensure compliance with all applicable laws and regulations.

For support: [Create an issue](https://github.com/sutazai/issues)