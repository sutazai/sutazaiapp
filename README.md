# SutazaiApp - Multi-Agent AI Platform

> **Version**: 1.0.0  
> **Status**: Production-Ready  
> **License**: Proprietary  
> **Last Updated**: 2025-11-13 21:30:00 UTC

---

## 🌟 Overview

SutazaiApp is a comprehensive multi-agent AI platform featuring JARVIS-style voice interaction, autonomous AI agents, and enterprise-grade infrastructure. The system leverages local LLM inference with Ollama, vector databases for semantic search, and a microservices architecture for scalability.

### Key Features

- **🎙️ JARVIS Voice Interface**: Voice-controlled AI assistant with wake word detection
- **🤖 Multi-Agent System**: 16+ specialized AI agents for various tasks
- **🧠 Local LLM Inference**: Privacy-first AI with Ollama and TinyLlama
- **📊 Vector Databases**: ChromaDB, Qdrant, and FAISS for semantic search
- **🔄 Service Mesh**: Kong API Gateway, Consul service discovery, RabbitMQ messaging
- **📈 Full Observability**: Prometheus metrics, Grafana dashboards, comprehensive logging
- **🐳 Container Management**: Unified Portainer stack for easy deployment

---

## 🚀 Quick Start

### Prerequisites

- **Docker** 20.10+ with Docker Compose v2
- **System**: 8+ CPU cores, 16+ GB RAM, 100+ GB disk
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, CentOS 8+)
- **Network**: Internet access for pulling images

### One-Command Deployment

```bash
# Clone repository
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp

# Deploy with automated script
./deploy-portainer.sh
```

### Manual Deployment

```bash
# Deploy stack
docker compose -f portainer-stack.yml up -d

# Initialize Ollama model
docker exec -it sutazai-ollama ollama pull tinyllama

# Check status
docker ps --filter "name=sutazai-"
```

---

## 📦 Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   JARVIS Frontend (11000)                    │
│                  Streamlit Voice Interface                   │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│              Kong API Gateway (10008/10009)                  │
│              Routes, Auth, Rate Limiting                     │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│              Backend API (10200) - FastAPI                   │
│          Service Orchestration & Business Logic              │
└───┬────────┬────────┬────────┬────────┬──────────────────────┘
    │        │        │        │        │
┌───▼───┐┌──▼───┐┌──▼───┐┌──▼───┐┌───▼────┐
│ Postgres││ Redis││ Neo4j││RabbitMQ││Consul│
│ (10000) ││(10001)││(10002)││(10004) ││(10006)│
│Database ││ Cache ││ Graph ││  Queue ││Service│
│         ││       ││       ││        ││ Mesh  │
└─────────┘└───────┘└───────┘└────────┘└───────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼────────┐    ┌────────▼─────────┐
│ Vector DBs     │    │ AI Services      │
│ - ChromaDB     │    │ - Ollama (11434) │
│ - Qdrant       │    │ - AI Agents      │
│ - FAISS        │    │   (11101-11999)  │
└────────────────┘    └──────────────────┘
```

### Network Architecture

- **Network**: `sutazai-network` (172.20.0.0/16)
- **Management**: 172.20.0.50-59 (Portainer)
- **Core**: 172.20.0.10-19 (Databases, Queue, Discovery)
- **Vectors**: 172.20.0.20-29 (ChromaDB, Qdrant, FAISS, Ollama)
- **Apps**: 172.20.0.30-39 (Backend, Frontend)
- **Monitoring**: 172.20.0.40-49 (Prometheus, Grafana)
- **Agents**: 172.20.0.100-199 (AI Agents)

---

## 🔌 Service Endpoints

| Service | URL | Credentials |
|---------|-----|-------------|
| **Portainer** | http://localhost:9000 | Set on first access |
| **JARVIS Frontend** | http://localhost:11000 | None |
| **Backend API** | http://localhost:10200 | JWT auth |
| **API Docs** | http://localhost:10200/docs | None |
| **Kong Proxy** | http://localhost:10008 | Via routes |
| **Kong Admin** | http://localhost:10009 | Internal |
| **RabbitMQ UI** | http://localhost:10005 | sutazai/sutazai_secure_2024 |
| **Neo4j Browser** | http://localhost:10002 | neo4j/sutazai_secure_2024 |
| **Consul UI** | http://localhost:10006 | None |
| **Grafana** | http://localhost:10201 | admin/sutazai_secure_2024 |
| **Prometheus** | http://localhost:10202 | None |
| **Ollama API** | http://localhost:11434 | None |

---

## 🤖 AI Agents

### Deployed Agents

| Agent | Port | Description | Status |
|-------|------|-------------|--------|
| **Letta** | 11101 | Memory AI (formerly MemGPT) | ✅ Active |
| **AutoGPT** | 11102 | Autonomous task execution | ✅ Active |
| **LocalAGI** | 11103 | Local AI orchestration | ✅ Active |
| **LangChain** | 11201 | LLM framework | ✅ Active |
| **Aider** | 11301 | AI pair programming | ✅ Active |
| **GPT-Engineer** | 11302 | Code generation | ✅ Active |
| **CrewAI** | 11401 | Multi-agent orchestration | ✅ Active |
| **Documind** | 11502 | Document processing | ✅ Active |
| **FinRobot** | 11601 | Financial analysis | ✅ Active |
| **ShellGPT** | 11701 | CLI assistant | ✅ Active |
| **Semgrep** | 11801 | Security analysis | ✅ Active |

### Agent Configuration

All agents use:
- **LLM Backend**: Ollama (http://sutazai-ollama:11434)
- **Default Model**: TinyLlama (1.1B parameters, 637MB)
- **Authentication**: JWT tokens from backend
- **Communication**: RabbitMQ message queue

---

## 📊 Monitoring

### Prometheus Metrics

Access Prometheus at: http://localhost:10202

Monitored services:
- All infrastructure components (PostgreSQL, Redis, Neo4j, RabbitMQ, Consul, Kong)
- Vector databases (ChromaDB, Qdrant, FAISS)
- AI services (Ollama, all agents)
- Application (Backend API, Frontend)

### Grafana Dashboards

Access Grafana at: http://localhost:10201 (admin/sutazai_secure_2024)

Pre-configured dashboards:
- System overview
- Container resource usage
- Service health status
- API performance
- Database metrics
- LLM inference latency

---

## 🛠️ Management

### Portainer

Access Portainer at: http://localhost:9000

Features:
- Container management (start, stop, restart, remove)
- Stack management (deploy, update, remove)
- Log viewer with real-time streaming
- Resource monitoring (CPU, memory, network)
- Volume management
- Network management

### Health Checks

```bash
# Quick health check
./scripts/health-check.sh

# Individual service checks
docker exec sutazai-postgres pg_isready -U jarvis
docker exec sutazai-redis redis-cli ping
docker exec sutazai-backend curl -f http://localhost:8000/health
docker exec sutazai-frontend curl -f http://localhost:11000/_stcore/health
```

### Logs

```bash
# View all logs
docker compose -f portainer-stack.yml logs -f

# View specific service logs
docker logs sutazai-backend -f
docker logs sutazai-frontend -f
docker logs sutazai-ollama -f

# Follow logs for multiple services
docker logs -f sutazai-postgres sutazai-redis sutazai-backend
```

---

## 🔧 Configuration

### Environment Variables

Key configuration in `.env` file (create if doesn't exist):

```bash
# Database
POSTGRES_PASSWORD=sutazai_secure_2024
NEO4J_PASSWORD=sutazai_secure_2024

# RabbitMQ
RABBITMQ_PASSWORD=sutazai_secure_2024

# JWT
SECRET_KEY=<generate-secure-key>

# Grafana
GRAFANA_PASSWORD=sutazai_secure_2024

# Ollama
OLLAMA_MODELS=/root/.ollama/models

# Features
ENABLE_VOICE_COMMANDS=true
ENABLE_GPU=false  # Set to true if NVIDIA GPU available
```

### Resource Limits

Adjust in `portainer-stack.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '2.0'
    reservations:
      memory: 512M
      cpus: '0.5'
```

---

## 📚 Documentation

### Core Documentation

- **[Portainer Deployment Guide](docs/PORTAINER_DEPLOYMENT_GUIDE.md)** - Complete deployment instructions
- **[Port Registry](IMPORTANT/ports/PortRegistry.md)** - All port assignments and network configuration
- **[TODO List](TODO.md)** - Development roadmap and pending tasks
- **[Changelog](CHANGELOG.md)** - Complete change history

### Additional Resources

- **Project Wiki**: https://deepwiki.com/sutazai/sutazaiapp
- **Architecture Documentation**: `/docs/architecture/`
- **API Reference**: http://localhost:10200/docs (when running)

---

## 🔒 Security

### Production Deployment

**⚠️ IMPORTANT**: Before deploying to production:

1. **Change all default passwords** in `portainer-stack.yml`
2. **Generate secure JWT secret**: `openssl rand -hex 32`
3. **Enable SSL/TLS** for all public endpoints
4. **Configure firewall** rules to restrict access
5. **Set up backup procedures** for all volumes
6. **Enable authentication** on all admin interfaces
7. **Review and harden** security settings

### Security Best Practices

- Use Kong API Gateway for rate limiting and authentication
- Enable network policies for service isolation
- Regularly update all container images
- Monitor security alerts via Prometheus/Grafana
- Implement proper logging and audit trails
- Use Semgrep agent for continuous security scanning

---

## 🐛 Troubleshooting

### Common Issues

#### Services Won't Start

```bash
# Check logs
docker logs <container-name>

# Check resource usage
docker stats

# Restart service
docker restart <container-name>
```

#### Port Conflicts

```bash
# Find process using port
sudo lsof -i :PORT
sudo netstat -tulpn | grep :PORT

# Change port in portainer-stack.yml
```

#### Ollama Connection Issues

```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Check if model is loaded
docker exec sutazai-ollama ollama list

# Pull model if missing
docker exec sutazai-ollama ollama pull tinyllama
```

#### Memory Issues

```bash
# Check system memory
free -h

# Reduce resource limits in portainer-stack.yml
# Restart services with lower limits
```

For more troubleshooting, see [PORTAINER_DEPLOYMENT_GUIDE.md](docs/PORTAINER_DEPLOYMENT_GUIDE.md#troubleshooting)

---

## 🔄 Updates

### Updating the Stack

```bash
# Pull latest changes
git pull origin main

# Pull updated images
docker compose -f portainer-stack.yml pull

# Update stack
docker compose -f portainer-stack.yml up -d

# Clean up old images
docker image prune -f
```

### Updating via Portainer

1. Go to **Stacks** → **sutazaiapp**
2. Click **Editor**
3. Make changes or pull from Git
4. Click **Update the stack**
5. Select "Re-pull images and redeploy"

---

## 📊 Project Status

### Current Phase

**Phase 6**: AI Agent Deployment - Lightweight Strategy

### Completion Status

- ✅ **Phase 1**: Core Infrastructure (100%)
- ✅ **Phase 2**: Service Layer (100%)
- ✅ **Phase 3**: API Gateway & Vector DBs (100%)
- ✅ **Phase 4**: Backend Application (100%)
- ✅ **Phase 5**: Frontend & Voice Interface (90%)
- ✅ **Phase 6**: AI Agents Setup (80%)
- ✅ **Phase 7**: MCP Bridge Services (90%)
- ⏳ **Phase 8**: Monitoring Stack (50%)
- ⏳ **Phase 9**: Integration Testing (0%)
- ⏳ **Phase 10**: Documentation & Cleanup (60%)

### System Health

| Component | Status | Health |
|-----------|--------|--------|
| Core Infrastructure | Running | ✅ Healthy |
| Vector Databases | Running | ✅ Healthy |
| Backend API | Running | ✅ Healthy |
| Frontend | Running | ⚠️ Needs Review |
| AI Agents (16) | Running | ⚠️ Degraded |
| MCP Bridge | Running | ⚠️ Needs Review |
| Monitoring | Running | ✅ Healthy |

---

## 🤝 Contributing

This is a proprietary project. For authorized contributors:

1. Create feature branch from `copilot/fix-issues-and-improve-performance`
2. Follow [Rules.md](IMPORTANT/Rules.md) guidelines
3. Test all changes thoroughly
4. Update documentation
5. Create pull request with detailed description

---

## 📝 License

Proprietary - © 2025 SutazaiApp Development Team

---

## 🆘 Support

For issues, questions, or feature requests:

1. Check documentation in `/docs` directory
2. Review [TODO.md](TODO.md) for known issues
3. Consult [PORTAINER_DEPLOYMENT_GUIDE.md](docs/PORTAINER_DEPLOYMENT_GUIDE.md)
4. Check project wiki: https://deepwiki.com/sutazai/sutazaiapp

---

**Made with ❤️ by the SutazaiApp Team**
