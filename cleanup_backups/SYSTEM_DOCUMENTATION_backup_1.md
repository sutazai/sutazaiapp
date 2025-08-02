# SutazAI AGI/ASI System Documentation v10.0

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Core Components](#core-components)
5. [AI Agents](#ai-agents)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Security](#security)

## System Overview

SutazAI is an enterprise-grade Autonomous General Intelligence (AGI) and Autonomous Super Intelligence (ASI) system designed for comprehensive AI capabilities including:

- **Multi-Agent Orchestration**: Coordinate multiple specialized AI agents
- **Natural Language Processing**: Advanced conversational AI
- **Code Generation**: Automated software development
- **Document Analysis**: Intelligent document processing
- **Knowledge Management**: Vector-based knowledge storage and retrieval
- **Real-time Communication**: WebSocket support for live updates

### Key Features

- 🚀 **High Performance**: Optimized for production workloads
- 🔒 **Security First**: Enterprise-grade security features
- 📊 **Comprehensive Monitoring**: Real-time metrics and alerts
- 🤖 **20+ AI Agents**: Specialized agents for various tasks
- 💾 **Multiple Storage Backends**: PostgreSQL, Redis, Qdrant, ChromaDB
- 🔄 **Auto-scaling**: Dynamic resource management
- 📝 **Full API Documentation**: Swagger/OpenAPI support

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Streamlit  │  │   Gradio     │  │   Web UI        │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                       API Gateway                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         FastAPI Backend (Port 8000)                  │  │
│  │  - REST API    - WebSocket    - Authentication      │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Agent   │  │  Model   │  │Document  │  │  Task    │  │
│  │ Manager  │  │ Manager  │  │Processor │  │ Scheduler│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     AI Agents Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ AutoGPT  │  │ CrewAI   │  │AgentGPT  │  │PrivateGPT│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │LlamaIndex│  │FlowiseAI │  │  Aider   │  │ TabbyML  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │PostgreSQL│  │  Redis   │  │  Qdrant  │  │ ChromaDB │  │
│  │   (5432) │  │  (6379)  │  │  (6333)  │  │  (8001)  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Backend**: Python 3.11+, FastAPI, SQLAlchemy
- **AI/ML**: Transformers, LangChain, Ollama
- **Databases**: PostgreSQL, Redis
- **Vector Stores**: Qdrant, ChromaDB
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana
- **Frontend**: Streamlit, Gradio

## Installation & Setup

### Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- Docker and Docker Compose installed
- Python 3.11 or higher
- At least 8GB RAM (16GB recommended)
- 50GB free disk space
- NVIDIA GPU (optional, for enhanced performance)

### Quick Start

```bash
# Clone or navigate to the repository
cd /opt/sutazaiapp

# Run the comprehensive launcher
sudo ./launch_sutazai_agi.sh

# Or use individual scripts:
# - Fix issues: sudo python3 fix_all_issues.py
# - Optimize: sudo ./optimize_and_deploy_agi.sh
# - Upgrade backend: sudo ./upgrade_backend.sh
```

### Manual Setup

1. **Install Dependencies**:
```bash
pip3 install -r requirements.txt
```

2. **Start Core Services**:
```bash
docker-compose up -d postgres redis ollama qdrant chromadb
```

3. **Initialize Database**:
```bash
python3 -c "from backend.models.base_models import Base; from sqlalchemy import create_engine; engine = create_engine('postgresql://sutazai:sutazai123@localhost:5432/sutazai_db'); Base.metadata.create_all(bind=engine)"
```

4. **Start Backend**:
```bash
python3 intelligent_backend_enterprise.py
```

5. **Deploy AI Agents**:
```bash
docker-compose up -d autogpt crewai agentgpt privategpt llamaindex flowise
```

## Core Components

### 1. Backend Service

The enhanced enterprise backend (`intelligent_backend_enterprise.py`) provides:

- **RESTful API**: Full CRUD operations
- **WebSocket Support**: Real-time bidirectional communication
- **Caching Layer**: Response caching for improved performance
- **Async Processing**: Non-blocking request handling
- **Multi-threading**: Parallel task execution

### 2. Database Layer

- **PostgreSQL**: Primary relational database
  - User management
  - Conversation history
  - Document metadata
  - Task tracking

- **Redis**: High-performance caching
  - Session management
  - Response caching
  - Rate limiting

### 3. Vector Stores

- **Qdrant**: Primary vector database
  - Document embeddings
  - Semantic search
  - Knowledge retrieval

- **ChromaDB**: Secondary vector store
  - Conversation embeddings
  - Context management

### 4. Model Management

- **Ollama Integration**: Local LLM hosting
  - Multiple model support
  - Automatic model loading
  - Resource management

## AI Agents

### Available Agents

| Agent | Port | Purpose | Status |
|-------|------|---------|--------|
| AutoGPT | 8080 | Autonomous task execution | 🟢 Active |
| CrewAI | 8102 | Multi-agent collaboration | 🟢 Active |
| AgentGPT | 8103 | Goal-oriented planning | 🟢 Active |
| PrivateGPT | 8104 | Private document Q&A | 🟢 Active |
| LlamaIndex | 8105 | Document indexing | 🟢 Active |
| FlowiseAI | 8106 | Visual workflow builder | 🟢 Active |
| TabbyML | 8081 | Code completion | 🟡 Optional |
| Aider | 8088 | Code editing assistant | 🟡 Optional |
| GPT-Engineer | 8087 | Project generation | 🟡 Optional |
| Semgrep | 8083 | Code security analysis | 🟡 Optional |

### Agent Capabilities

1. **Reasoning Agent**: Complex problem solving and logical analysis
2. **Code Generation Agent**: Automated code creation and optimization
3. **Document Analysis Agent**: Text extraction and summarization
4. **Research Agent**: Information gathering and synthesis
5. **Planning Agent**: Task breakdown and scheduling

## API Reference

### Authentication

All API endpoints require authentication via JWT tokens or API keys.

```bash
# Get auth token
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'
```

### Core Endpoints

#### Chat Completion
```bash
POST /api/chat
{
  "message": "Explain quantum computing",
  "model": "llama3.2:1b",
  "temperature": 0.7,
  "max_tokens": 500
}
```

#### Document Processing
```bash
POST /api/documents/upload
Content-Type: multipart/form-data

POST /api/documents/process
{
  "content": "document text",
  "operation": "summarize"
}
```

#### Agent Execution
```bash
POST /api/agents/execute
{
  "agent_type": "reasoning",
  "task": "Analyze market trends",
  "context": {}
}
```

#### System Status
```bash
GET /health
GET /api/status
GET /api/performance/summary
GET /api/models
GET /api/agents
```

### WebSocket API

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/client-123');

// Send message
ws.send(JSON.stringify({
  type: 'chat',
  data: {
    message: 'Hello AI',
    model: 'llama3.2:1b'
  }
}));

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=postgresql://sutazai:sutazai123@localhost:5432/sutazai_db
REDIS_URL=redis://localhost:6379

# AI Models
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.2:1b

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# Performance
MAX_WORKERS=8
REQUEST_TIMEOUT=120
CACHE_TTL=300

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs
```

### Docker Compose Configuration

Modify `docker-compose.yml` to adjust:
- Resource limits
- Port mappings
- Volume mounts
- Environment variables

## Troubleshooting

### Common Issues

1. **Backend not starting**
   ```bash
   # Check logs
   tail -f logs/backend/enterprise.log
   
   # Run fix script
   sudo python3 fix_all_issues.py
   ```

2. **Docker containers not running**
   ```bash
   # Check container status
   docker-compose ps
   
   # View container logs
   docker-compose logs [service-name]
   
   # Restart containers
   docker-compose restart
   ```

3. **Database connection errors**
   ```bash
   # Check PostgreSQL
   docker exec sutazai-postgres pg_isready -U sutazai
   
   # Reset database
   docker-compose down postgres
   docker-compose up -d postgres
   ```

4. **Model loading issues**
   ```bash
   # List available models
   docker exec sutazai-ollama ollama list
   
   # Pull missing models
   docker exec sutazai-ollama ollama pull llama3.2:1b
   ```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python3 intelligent_backend_enterprise.py
```

## Performance Optimization

### System Tuning

1. **Increase file descriptors**:
   ```bash
   echo "fs.file-max = 2097152" >> /etc/sysctl.conf
   sysctl -p
   ```

2. **Optimize Docker**:
   ```json
   {
     "storage-driver": "overlay2",
     "log-driver": "json-file",
     "log-opts": {
       "max-size": "10m",
       "max-file": "3"
     }
   }
   ```

3. **Database optimization**:
   ```sql
   ALTER SYSTEM SET shared_buffers = '256MB';
   ALTER SYSTEM SET effective_cache_size = '1GB';
   ```

### Monitoring

Access monitoring dashboards:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- API Metrics: http://localhost:8000/api/metrics

## Security

### Best Practices

1. **Change default passwords**
2. **Enable SSL/TLS for production**
3. **Configure firewall rules**
4. **Regular security updates**
5. **Audit logs monitoring**

### API Security

- JWT token authentication
- API key management
- Rate limiting
- CORS configuration
- Input validation

## Support & Resources

- **Logs**: Check `/opt/sutazaiapp/logs/`
- **Documentation**: `/opt/sutazaiapp/docs/`
- **API Docs**: http://localhost:8000/api/docs
- **System Status**: Run `./sutazai.sh status`

## License

SutazAI is released under the MIT License. See LICENSE file for details.

---

**Version**: 10.0 Enterprise  
**Last Updated**: $(date)  
**Maintained By**: SutazAI Team