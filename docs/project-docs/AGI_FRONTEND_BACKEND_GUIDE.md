# SutazAI AGI/ASI - Frontend & Backend Integration Guide

## Overview

The SutazAI AGI/ASI system includes enhanced frontend and backend components that integrate with all 30+ AI agents, providing a unified control center for autonomous AI operations.

## Enhanced Components

### 1. AGI-Enhanced Frontend (`frontend/app_agi_enhanced.py`)

**Features:**
- **Unified Dashboard**: Real-time overview of all AI agents and system status
- **Multi-Agent Orchestration**: Visual interface for coordinating multiple agents
- **Task Builder**: Drag-and-drop workflow designer for complex AGI tasks
- **Real-time Monitoring**: Live metrics, health checks, and performance graphs
- **Agent Management**: Individual control and configuration for each AI agent
- **Model Selection**: Easy switching between Ollama models

**Key Interfaces:**
1. **Dashboard** - System-wide metrics and agent status grid
2. **AI Agents** - Detailed management of all 30+ agents
3. **Orchestration** - Multi-agent task coordination
4. **Task Builder** - Visual workflow designer
5. **Monitoring** - Real-time system performance
6. **Settings** - Comprehensive configuration options

### 2. AGI-Enhanced Backend (`backend/main_agi_enhanced.py`)

**Features:**
- **Unified API**: Single endpoint for all AI agent interactions
- **Service Orchestration**: Coordinates multi-agent workflows
- **Model Management**: Handles Ollama and LiteLLM proxy integration
- **Real-time WebSocket**: Live updates and streaming responses
- **Task Queue**: Redis-based task management
- **Health Monitoring**: Comprehensive health checks for all services

**API Endpoints:**
- `GET /` - Service information
- `GET /health` - Comprehensive health check
- `GET /api/v1/agents` - List all AI agents
- `POST /api/v1/agents/execute` - Execute task on specific agent
- `POST /api/v1/orchestrate` - Multi-agent orchestration
- `GET /api/v1/models` - List available models
- `POST /api/v1/generate` - Generate completion
- `POST /api/v1/workflows/execute` - Execute complex workflows
- `GET /api/v1/metrics` - System metrics
- `WS /ws` - WebSocket for real-time updates

## Deployment

### Option 1: Use Enhanced Components Directly

```bash
# Backend
cd /opt/sutazaiapp
python backend/main_agi_enhanced.py

# Frontend
streamlit run frontend/app_agi_enhanced.py --server.port 8501
```

### Option 2: Deploy with Docker Compose Override

```bash
# Use the AGI-specific compose file
docker-compose -f docker-compose.yml -f docker-compose-agi.yml up -d backend-agi
```

### Option 3: Full System Deployment

The deployment script automatically detects and uses the enhanced components:

```bash
./deploy_complete_sutazai_system_improved.sh
```

## Configuration

### Backend Environment Variables

```bash
# Database
POSTGRES_HOST=postgres
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai123
POSTGRES_DB=sutazai

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Model Services
OLLAMA_URL=http://ollama:11434
LITELLM_URL=http://litellm:4000
SERVICE_HUB_URL=http://service-hub:8080

# API Configuration
OPENAI_API_BASE=http://litellm:4000/v1
OPENAI_API_KEY=sk-local
```

### Frontend Configuration

The frontend automatically connects to:
- Backend API: `http://localhost:8000`
- Service Hub: `http://localhost:8114`
- LiteLLM Proxy: `http://localhost:4000`

## Agent Integration

### Core Agents
- **AutoGPT** (8080): Autonomous task execution
- **CrewAI** (8096): Multi-agent collaboration
- **Aider** (8095): AI pair programming
- **GPT-Engineer** (8097): Full project generation
- **LlamaIndex** (8098): Document analysis

### Advanced Agents
- **LocalAGI** (8103): Local AGI implementation
- **AutoGen** (8104): Microsoft's multi-agent framework
- **AgentZero** (8105): Zero-shot task completion
- **BigAGI** (8106): Advanced conversational AI
- **Dify** (8107): Visual AI workflow builder

### Specialized Agents
- **OpenDevin** (8108): AI software engineer
- **FinRobot** (8109): Financial analysis
- **RealtimeSTT** (8110): Speech-to-text
- **Code Improver** (8113): Autonomous code enhancement

### Workflow Agents
- **LangFlow** (8090): Visual LLM flows
- **Flowise** (8099): Drag-drop LLM flows
- **n8n** (5678): Workflow automation
- **Service Hub** (8114): Central orchestration

## Usage Examples

### 1. Execute Single Agent Task

```python
# Via API
curl -X POST http://localhost:8000/api/v1/agents/execute \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "crewai",
    "task": "Analyze market trends for AI companies",
    "parameters": {"depth": "detailed"}
  }'
```

### 2. Multi-Agent Orchestration

```python
# Orchestrate multiple agents
curl -X POST http://localhost:8000/api/v1/orchestrate \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "analysis",
    "task_data": {
      "query": "Create a comprehensive business plan"
    },
    "agents": ["crewai", "finrobot", "autogen"]
  }'
```

### 3. Workflow Execution

```python
# Execute complex workflow
curl -X POST http://localhost:8000/api/v1/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "research-and-code",
    "steps": [
      {
        "type": "agent_task",
        "agent": "crewai",
        "task": "Research best practices for API design"
      },
      {
        "type": "agent_task",
        "agent": "gpt-engineer",
        "task": "Generate REST API based on research"
      },
      {
        "type": "agent_task",
        "agent": "code-improver",
        "task": "Optimize and secure the generated code"
      }
    ]
  }'
```

## Monitoring

### Health Check
```bash
# Check system health
curl http://localhost:8000/health

# Check specific service via hub
curl http://localhost:8114/health/autogpt
```

### Metrics
```bash
# Get system metrics
curl http://localhost:8000/api/v1/metrics

# Real-time monitoring
# Access Grafana: http://localhost:3000
# Access Prometheus: http://localhost:9090
```

## Troubleshooting

### Frontend Issues
1. **Cannot connect to backend**:
   - Check backend is running: `curl http://localhost:8000/health`
   - Verify CORS settings in backend

2. **Agents show as unavailable**:
   - Check Service Hub: `curl http://localhost:8114/services`
   - Verify agent containers: `docker ps | grep sutazai`

### Backend Issues
1. **Database connection failed**:
   - Check PostgreSQL: `docker logs sutazai-postgres`
   - Verify credentials in environment

2. **Service unavailable errors**:
   - Check service health: `curl http://localhost:8114/health`
   - Review container logs: `docker logs sutazai-<service>`

### Model Issues
1. **Ollama not responding**:
   - Check Ollama status: `curl http://localhost:11434/api/tags`
   - Pull required models: `docker exec sutazai-ollama ollama pull tinyllama`

2. **LiteLLM proxy errors**:
   - Check proxy health: `curl http://localhost:4000/health`
   - Verify config: `docker exec sutazai-litellm cat /app/config.yaml`

## Best Practices

1. **Resource Management**:
   - Monitor CPU/RAM usage via the monitoring dashboard
   - Adjust concurrent task limits based on system capacity

2. **Agent Selection**:
   - Use specialized agents for domain-specific tasks
   - Combine complementary agents for complex workflows

3. **Error Handling**:
   - Always check task results for errors
   - Implement retry logic for critical operations

4. **Security**:
   - Keep API keys secure (even though using local models)
   - Regularly update agent containers

## Next Steps

1. Access the AGI Control Center: http://localhost:8501
2. Explore the agent catalog in the dashboard
3. Try the orchestration interface for multi-agent tasks
4. Build custom workflows with the Task Builder
5. Monitor system performance in real-time

The enhanced frontend and backend provide a complete AGI/ASI control system with seamless integration of all AI agents, unified model management, and powerful orchestration capabilities.