# SutazAI Universal Agent System

A comprehensive, autonomous AI agent infrastructure that operates completely independently using local Ollama models and Redis messaging. This system provides advanced multi-agent coordination, workflow orchestration, and autonomous task execution capabilities.

## 🌟 Key Features

### Complete Independence
- **No External APIs**: Operates entirely with local Ollama models
- **Self-Contained**: All dependencies are local (Redis, Ollama)
- **Autonomous Operation**: Agents coordinate and execute tasks independently
- **Scalable Architecture**: Designed to handle complex multi-agent workflows

### Advanced Agent Capabilities
- **Dynamic Agent Creation**: Factory pattern for creating specialized agents
- **Inter-Agent Communication**: Redis-based message bus with advanced routing
- **Workflow Orchestration**: Complex multi-step workflow coordination
- **Agent Registry**: Centralized discovery and health monitoring
- **Autonomous Execution**: Agents can pursue goals independently

### Specialized Agent Types
- **Code Generator**: Advanced code generation using CodeLlama
- **Orchestrator**: Coordinates complex multi-agent workflows
- **Generic Agent**: Universal fallback for any task type
- **Security Analyzer**: (Extensible framework for adding more)

## 🏗️ Architecture

### Core Components

#### 1. Base Agent (`base_agent.py`)
- Foundation class for all agents
- Redis messaging integration
- Ollama model communication
- Health monitoring and heartbeats
- Task execution framework

#### 2. Universal Agent Factory (`universal_agent_factory.py`)
- Dynamic agent creation and management
- Template-based agent configuration
- Capability-based agent selection
- Load balancing and resource allocation

#### 3. Agent Message Bus (`agent_message_bus.py`)
- Advanced pub/sub messaging
- Message routing strategies (direct, broadcast, multicast, load-balanced)
- Message persistence and replay
- Priority handling and filtering

#### 4. Orchestration Controller (`orchestration_controller.py`)
- Multi-agent workflow coordination
- Task decomposition and distribution
- Dependency management
- Error handling and recovery

#### 5. Agent Registry (`agent_registry.py`)
- Centralized agent discovery
- Health monitoring and status tracking
- Performance metrics and analytics
- Capability-based selection

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- At least 8GB RAM (16GB recommended)
- Python 3.11+ (for local development)

### 1. Start the System
```bash
# Using Docker Compose (Recommended)
docker-compose -f docker-compose-new-universal-agents.yml up -d

# Or run locally
python scripts/start_universal_agent_system.py
```

### 2. Verify System Status
```bash
# Check health
curl http://localhost:9101/health

# View system metrics
curl http://localhost:9100/metrics

# Check agent registry
docker exec -it sutazai-universal-agent-redis redis-cli HGETALL sutazai:agent_registry
```

### 3. Create Your First Agent
```python
from backend.ai_agents.core import create_agent

# Create a code generator agent
agent = await create_agent(
    agent_id="my-code-generator",
    agent_type="code_generator",
    config_overrides={
        "model_config": {
            "model": "codellama",
            "temperature": 0.2
        }
    }
)
```

### 4. Execute a Workflow
```python
from backend.ai_agents.core import get_orchestration_controller

controller = get_orchestration_controller()

# Create workflow
workflow_spec = {
    "name": "Build Python API",
    "description": "Create a complete Python API with tests",
    "tasks": [
        {
            "id": "analyze-requirements",
            "name": "Analyze Requirements",
            "task_type": "analyze",
            "required_capabilities": ["reasoning"],
            "input_data": {
                "requirements": "Build a REST API for user management"
            }
        },
        {
            "id": "generate-code",
            "name": "Generate API Code",
            "task_type": "generate_code",
            "required_capabilities": ["code_generation"],
            "dependencies": ["analyze-requirements"],
            "input_data": {
                "specification": "FastAPI with CRUD operations",
                "language": "python"
            }
        }
    ]
}

workflow_id = await controller.create_workflow(workflow_spec)
await controller.start_workflow(workflow_id)
```

## 📖 Detailed Usage

### Agent Types

#### Code Generator Agent
Specialized in code generation tasks:
```python
# Generate code from specification
result = await agent.execute_task("task-1", {
    "task_type": "generate_code",
    "specification": "Create a Python function to calculate fibonacci numbers",
    "language": "python",
    "code_type": "function"
})
```

#### Orchestrator Agent
Coordinates complex workflows:
```python
# Create and execute workflow
result = await orchestrator.execute_task("workflow-1", {
    "task_type": "create_workflow",
    "request": "Build a complete web application with backend and frontend",
    "workflow_type": "code_development"
})
```

#### Generic Agent
Handles any type of task:
```python
# General task execution
result = await generic_agent.execute_task("task-1", {
    "task_type": "analyze",
    "content": "Analyze this business process and suggest improvements",
    "analysis_type": "process_optimization"
})
```

### Advanced Features

#### Custom Agent Creation
```python
# Create agent with specific capabilities
agent = await create_agent_by_capabilities(
    agent_id="custom-agent",
    required_capabilities=["code_generation", "security_analysis"],
    preferred_type="code_generator"
)
```

#### Message Bus Communication
```python
from backend.ai_agents.core import send_message, broadcast_message

# Send direct message
await send_message(
    sender_id="agent-1",
    receiver_id="agent-2",
    message_type="code_review_request",
    content={"code": "def hello(): return 'world'"}
)

# Broadcast to all agents
await broadcast_message(
    sender_id="orchestrator",
    message_type="system_shutdown",
    content={"reason": "maintenance"}
)
```

#### Agent Registry Queries
```python
from backend.ai_agents.core import get_agent_registry

registry = get_agent_registry()

# Find agents by capability
code_agents = registry.find_agents_by_capability([AgentCapability.CODE_GENERATION])

# Get system health
health = await registry.health_check()
```

## ⚙️ Configuration

### System Configuration (`config/universal_agents.json`)
```json
{
  "redis": {
    "url": "redis://localhost:6379",
    "namespace": "sutazai"
  },
  "ollama": {
    "url": "http://localhost:11434",
    "default_model": "codellama"
  },
  "initial_agents": [
    {
      "id": "orchestrator-001",
      "type": "orchestrator",
      "name": "Master Orchestrator",
      "config": {
        "model": "llama2",
        "max_concurrent_tasks": 10
      }
    }
  ]
}
```

### Agent-Specific Configuration
Each agent type can be configured with:
- **Model settings**: Choose specific Ollama models
- **Capability sets**: Define what the agent can do
- **Resource limits**: Control memory and task concurrency
- **Behavior parameters**: Temperature, max tokens, etc.

## 🔧 Development

### Adding New Agent Types

1. **Create Agent Class**:
```python
# backend/ai_agents/specialized/my_agent.py
from ..core.base_agent import BaseAgent

class MySpecializedAgent(BaseAgent):
    async def on_initialize(self):
        # Custom initialization
        pass
    
    async def on_task_execute(self, task_id: str, task_data: dict):
        # Custom task execution
        return {"success": True, "result": "task completed"}
```

2. **Register Agent Template**:
```python
# In the factory initialization
self.register_template(AgentTemplate(
    agent_type="my_specialized_agent",
    class_path="backend.ai_agents.specialized.my_agent.MySpecializedAgent",
    default_config={
        "name": "My Specialized Agent",
        "capabilities": ["my_capability"],
        "model_config": {"model": "llama2"}
    }
))
```

### Custom Workflow Types
```python
# Define custom workflow template
workflow_template = {
    "name": "My Custom Workflow",
    "phases": [
        {
            "name": "preparation",
            "tasks": [
                {
                    "name": "setup_environment",
                    "agent_type": "generic",
                    "capabilities": ["reasoning"]
                }
            ]
        }
    ]
}
```

## 📊 Monitoring

### System Metrics
- Agent health and status
- Task execution statistics
- Message bus throughput
- Resource utilization
- Error rates and patterns

### Health Checks
```bash
# System health
curl http://localhost:9101/health

# Agent status
curl http://localhost:9100/agents/status

# Workflow progress
curl http://localhost:9100/workflows/active
```

### Logging
Comprehensive logging across all components:
- Agent lifecycle events
- Task execution details
- Message bus activity
- Error tracking and debugging

## 🛡️ Security

### Local Operation
- No external API calls
- All models run locally via Ollama
- Data never leaves your infrastructure

### Message Security
- Redis-based secure messaging
- Message expiration and cleanup
- Agent authentication via registry

### Resource Protection
- Task execution limits
- Memory usage monitoring
- CPU usage controls
- Automatic cleanup of failed tasks

## 🔍 Troubleshooting

### Common Issues

#### Agents Not Starting
```bash
# Check Redis connection
docker exec -it sutazai-universal-agent-redis redis-cli ping

# Check Ollama models
docker exec -it sutazai-universal-agent-ollama ollama list

# View agent logs
docker logs sutazai-universal-agent-system
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check agent workloads
curl http://localhost:9100/agents/workloads

# View message queue status
curl http://localhost:9100/message-bus/stats
```

#### Task Execution Failures
```bash
# Check task status
curl http://localhost:9100/workflows/{workflow_id}/status

# View error logs
tail -f logs/universal_agents.log | grep ERROR

# Check agent capabilities
curl http://localhost:9100/agents/{agent_id}/capabilities
```

## 🤝 Integration

### Existing SutazAI Services
The Universal Agent System integrates seamlessly with:
- AutoGPT workflows
- CrewAI multi-agent systems
- TabbyML code completion
- LiteLLM model management

### API Endpoints
```python
# REST API integration
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8910/agents/create",
        json={
            "agent_id": "api-agent",
            "agent_type": "generic",
            "config": {"model": "llama2"}
        }
    )
```

## 📈 Scaling

### Horizontal Scaling
- Run multiple agent instances
- Distribute across multiple machines
- Load balance through Redis messaging

### Vertical Scaling
- Increase agent concurrent task limits
- Allocate more resources to Ollama
- Optimize Redis configuration for throughput

## 🔄 Updates and Maintenance

### Model Updates
```bash
# Update Ollama models
docker exec -it sutazai-universal-agent-ollama ollama pull codellama:latest

# Restart agents to use new models
docker restart sutazai-universal-agent-system
```

### System Updates
```bash
# Update the system
git pull origin main
docker-compose -f docker-compose-new-universal-agents.yml build
docker-compose -f docker-compose-new-universal-agents.yml up -d
```

## 📝 License

This Universal Agent System is part of the SutazAI project and follows the same licensing terms.

## 🆘 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review logs in the `logs/` directory
3. Submit issues with detailed error information
4. Include system configuration and agent setup details

---

**Built with ❤️ for autonomous AI agent orchestration**