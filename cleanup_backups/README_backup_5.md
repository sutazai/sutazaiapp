# AI Agent Interface System

A comprehensive interface system for managing and coordinating all 38 AI agents in the SutazAI AGI/ASI system. This system provides universal client libraries, agent discovery, workflow orchestration, real-time communication, health monitoring, and complete API wrappers.

## 🚀 Quick Start

```python
import asyncio
from universal_client import UniversalAgentClient, AgentType, Priority
from api_wrappers import UnifiedAgentAPI

async def main():
    # Initialize the system
    async with UniversalAgentClient() as client:
        api = UnifiedAgentAPI(client)
        
        # Use any agent directly
        result = await api.code_improver.analyze_code_quality(
            code="def hello(): print('Hello, World!')",
            language="python"
        )
        
        print(f"Analysis result: {result.data}")

asyncio.run(main())
```

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
- [Agent Types](#agent-types)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Contributing](#contributing)

## 🎯 System Overview

The AI Agent Interface System provides a unified way to interact with all 38 specialized AI agents in the SutazAI ecosystem. It includes:

- **Universal Client Library**: Standardized interface for all agent interactions
- **Agent Discovery**: Intelligent matching of tasks to suitable agents
- **Workflow Orchestration**: Complex multi-agent workflow management
- **Real-time Communication**: WebSocket-based agent coordination
- **Health Monitoring**: Comprehensive agent health and performance tracking
- **API Wrappers**: Typed interfaces for all 38 agents
- **Integration Examples**: Complete usage patterns and scenarios

## 🏗️ Core Components

### 1. Universal Client (`universal_client.py`)

The foundation client library providing standardized access to all agents.

```python
from universal_client import UniversalAgentClient, AgentType

async with UniversalAgentClient() as client:
    response = await client.execute_task(
        agent_type=AgentType.CODE_GENERATION_IMPROVER,
        task_description="Analyze code quality",
        parameters={"code": "...", "language": "python"}
    )
```

**Key Features:**
- Async/await support
- Automatic retry logic
- Connection pooling
- Health checking
- Response validation

### 2. Agent Discovery (`discovery_service.py`)

Intelligent agent discovery and capability matching system.

```python
from discovery_service import DiscoveryService

discovery = DiscoveryService(client)
await discovery.start()

# Find best agent for a task
best_match = discovery.find_best_agent(
    task_description="Security testing and vulnerability assessment",
    capabilities=["security_testing", "vulnerability_scanning"],
    priority=Priority.HIGH
)
```

**Key Features:**
- Capability-based matching
- Load balancing
- Performance-based selection
- Agent health awareness
- Fuzzy matching algorithms

### 3. Workflow Orchestration (`workflow_orchestrator.py`)

Advanced multi-agent workflow management with dependency handling.

```python
from workflow_orchestrator import WorkflowEngine, WorkflowBuilder

# Create workflow
workflow = (
    WorkflowBuilder("code_analysis", "Comprehensive Code Analysis")
    .add_task("fetch_code", "Fetch Repository", agent_type="git-specialist")
    .add_task("analyze_quality", "Code Quality Analysis", 
              agent_type="code-generation-improver", 
              dependencies=["fetch_code"])
    .add_task("security_scan", "Security Analysis",
              agent_type="semgrep-security-analyzer",
              dependencies=["fetch_code"])
    .build()
)

# Execute workflow
engine = WorkflowEngine(client, discovery)
engine.register_workflow(workflow)
execution_id = await engine.start_workflow("code_analysis")
```

**Key Features:**
- Dependency management
- Parallel execution
- Error handling & retry
- Progress monitoring
- Conditional execution

### 4. Real-time Communication (`communication_protocols.py`)

WebSocket-based real-time agent communication and coordination.

```python
from communication_protocols import CommunicationHub, AgentCommunicator

hub = CommunicationHub()
await hub.start()

communicator = AgentCommunicator("my-agent", hub)

# Send task to another agent
await communicator.send_task_assignment(
    target_agent="security-specialist",
    task_description="Perform security audit",
    parameters={"target": "web-app"}
)
```

**Key Features:**
- WebSocket connections
- Message queuing
- Heartbeat monitoring
- Event-driven architecture
- Pub/sub messaging

### 5. Health Monitoring (`health_monitor.py`)

Comprehensive health monitoring and automated recovery system.

```python
from health_monitor import HealthMonitor

monitor = HealthMonitor(client, discovery)
await monitor.start()

# Get system health overview
overview = monitor.get_system_health_overview()
print(f"System health: {overview['system_health_percentage']:.1f}%")

# Get individual agent health
summary = monitor.get_agent_health_summary("code-generation-improver")
```

**Key Features:**
- Real-time health checking
- Performance metrics
- Alert management
- Automated recovery
- Historical tracking

### 6. API Wrappers (`api_wrappers.py`)

Specialized API wrappers for all 38 agents with typed interfaces.

```python
from api_wrappers import UnifiedAgentAPI

api = UnifiedAgentAPI(client)

# Use specialized wrappers
result = await api.agi_architect.design_system_architecture(
    requirements={"type": "microservices", "scale": "high"},
    constraints={"budget": "moderate"}
)

# Or access any agent generically
wrapper = api.get_agent("custom-agent-id")
result = await wrapper.execute("custom task")
```

## 🤖 Agent Types

The system supports all 38 specialized AI agents:

### Core System Agents
- **AGI System Architect**: System design and architecture
- **Autonomous System Controller**: Autonomous decision making
- **AI Agent Orchestrator**: Multi-agent coordination

### Infrastructure & DevOps
- **Infrastructure DevOps Manager**: Container and CI/CD management
- **Deployment Automation Master**: System deployment
- **Hardware Resource Optimizer**: Resource optimization

### AI & ML Specialists
- **Ollama Integration Specialist**: Local LLM management
- **LiteLLM Proxy Manager**: API proxy management
- **Senior AI Engineer**: ML architecture and RAG systems
- **Deep Learning Brain Manager**: Neural network evolution

### Development Specialists
- **Code Generation Improver**: Code quality analysis
- **OpenDevin Code Generator**: Autonomous code generation
- **Senior Frontend Developer**: UI/UX development
- **Senior Backend Developer**: API and database development

### Quality & Testing
- **Testing QA Validator**: Comprehensive testing
- **Security Pentesting Specialist**: Security assessment
- **Semgrep Security Analyzer**: Static code analysis

### And 21 more specialized agents...

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd sutazaiapp/backend/ai_agents

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

### Dependencies

```
aiohttp>=3.8.0
aioredis>=2.0.0
asyncio
networkx>=2.8.0
psutil>=5.9.0
dataclasses-json>=0.5.7
```

## 🔧 Configuration

### Environment Variables

```bash
# Agent endpoints (optional - will use defaults)
export SUTAZAI_BASE_URL="http://localhost"

# Redis for distributed messaging (optional)
export REDIS_URL="redis://localhost:6379"

# Health monitoring settings
export HEALTH_CHECK_INTERVAL=30
export MAX_RETRIES=3
```

### Configuration File

```json
{
  "agents": [
    {
      "id": "agi-system-architect",
      "name": "AGI System Architect",
      "endpoint": "http://localhost:8001",
      "capabilities": ["system_design", "architecture_optimization"],
      "priority": "critical"
    }
  ],
  "workflow_settings": {
    "max_parallel_tasks": 5,
    "default_timeout": 300
  },
  "health_monitoring": {
    "check_interval": 30,
    "failure_threshold": 3
  }
}
```

## 💡 Usage Examples

### Basic Agent Interaction

```python
async def basic_example():
    async with UniversalAgentClient() as client:
        # Execute task on specific agent
        response = await client.execute_task(
            agent_type=AgentType.CODE_GENERATION_IMPROVER,
            task_description="Analyze code quality",
            parameters={"code": "def hello(): print('world')", "language": "python"}
        )
        print(f"Result: {response.result}")
```

### Agent Discovery

```python
async def discovery_example():
    async with UniversalAgentClient() as client:
        discovery = DiscoveryService(client)
        await discovery.start()
        
        # Find agents by capability
        security_agents = discovery.registry.get_agents_by_capability("security_testing")
        
        # Get recommendations
        recommendations = discovery.get_agent_recommendations(
            "I need help with database optimization"
        )
        
        await discovery.stop()
```

### Workflow Orchestration

```python
async def workflow_example():
    async with UniversalAgentClient() as client:
        discovery = DiscoveryService(client)
        await discovery.start()
        
        engine = WorkflowEngine(client, discovery)
        
        # Create and execute workflow
        workflow = create_development_workflow()
        engine.register_workflow(workflow)
        
        execution_id = await engine.start_workflow(
            "development_pipeline",
            parameters={"project": "web-app"}
        )
        
        # Monitor execution
        while True:
            status = engine.get_execution_status(execution_id)
            if status['status'] in ['completed', 'failed']:
                break
            await asyncio.sleep(5)
        
        await discovery.stop()
```

### Health Monitoring

```python
async def monitoring_example():
    async with UniversalAgentClient() as client:
        discovery = DiscoveryService(client)
        await discovery.start()
        
        monitor = HealthMonitor(client, discovery)
        await monitor.start()
        
        # Get system health
        overview = monitor.get_system_health_overview()
        print(f"System Health: {overview['system_health_percentage']:.1f}%")
        
        # Get individual agent health
        for agent_id in ['agi-system-architect', 'code-generation-improver']:
            summary = monitor.get_agent_health_summary(agent_id)
            if summary:
                print(f"{agent_id}: {summary['current_status']}")
        
        await monitor.stop()
        await discovery.stop()
```

### Complete Integration

```python
async def complete_example():
    """Complete system integration example"""
    examples = IntegrationExamples()
    await examples.initialize_system()
    
    try:
        # Use unified API
        result = await examples.api.agi_architect.design_system_architecture(
            requirements={"type": "microservices", "scalability": "high"},
            constraints={"timeline": "3_months"}
        )
        
        if result.success:
            print("Architecture designed successfully!")
            
            # Chain with code generation
            code_result = await examples.api.backend_dev.design_api(
                api_spec=result.data.get('api_requirements'),
                framework="fastapi"
            )
            
            if code_result.success:
                print("API code generated!")
    
    finally:
        await examples.cleanup_system()
```

## 📚 API Reference

### UniversalAgentClient

#### Methods

- `execute_task(agent_type, task_description, parameters, priority, timeout)`: Execute task on agent
- `execute_parallel_tasks(tasks)`: Execute multiple tasks in parallel
- `health_check_all()`: Check health of all agents
- `list_agents()`: Get list of available agents
- `find_agents_by_capability(capability)`: Find agents with specific capability

### DiscoveryService

#### Methods

- `find_best_agent(task_description, capabilities, priority)`: Find optimal agent
- `find_agents_by_requirements(requirements)`: Find agents matching requirements
- `get_agent_recommendations(task_description)`: Get agent recommendations
- `get_registry_stats()`: Get discovery statistics

### WorkflowEngine

#### Methods

- `register_workflow(workflow_definition)`: Register workflow
- `start_workflow(workflow_id, parameters)`: Start workflow execution
- `get_execution_status(execution_id)`: Get workflow status
- `cancel_workflow(execution_id)`: Cancel running workflow
- `get_metrics()`: Get engine metrics

### HealthMonitor

#### Methods

- `get_system_health_overview()`: Get system-wide health metrics
- `get_agent_health_summary(agent_id)`: Get individual agent health
- `start()`: Start health monitoring
- `stop()`: Stop health monitoring

## 🏛️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Integration Examples  │  API Wrappers  │  Unified API     │
├─────────────────────────────────────────────────────────────┤
│     Orchestration Layer                                     │
├──────────────────┬──────────────────┬───────────────────────┤
│ Workflow Engine  │ Discovery Service│  Communication Hub   │
├─────────────────────────────────────────────────────────────┤
│            Core Infrastructure Layer                        │
├──────────────────┬──────────────────┬───────────────────────┤
│ Universal Client │  Health Monitor  │  Message Protocols   │
├─────────────────────────────────────────────────────────────┤
│                   Agent Layer (38 Agents)                  │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions

1. **Application Layer**: High-level interfaces and examples
2. **Orchestration Layer**: Workflow management and agent coordination
3. **Infrastructure Layer**: Core services and communication
4. **Agent Layer**: Individual specialized AI agents

### Data Flow

1. Request → Universal Client → Agent Discovery
2. Discovery → Load Balancer → Selected Agent
3. Agent → Task Execution → Response
4. Response → Health Monitor → Metrics Update
5. Parallel: Communication Hub → Real-time Updates

## 🔍 Monitoring and Observability

### Health Metrics

- **Agent Status**: Online/Offline/Busy/Error
- **Response Times**: Average, P95, P99
- **Success Rates**: Task completion rates
- **Resource Usage**: CPU, Memory, Disk
- **Error Rates**: Failure frequencies

### Alerting

- **Critical**: Agent failures, system outages
- **Warning**: High response times, resource issues
- **Info**: Deployment notifications, status changes

### Dashboard Access

The system provides comprehensive monitoring dashboards:

```python
# Get system overview
overview = monitor.get_system_health_overview()

# Get detailed metrics
metrics = monitor.get_health_metrics()

# Get active alerts
alerts = monitor.alert_manager.get_active_alerts()
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/performance/

# Run with coverage
python -m pytest --cov=ai_agents tests/
```

### Integration Testing

```python
# Run integration examples
python integration_examples.py

# Run specific example
python -c "
import asyncio
from integration_examples import example_basic_agent_interaction
asyncio.run(example_basic_agent_interaction())
"
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["python", "-m", "ai_agents.server"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agent-system
  template:
    spec:
      containers:
      - name: agent-system
        image: sutazai/agent-system:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

## 🔧 Troubleshooting

### Common Issues

#### Agent Connection Failures

```python
# Check agent health
health_results = await client.health_check_all()
offline_agents = [
    agent_id for agent_id, is_healthy in health_results.items() 
    if not is_healthy
]
print(f"Offline agents: {offline_agents}")
```

#### Performance Issues

```python
# Monitor response times
metrics = monitor.get_health_metrics()
slow_agents = [
    agent_id for agent_id in metrics 
    if metrics[agent_id]['avg_response_time'] > 5000
]
print(f"Slow agents: {slow_agents}")
```

#### Workflow Failures

```python
# Check workflow status
status = engine.get_execution_status(execution_id)
failed_tasks = [
    task_id for task_id, task_info in status['tasks'].items()
    if task_info['status'] == 'failed'
]
print(f"Failed tasks: {failed_tasks}")
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug client
client = UniversalAgentClient(debug=True)
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd sutazaiapp/backend/ai_agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Run tests
pytest

# Run linting
flake8 .
black .
```

### Adding New Agents

1. Add agent type to `AgentType` enum in `universal_client.py`
2. Create wrapper class in `api_wrappers.py`
3. Add configuration to default agent list
4. Update documentation and examples
5. Write tests for new agent functionality

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public methods
- Include unit tests for new functionality
- Update integration examples as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the troubleshooting section
- Review the integration examples
- Consult the API reference

## 🚀 Roadmap

### Upcoming Features

- [ ] GraphQL API interface
- [ ] Advanced workflow templates
- [ ] Machine learning-based agent selection
- [ ] Enhanced security features
- [ ] Performance optimizations
- [ ] Additional monitoring integrations
- [ ] Cloud-native deployments

---

Built with ❤️ for the SutazAI AGI/ASI Autonomous System