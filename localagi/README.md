# LocalAGI Autonomous Orchestration System

## Overview

The LocalAGI Autonomous Orchestration System is a comprehensive framework that enables fully autonomous coordination of 38 AI agents without external dependencies. It provides self-organizing agent swarms, autonomous decision-making, recursive task decomposition, and self-improving workflows.

## Architecture

### Core Components

1. **Autonomous Orchestration Engine** (`engine/autonomous_orchestration_engine.py`)
   - Manages all 38 AI agents
   - Handles task assignment and execution
   - Implements self-improvement mechanisms
   - Provides real-time performance monitoring

2. **Decision Engine** (`engine/decision_engine.py`)
   - Multi-criteria decision making algorithms
   - Adaptive decision strategies (genetic, reinforcement learning, Monte Carlo, Bayesian)
   - Performance-based strategy adaptation

3. **Task Decomposer** (`engine/task_decomposer.py`)
   - Recursive task breakdown into manageable subtasks
   - Dependency management and execution planning
   - Dynamic complexity assessment

4. **Swarm Coordinator** (`swarms/swarm_coordinator.py`)
   - Self-organizing agent swarms
   - Leader election and consensus mechanisms
   - Dynamic swarm formation and dissolution

5. **Workflow Engine** (`workflows/self_improving_workflow_engine.py`)
   - Self-adapting workflows based on performance feedback
   - Cross-workflow learning and optimization
   - Performance metric tracking

6. **Collaborative Problem Solver** (`frameworks/collaborative_problem_solver.py`)
   - Multi-agent collaborative problem solving
   - Multiple collaboration patterns (divide-and-conquer, brainstorming, expert panels)
   - Consensus building and solution synthesis

7. **Goal Achievement System** (`goals/autonomous_goal_achievement_system.py`)
   - Independent goal pursuit and achievement
   - Adaptive planning and replanning
   - Goal hierarchy management

8. **Coordination Protocols** (`protocols/autonomous_coordination_protocols.py`)
   - Agent-to-agent communication protocols
   - Consensus algorithms and leader election
   - Failure recovery and system resilience

## Key Features

### Autonomous Capabilities

- **Self-Organizing Agent Swarms**: Agents automatically form teams based on task requirements
- **Autonomous Decision Making**: Multi-criteria decision algorithms adapt to performance
- **Recursive Task Decomposition**: Complex tasks are broken down automatically
- **Self-Improving Workflows**: Workflows learn and adapt from execution history
- **Independent Goal Achievement**: System pursues complex goals without human intervention

### Integration Features

- **Local Model Support**: Uses Ollama for all AI inference, no external API dependencies
- **Redis Integration**: State persistence and inter-agent communication
- **38 Agent Coordination**: Manages the complete SutazAI agent ecosystem
- **Real-time Monitoring**: Comprehensive system health and performance tracking

## Installation and Setup

### Prerequisites

- Python 3.8+
- Docker and docker-compose
- Ollama service running
- Redis instance available
- PostgreSQL database

### Quick Start

1. **Deploy the System**:
   ```bash
   cd /opt/sutazaiapp
   ./scripts/deploy_localagi_system.sh
   ```

2. **Manual Installation**:
   ```bash
   # Install dependencies
   pip install -r localagi/requirements.txt
   
   # Start infrastructure services
   docker-compose up -d postgres redis ollama chromadb
   
   # Initialize LocalAGI
   python3 localagi/main.py
   ```

3. **Configuration**:
   - Main config: `localagi/configs/autonomous_orchestrator_config.yaml`
   - Environment variables in `.env` file

## Usage

### Python API

```python
from localagi.main import get_localagi_system

# Initialize the system
system = await get_localagi_system()

# Submit autonomous tasks
task_id = await system.submit_autonomous_task(
    description="Optimize system performance",
    requirements=["analysis", "optimization"],
    priority=0.8,
    autonomous_mode=True
)

# Create self-improving workflows
workflow_steps = [
    {
        'name': 'Data Collection',
        'description': 'Collect system metrics',
        'agent_capability': 'monitoring',
        'success_criteria': ['Data collected successfully']
    },
    {
        'name': 'Analysis',
        'description': 'Analyze collected data',
        'agent_capability': 'analysis',
        'preconditions': ['Data Collection']
    }
]

workflow_id = await system.create_autonomous_workflow(
    description="System Optimization Workflow",
    steps=workflow_steps,
    optimization_objectives=['efficiency', 'quality']
)

# Execute workflow
result = await system.execute_workflow(workflow_id)

# Solve problems collaboratively
problem_result = await system.solve_problem_collaboratively(
    problem_description="How to improve AI agent coordination?",
    problem_type="optimization",
    max_agents=5
)

# Form agent swarms
swarm_result = await system.form_agent_swarm(
    goal="Coordinate optimization tasks",
    required_capabilities=["analysis", "optimization"],
    max_size=8
)

# Get comprehensive status
status = await system.get_comprehensive_status()
```

### REST API

The system provides a REST API through the backend service:

- **System Status**: `GET /api/v1/localagi/status`
- **Submit Task**: `POST /api/v1/localagi/tasks`
- **Create Workflow**: `POST /api/v1/localagi/workflows`
- **Problem Solving**: `POST /api/v1/localagi/problems/solve`

## Configuration

### Main Configuration (`configs/autonomous_orchestrator_config.yaml`)

```yaml
# System Configuration
system:
  name: "SutazAI LocalAGI Orchestration"
  version: "1.0.0"
  max_agents: 38
  enable_autonomous_mode: true

# Ollama Configuration
ollama:
  base_url: "http://ollama:11434"
  models:
    reasoning: "deepseek-r1:8b"
    general: "qwen2.5:3b"
    embedding: "nomic-embed-text"

# Decision Engine Settings
decision_engine:
  algorithms:
    - genetic
    - reinforcement_learning
    - monte_carlo
    - bayesian_optimization
  adaptation_threshold: 0.1
  exploration_rate: 0.15

# Performance Settings
performance:
  max_concurrent_tasks: 50
  task_timeout: 300
  health_check_interval: 60
  optimization_interval: 300
```

### Environment Variables

```bash
# Core Settings
LOCALAGI_LOG_LEVEL=INFO
LOCALAGI_MAX_AGENTS=38
LOCALAGI_ENABLE_AUTONOMOUS=true

# Service URLs
OLLAMA_HOST=http://ollama:11434
REDIS_URL=redis://redis:6379/1
DATABASE_URL=postgresql://user:pass@postgres:5432/sutazai

# Performance Settings
LOCALAGI_MAX_CONCURRENT_TASKS=10
LOCALAGI_TASK_TIMEOUT=300
LOCALAGI_MEMORY_LIMIT=4G
```

## Testing

### Integration Tests

```bash
# Run comprehensive integration tests
python3 tests/integration/test_localagi_integration.py

# Run system validation
python3 scripts/test_localagi_system.py
```

### Manual Testing

```bash
# Test basic connectivity
curl http://localhost:8115/health

# Submit test task
curl -X POST http://localhost:8115/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"description": "Test task", "requirements": ["testing"]}'

# Check system status
curl http://localhost:8115/api/status
```

## Monitoring and Observability

### System Metrics

- **Agent Performance**: Individual agent success rates and response times
- **Task Metrics**: Task completion rates, queue lengths, processing times
- **Workflow Performance**: Workflow success rates, optimization improvements
- **Resource Usage**: Memory, CPU, and network utilization
- **System Health**: Component availability, error rates, recovery times

### Logging

- **Main Log**: `/opt/sutazaiapp/logs/localagi.log`
- **Component Logs**: Individual logs for each major component
- **Performance Logs**: Detailed performance metrics and optimization data

### Health Checks

The system provides comprehensive health checking:

```python
# Get detailed system status
status = await system.get_comprehensive_status()

# Key metrics included:
# - system_status: operational/degraded/failed
# - uptime: system uptime in seconds
# - active_agents: number of operational agents
# - task_metrics: task processing statistics
# - performance_metrics: system performance data
```

## Autonomous Features in Detail

### Self-Organizing Agent Swarms

- **Dynamic Formation**: Swarms form automatically based on task requirements
- **Leader Election**: Consensus-based leader selection for coordination
- **Load Balancing**: Automatic work distribution across swarm members
- **Failure Recovery**: Automatic swarm reconfiguration on member failures

### Autonomous Decision Making

- **Multi-Algorithm Support**: Genetic algorithms, reinforcement learning, Monte Carlo, Bayesian optimization
- **Performance Adaptation**: Decision strategies adapt based on historical performance
- **Context Awareness**: Decisions consider current system state and resource availability
- **Learning Integration**: Decision patterns improve over time through machine learning

### Self-Improving Workflows

- **Performance Tracking**: Detailed metrics for each workflow execution
- **Parameter Adaptation**: Automatic adjustment of workflow parameters
- **Pattern Learning**: Recognition of successful execution patterns
- **Cross-Workflow Learning**: Insights shared between different workflows

### Collaborative Problem Solving

- **Multiple Patterns**: Divide-and-conquer, brainstorming, expert panels, competitive solutions
- **Consensus Building**: Structured approach to reaching agreement between agents
- **Solution Synthesis**: Combination of partial solutions into comprehensive answers
- **Quality Assurance**: Peer review and validation of proposed solutions

## Troubleshooting

### Common Issues

1. **Agents Not Loading**:
   - Check agent registry configuration
   - Verify agent service availability
   - Review agent definition files

2. **Ollama Connection Issues**:
   - Verify Ollama service is running
   - Check network connectivity
   - Confirm model availability

3. **Redis Connection Problems**:
   - Ensure Redis service is accessible
   - Verify authentication credentials
   - Check network configuration

4. **Performance Issues**:
   - Monitor resource usage
   - Adjust concurrent task limits
   - Review agent load balancing

### Debug Mode

Enable debug logging for detailed troubleshooting:

```yaml
# In config file
logging:
  level: DEBUG
  detailed_performance: true
  component_tracing: true
```

## Contributing

### Development Setup

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd sutazaiapp/localagi
   pip install -r requirements-dev.txt
   ```

2. **Run Tests**:
   ```bash
   pytest tests/ -v
   python3 scripts/test_localagi_system.py
   ```

3. **Code Style**:
   ```bash
   black localagi/
   flake8 localagi/
   mypy localagi/
   ```

### Architecture Guidelines

- **Modularity**: Each component should be independently testable
- **Async/Await**: Use async programming for all I/O operations
- **Error Handling**: Comprehensive error handling and recovery
- **Performance**: Optimize for concurrent execution and resource efficiency
- **Documentation**: Maintain comprehensive docstrings and comments

## Security Considerations

- **Local Operation**: No external API dependencies reduce attack surface
- **Input Validation**: All inputs are validated and sanitized
- **Resource Limits**: Built-in protections against resource exhaustion
- **Access Control**: Configurable access controls for system operations
- **Audit Logging**: Comprehensive logging of all system operations

## Performance Optimization

### Recommended Settings

- **CPU**: Minimum 4 cores, recommended 8+ cores
- **Memory**: Minimum 8GB, recommended 16GB+ for full agent deployment
- **Storage**: SSD recommended for database and model storage
- **Network**: Low-latency network for optimal agent communication

### Scaling Considerations

- **Horizontal Scaling**: Multiple LocalAGI instances can coordinate
- **Agent Distribution**: Agents can be distributed across multiple hosts
- **Load Balancing**: Built-in load balancing across available agents
- **Resource Monitoring**: Automatic resource usage optimization

## License

This project is part of the SutazAI system and follows the project's licensing terms.

## Support

For support and questions:
- Review the comprehensive logs in `/opt/sutazaiapp/logs/`
- Run the diagnostic script: `python3 scripts/test_localagi_system.py`
- Check system status: `curl http://localhost:8115/api/status`
- Monitor system health through Grafana dashboard

---

*LocalAGI represents a significant advancement in autonomous AI system orchestration, providing unprecedented capabilities for self-managing, self-improving AI agent ecosystems.*