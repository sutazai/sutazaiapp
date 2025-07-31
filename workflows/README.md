# Dify Workflow Automation System for SutazAI AGI

This directory contains the complete Dify workflow automation system designed to make the 38-agent SutazAI system self-managing and user-friendly through visual interfaces.

## 📁 Directory Structure

```
workflows/
├── README.md                           # This file
├── dify_config.yaml                    # Main Dify platform configuration
├── templates/                          # Visual workflow templates
│   └── agent_coordination_patterns.json
├── interfaces/                         # No-code user interfaces
│   └── no_code_orchestrator.json
├── automation/                         # Automated workflows
│   ├── task_distribution_router.json
│   ├── performance_monitoring.json
│   └── self_healing_recovery.json
├── configs/                           # Integration configurations
│   └── agent_integration.yaml
├── scripts/                           # Deployment and management scripts
│   ├── deploy_dify_workflows.py
│   └── workflow_manager.py
└── deployments/                       # Docker configurations
    ├── docker-compose.dify.yml
    ├── Dockerfile.workflow_manager
    ├── Dockerfile.task_router
    ├── Dockerfile.self_healer
    ├── Dockerfile.monitor
    └── requirements.txt
```

## 🚀 Quick Start

### 1. Deploy the Dify Workflow System

```bash
# Run the deployment script
cd /opt/sutazaiapp
python3 workflows/scripts/deploy_dify_workflows.py

# Add Dify services to docker-compose.yml
cat workflows/deployments/docker-compose.dify.yml >> docker-compose.yml

# Start the Dify services
docker-compose up -d dify-api dify-web workflow-manager
```

### 2. Access the Interfaces

- **Dify Web Interface**: http://localhost:8108
- **Workflow Manager API**: http://localhost:8109
- **Agent Registry**: http://localhost:8300
- **Grafana Monitoring**: http://localhost:3000

### 3. Create Your First Workflow

1. Open the Dify web interface
2. Import a template from `templates/agent_coordination_patterns.json`
3. Customize using the visual drag-and-drop editor
4. Deploy and execute your workflow

## 🎯 Key Features

### Visual Workflow Templates

**Sequential Agent Chain**
- Chain multiple agents in sequence for complex task processing
- Perfect for code review → security scan → deployment pipelines

**Parallel Agent Swarm**
- Execute multiple agents simultaneously for distributed processing
- Ideal for parallel development tasks across frontend, backend, and AI

**Hierarchical Agent Tree**
- Tree structure with master agent coordinating specialized sub-agents
- Best for complex system architecture decisions

**Collaborative Agent Mesh**
- Agents collaborate dynamically based on expertise and availability
- Great for research and problem-solving tasks

### No-Code Interface

The visual interface provides:
- **Drag-and-drop** workflow builder
- **Real-time validation** of workflow logic
- **Agent capability** matching
- **Execution monitoring** with live updates
- **Template library** for common patterns
- **Export/import** workflows as JSON/YAML

### Automated Task Distribution

Intelligent routing based on:
- **Capability matching** (40% weight): Match task requirements with agent skills
- **Workload balancing** (30% weight): Distribute to least loaded agents
- **Performance optimization** (20% weight): Route to best performing agents
- **Availability check** (10% weight): Ensure target agent is healthy

### Performance Monitoring

Real-time metrics:
- **Response times** with P95/P99 percentiles
- **Success rates** per agent
- **Throughput** measurements
- **Resource usage** (CPU, memory, queue depth)
- **Error rates** with categorization

### Self-Healing System

Autonomous recovery for:
- **Agent failures**: Automatic restart with backoff
- **Resource exhaustion**: Memory cleanup and reallocation
- **Load imbalances**: Dynamic load redistribution
- **Dependency issues**: Circuit breaker activation
- **Data corruption**: Backup restoration with validation

## 🔧 Configuration

### Model Configuration (Lightweight)

For initial deployment, the system uses lightweight models:
```yaml
models:
  providers:
    ollama:
      models:
        - name: "tinyllama:latest"      # 1.1B parameters
        - name: "llama3.2:1b"          # 1B parameters  
        - name: "qwen2.5:0.5b"         # 0.5B parameters
```

### Agent Integration

All 38 SutazAI agents are pre-configured with:
- **Capability mappings** for intelligent routing
- **API endpoints** for communication
- **Timeout settings** optimized per agent type
- **Retry policies** with exponential backoff
- **Resource limits** to prevent overload

### Security

- **JWT authentication** for API access
- **RBAC** with admin/operator/viewer roles
- **Rate limiting** (100 requests/minute default)
- **Input validation** and sanitization
- **Audit logging** for all workflow executions

## 📊 Monitoring & Alerting

### Grafana Dashboards

1. **Agent Overview Dashboard**
   - Health status grid for all agents
   - Response time distribution
   - Task throughput metrics
   - Error rates by agent

2. **System Performance Dashboard**
   - Resource usage gauges
   - Workflow execution heatmaps
   - Message bus metrics
   - Queue depth monitoring

### Alert Rules

- **Agent Down**: Triggered when agent becomes unresponsive
- **High Response Time**: P95 > 30 seconds for 5 minutes
- **High Error Rate**: >10% errors for 3 minutes
- **Resource Exhaustion**: >90% memory usage for 2 minutes
- **Queue Overflow**: >100 pending tasks for 1 minute

## 🔄 Workflow Patterns

### 1. Code Development Workflow
```
Input → Product Manager → Architect → Developer → QA → Deployment
```

### 2. System Monitoring Workflow
```
Metrics Collection → Analysis → Issue Detection → Self-Healing → Validation
```

### 3. Research & Problem Solving
```
Problem Analysis → Web Search → Expert Consultation → Solution Synthesis
```

### 4. Security Audit Workflow
```
Code Scan → Vulnerability Assessment → Penetration Test → Report Generation
```

## 🛠️ Advanced Usage

### Custom Workflow Creation

```python
# Example: Create a custom workflow
workflow = {
    "name": "Custom AI Pipeline",
    "pattern": "sequential",
    "nodes": [
        {
            "id": "analyzer",
            "type": "agent",
            "agent_id": "ai-product-manager",
            "config": {"action": "analyze_requirements"}
        },
        {
            "id": "implementer", 
            "type": "agent",
            "agent_id": "senior-ai-engineer",
            "config": {"action": "implement_solution"}
        }
    ],
    "connections": [
        {"from": "analyzer", "to": "implementer"}
    ]
}
```

### API Integration

```bash
# Start workflow execution
curl -X POST http://localhost:8109/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "sequential_agent_chain",
    "input_data": {
      "task": "Optimize system performance",
      "priority": "high"
    }
  }'

# Check workflow status
curl http://localhost:8109/workflows/{execution_id}/status
```

## 🚨 Troubleshooting

### Common Issues

1. **Dify service won't start**
   - Check if PostgreSQL and Redis are running
   - Verify environment variables in docker-compose.yml
   - Check logs: `docker logs sutazai-dify-api`

2. **Agents not responding**
   - Verify agent registry is accessible
   - Check agent health endpoints
   - Review task routing configuration

3. **Workflows stuck in pending**
   - Check Redis queue: `redis-cli LLEN workflow_queue`
   - Verify workflow manager is running
   - Check for resource constraints

4. **Memory issues with models**
   - Switch to even smaller models (tinyllama only)
   - Reduce concurrent executions
   - Increase Docker memory limits

### Log Locations

- **Dify API**: `/opt/sutazaiapp/logs/dify_deployment.log`
- **Workflow Manager**: `/opt/sutazaiapp/logs/workflow_manager.log`
- **Docker logs**: `docker logs [container_name]`

## 📈 Performance Optimization

### For Resource-Constrained Environments

1. **Use smallest models**: Start with tinyllama:latest only
2. **Reduce concurrency**: Set `max_concurrent_tasks: 1` per agent
3. **Increase timeouts**: Allow more time for small model responses
4. **Enable caching**: Cache frequent agent responses
5. **Batch operations**: Group small tasks together

### Scaling Up

1. **Add GPU support**: Configure CUDA for larger models
2. **Horizontal scaling**: Deploy multiple Dify instances
3. **Load balancing**: Use nginx for traffic distribution
4. **Database optimization**: Tune PostgreSQL for performance

## 🤝 Contributing

To extend the workflow system:

1. **Add new patterns**: Create templates in `templates/`
2. **Custom agents**: Register in `configs/agent_integration.yaml`
3. **New interfaces**: Add components to `interfaces/`
4. **Monitoring**: Extend dashboards in Grafana

## 📝 License

This workflow system is part of the SutazAI AGI project and follows the same licensing terms.

---

**Next Steps:**
1. Deploy the system using the provided scripts
2. Start with simple sequential workflows
3. Gradually introduce parallel and hierarchical patterns
4. Monitor performance and optimize based on your hardware
5. Expand with custom workflows as your system grows

The Dify workflow system transforms your 38-agent infrastructure into a self-managing, user-friendly AI automation platform. Start simple with lightweight models and scale up as your requirements grow! 🚀