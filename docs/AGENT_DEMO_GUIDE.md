# SutazAI Agent System - Complete Usage Guide

This guide demonstrates how to use all available AI agents in the SutazAI system, including agent communication through Redis, task orchestration, and common workflows.

## Quick Start

```bash
# 1. Ensure Redis and Ollama are running
systemctl start redis
systemctl start ollama

# 2. Install required models
ollama pull codellama
ollama pull llama2

# 3. Run the demo
python sutazai_agent_demo.py
```

## System Architecture

The SutazAI system consists of several key components:

### Core Components

1. **Agent Communication Bus**: Redis-based messaging system for inter-agent communication
2. **Agent Registry**: Service discovery and capability management
3. **Base Agent Framework**: Universal foundation for all agent types
4. **Specialized Agents**: Domain-specific agent implementations

### Available Agents

| Agent Type | Primary Capabilities | Use Cases |
|------------|---------------------|-----------|
| **Code Generator** | Code generation, completion, refactoring | Software development, code reviews |
| **Security Analyzer** | Vulnerability scanning, security analysis | Code security, compliance checks |
| **OpenDevin Generator** | Autonomous software engineering | Complex development tasks |
| **Dify Automation** | Workflow automation, AI applications | Process automation, RAG systems |
| **AgentGPT Executor** | Goal-driven task execution | Autonomous problem solving |
| **Langflow Designer** | Visual workflow creation | No-code AI applications |
| **LocalAGI Orchestrator** | Multi-agent coordination | Complex orchestration tasks |
| **BigAGI Manager** | Conversational AI interfaces | Advanced chat systems |
| **FlowiseAI Manager** | Visual LangChain applications | LLM workflow design |
| **AgentZero Coordinator** | General-purpose task handling | Adaptive problem solving |

## Core Usage Patterns

### 1. Agent Instantiation

```python
from sutazai_agent_demo import SutazAIAgentDemo

# Initialize the demo system
demo = SutazAIAgentDemo(
    redis_url="redis://localhost:6379/0",
    ollama_url="http://localhost:11434"
)

# Initialize system components
await demo.initialize()

# Create all available agents
agents = await demo.create_all_agents()
```

### 2. Direct Agent Communication

```python
# Get a specific agent
code_agent = demo.active_agents['code_generator']

# Send a direct message
message_id = await code_agent.send_message(
    receiver_id="another_agent_id",
    message_type="generate_code",
    content={
        "specification": "Create a REST API endpoint",
        "language": "python",
        "framework": "FastAPI"
    }
)

# Broadcast to all agents
await code_agent.send_message(
    receiver_id="broadcast",
    message_type="system_announcement",
    content={"message": "System maintenance starting"}
)
```

### 3. Task Execution

```python
# Execute a code generation task
task_data = {
    "task_type": "generate_code",
    "specification": "Create a Python function that sorts a list",
    "language": "python",
    "code_type": "function"
}

result = await code_agent.execute_task("unique_task_id", task_data)

if result["success"]:
    generated_code = result["result"]["generated_code"]
    print(f"Generated code: {generated_code}")
else:
    print(f"Task failed: {result['error']}")
```

### 4. Agent Discovery and Capabilities

```python
# Find agents with specific capabilities
from backend.ai_agents.core.base_agent import AgentCapability

code_agents = [
    agent_id for agent_id, agent in demo.active_agents.items()
    if AgentCapability.CODE_GENERATION in agent.capabilities
]

security_agents = [
    agent_id for agent_id, agent in demo.active_agents.items()
    if AgentCapability.SECURITY_ANALYSIS in agent.capabilities
]

# Query agent capabilities
agent_info = await agent.get_agent_info()
print(f"Agent capabilities: {agent_info['capabilities']}")
```

### 5. Collaborative Workflows

```python
# Complex multi-agent workflow
async def collaborative_development_workflow():
    # Step 1: Generate code
    code_task = {
        "task_type": "generate_code",
        "specification": "Create a user authentication system",
        "language": "python"
    }
    code_result = await code_agent.execute_task("auth_gen", code_task)
    
    if code_result["success"]:
        generated_code = code_result["result"]["generated_code"]
        
        # Step 2: Security analysis
        security_task = {
            "task_type": "security_scan",
            "code": generated_code,
            "language": "python",
            "focus": "authentication"
        }
        security_result = await security_agent.execute_task("auth_scan", security_task)
        
        # Step 3: Generate documentation
        doc_task = {
            "task_type": "generate_documentation", 
            "code": generated_code,
            "format": "markdown"
        }
        doc_result = await doc_agent.execute_task("auth_docs", doc_task)
        
        return {
            "code": code_result,
            "security": security_result,
            "documentation": doc_result
        }
```

## Advanced Usage Patterns

### 1. Communication Bus Integration

```python
from backend.ai_agents.communication.agent_bus import AgentCommunicationBus

# Initialize communication bus
bus = AgentCommunicationBus("redis://localhost:6379/0")
await bus.initialize()

# Request task execution through the bus
result = await bus.request_task_execution(
    task={
        "type": "code_generation",
        "specification": "Create a web scraper",
        "language": "python"
    },
    required_capabilities={"code_generation", "web_scraping"},
    priority=MessagePriority.HIGH,
    timeout=300
)
```

### 2. Agent Registry Operations

```python
from backend.app.agents.registry import AgentRegistry

registry = AgentRegistry()

# Register a new agent
registry.register_agent(
    name="custom_agent",
    description="Custom specialized agent",
    capabilities=["custom_capability"],
    endpoint="http://localhost:8080",
    max_concurrent_tasks=3
)

# Find agents by capability
capable_agents = registry.find_agents_by_capability("code_generation")

# Get registry statistics
stats = registry.get_registry_stats()
print(f"Total agents: {stats['total_agents']}")
```

### 3. Custom Agent Implementation

```python
from backend.ai_agents.core.base_agent import BaseAgent, AgentConfig

class CustomAgent(BaseAgent):
    async def on_initialize(self):
        """Initialize custom agent"""
        self.register_message_handler("custom_task", self._handle_custom_task)
        
    async def on_task_execute(self, task_id: str, task_data: Dict[str, Any]):
        """Execute custom task logic"""
        task_type = task_data.get("task_type")
        
        if task_type == "custom_processing":
            # Your custom logic here
            result = await self._process_custom_task(task_data)
            return {
                "success": True,
                "result": result,
                "task_id": task_id
            }
        
        return {
            "success": False,
            "error": f"Unknown task type: {task_type}",
            "task_id": task_id
        }
    
    async def _handle_custom_task(self, message):
        """Handle custom message type"""
        # Process custom message
        result = await self.execute_task(message.id, message.content)
        
        # Send response
        await self.send_message(
            message.sender_id,
            "custom_task_result",
            result
        )
```

### 4. Error Handling Best Practices

```python
async def robust_task_execution(agent, task_data, max_retries=3):
    """Execute task with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            result = await agent.execute_task(f"task_{uuid.uuid4()}", task_data)
            
            if result.get("success"):
                return result
            else:
                logger.warning(f"Task failed (attempt {attempt + 1}): {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Task execution error (attempt {attempt + 1}): {e}")
            
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return {"success": False, "error": "Max retries exceeded"}

# Usage
result = await robust_task_execution(code_agent, {
    "task_type": "generate_code",
    "specification": "Create a database connection pool",
    "language": "python"
})
```

### 5. System Monitoring and Metrics

```python
async def monitor_system_health():
    """Monitor overall system health"""
    
    # Check agent registry
    registry_stats = registry.get_registry_stats()
    
    # Check communication bus
    bus_metrics = await bus.get_system_metrics()
    
    # Check individual agent health
    agent_health = {}
    for agent_id, agent in active_agents.items():
        try:
            info = await agent.get_agent_info()
            agent_health[agent_id] = {
                "status": info["status"],
                "uptime": info["uptime"],
                "task_count": info["task_count"],
                "error_count": info["error_count"],
                "healthy": info["error_count"] < 10  # Custom health criteria
            }
        except Exception as e:
            agent_health[agent_id] = {"healthy": False, "error": str(e)}
    
    return {
        "registry": registry_stats,
        "communication": bus_metrics,
        "agents": agent_health,
        "overall_health": all(
            agent.get("healthy", False) 
            for agent in agent_health.values()
        )
    }
```

## Configuration Examples

### Agent Configuration

```python
# Code Generator Agent
code_agent_config = AgentConfig(
    agent_id="code_gen_001",
    agent_type="CodeGeneratorAgent",
    name="Primary Code Generator",
    description="Specialized code generation and analysis agent",
    capabilities=[
        AgentCapability.CODE_GENERATION,
        AgentCapability.CODE_ANALYSIS,
        AgentCapability.REASONING
    ],
    model_config={
        "ollama_url": "http://localhost:11434",
        "model": "codellama:13b"
    },
    redis_config={
        "url": "redis://localhost:6379/0"
    },
    max_concurrent_tasks=5,
    heartbeat_interval=30,
    message_timeout=300
)

# Security Analyzer Agent
security_agent_config = AgentConfig(
    agent_id="security_001",
    agent_type="SecurityAnalyzerAgent",
    name="Security Scanner",
    description="Code security analysis and vulnerability detection",
    capabilities=[
        AgentCapability.SECURITY_ANALYSIS,
        AgentCapability.CODE_ANALYSIS,
        AgentCapability.TESTING
    ],
    model_config={
        "ollama_url": "http://localhost:11434",
        "model": "llama2:13b"
    },
    redis_config={
        "url": "redis://localhost:6379/0"
    },
    max_concurrent_tasks=3,
    heartbeat_interval=30
)
```

### Redis Configuration

```python
# Redis connection configuration
redis_config = {
    "url": "redis://localhost:6379/0",
    "connection_pool": {
        "max_connections": 20,
        "retry_on_timeout": True,
        "socket_connect_timeout": 5,
        "socket_timeout": 5
    },
    "channels": {
        "task_queue": "sutazai:tasks",
        "priority_queue": "sutazai:priority_tasks",
        "broadcast": "sutazai:broadcast",
        "agent_status": "sutazai:agent_status"
    }
}
```

## Real-World Workflow Examples

### 1. Complete Web Application Development

```python
async def build_web_application():
    """End-to-end web application development workflow"""
    
    # Step 1: Generate backend API
    api_task = {
        "task_type": "generate_code",
        "specification": """
        Create a FastAPI application with:
        - User authentication (JWT)
        - CRUD operations for users
        - Database integration (SQLAlchemy)
        - Input validation (Pydantic)
        """,
        "language": "python",
        "framework": "FastAPI"
    }
    
    api_result = await opendevin_agent.execute_task("api_gen", api_task)
    
    # Step 2: Security analysis
    if api_result["success"]:
        security_task = {
            "task_type": "security_scan",
            "code": api_result["result"]["generated_code"],
            "language": "python",
            "focus": "authentication,sql_injection,xss"
        }
        
        security_result = await security_agent.execute_task("api_security", security_task)
        
        # Step 3: Generate frontend
        if security_result["success"] and not security_result["result"]["high_risk_issues"]:
            frontend_task = {
                "task_type": "generate_frontend",
                "api_spec": api_result["result"]["api_specification"],
                "framework": "React",
                "features": ["authentication", "user_management", "responsive_design"]
            }
            
            frontend_result = await langflow_agent.execute_task("frontend_gen", frontend_task)
            
            # Step 4: Generate deployment configuration
            deploy_task = {
                "task_type": "generate_deployment",
                "application_type": "web_app",
                "backend_language": "python",
                "frontend_framework": "react",
                "deployment_target": "docker"
            }
            
            deploy_result = await dify_agent.execute_task("deploy_gen", deploy_task)
            
            return {
                "backend": api_result,
                "security_analysis": security_result,
                "frontend": frontend_result,
                "deployment": deploy_result,
                "success": True
            }
    
    return {"success": False, "error": "Workflow failed at security analysis"}
```

### 2. Automated Code Review Process

```python
async def automated_code_review(code_files):
    """Comprehensive automated code review workflow"""
    
    review_results = {}
    
    for file_path, code_content in code_files.items():
        # Step 1: Static analysis
        analysis_task = {
            "task_type": "analyze_code",
            "code": code_content,
            "language": "python",
            "analysis_type": "comprehensive"
        }
        
        analysis_result = await code_agent.execute_task(
            f"analysis_{file_path.replace('/', '_')}", 
            analysis_task
        )
        
        # Step 2: Security scan
        security_task = {
            "task_type": "security_scan", 
            "code": code_content,
            "language": "python",
            "rules": ["owasp-top-10", "pci-compliance"]
        }
        
        security_result = await security_agent.execute_task(
            f"security_{file_path.replace('/', '_')}",
            security_task
        )
        
        # Step 3: Performance analysis
        performance_task = {
            "task_type": "performance_analysis",
            "code": code_content,
            "language": "python",
            "focus": ["complexity", "memory_usage", "bottlenecks"]
        }
        
        performance_result = await agentgpt_agent.execute_task(
            f"performance_{file_path.replace('/', '_')}",
            performance_task
        )
        
        # Aggregate results
        review_results[file_path] = {
            "static_analysis": analysis_result,
            "security_scan": security_result,
            "performance_analysis": performance_result,
            "overall_score": calculate_code_quality_score(
                analysis_result, security_result, performance_result
            )
        }
    
    return review_results

def calculate_code_quality_score(analysis, security, performance):
    """Calculate overall code quality score"""
    score = 0
    
    # Static analysis score (0-40 points)
    if analysis.get("success"):
        issues = analysis["result"].get("issues", [])
        critical_issues = len([i for i in issues if i["severity"] == "critical"])
        score += max(0, 40 - (critical_issues * 10))
    
    # Security score (0-40 points)  
    if security.get("success"):
        vulnerabilities = security["result"].get("vulnerabilities", [])
        high_risk = len([v for v in vulnerabilities if v["risk"] == "high"])
        score += max(0, 40 - (high_risk * 15))
    
    # Performance score (0-20 points)
    if performance.get("success"):
        complexity = performance["result"].get("complexity_score", 10)
        score += max(0, 20 - complexity)
    
    return min(100, score)
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis status
   systemctl status redis
   
   # Start Redis if not running
   sudo systemctl start redis
   
   # Test connection
   redis-cli ping
   ```

2. **Ollama Model Not Found**
   ```bash
   # List available models
   ollama list
   
   # Pull required models
   ollama pull codellama
   ollama pull llama2
   ```

3. **Agent Initialization Failed**
   ```python
   # Check agent logs
   logging.basicConfig(level=logging.DEBUG)
   
   # Verify configuration
   agent_config = AgentConfig(...)
   print(f"Config valid: {agent_config}")
   ```

4. **Task Execution Timeout**
   ```python
   # Increase timeout in agent config
   agent_config.message_timeout = 600  # 10 minutes
   
   # Or handle timeout gracefully
   try:
       result = await asyncio.wait_for(
           agent.execute_task(task_id, task_data),
           timeout=300
       )
   except asyncio.TimeoutError:
       logger.error("Task execution timed out")
   ```

### Performance Optimization

1. **Agent Pool Management**
   ```python
   # Limit concurrent agents
   max_agents = 5
   semaphore = asyncio.Semaphore(max_agents)
   
   async def create_agent_with_limit(config):
       async with semaphore:
           return await create_agent(config)
   ```

2. **Redis Connection Pooling**
   ```python
   # Configure connection pool
   redis_pool = aioredis.ConnectionPool.from_url(
       "redis://localhost:6379/0",
       max_connections=20,
       retry_on_timeout=True
   )
   ```

3. **Task Batching**
   ```python
   # Batch similar tasks
   async def batch_code_generation(specifications):
       tasks = []
       for spec in specifications:
           task = agent.execute_task(f"batch_{uuid.uuid4()}", {
               "task_type": "generate_code",
               "specification": spec
           })
           tasks.append(task)
       
       return await asyncio.gather(*tasks, return_exceptions=True)
   ```

## Best Practices

1. **Always use try-catch blocks** for agent operations
2. **Implement proper logging** throughout your workflows
3. **Monitor agent health** and performance metrics
4. **Use appropriate timeouts** for long-running tasks
5. **Clean up resources** when shutting down agents
6. **Validate task data** before sending to agents
7. **Handle partial failures** in multi-agent workflows
8. **Use exponential backoff** for retry logic
9. **Monitor Redis memory usage** with many agents
10. **Test error scenarios** in development

## Next Steps

- Explore advanced orchestration patterns
- Implement custom agent types for your domain
- Set up monitoring and alerting systems
- Scale horizontally with multiple Redis instances
- Integrate with external services and APIs

For more examples and advanced usage, see the included demo file: `sutazai_agent_demo.py`