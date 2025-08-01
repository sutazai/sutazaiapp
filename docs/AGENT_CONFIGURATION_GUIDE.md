# Agent Configuration Guide

This guide explains how to configure and use the AI agents in SutazAI for practical task automation.

## Agent Structure

Each agent is a specialized component designed for specific tasks. Here's how they work:

### Basic Agent Configuration

```json
{
  "name": "code-review-agent",
  "type": "reviewer",
  "capabilities": [
    "code_analysis",
    "quality_check",
    "suggestion_generation"
  ],
  "model": "tinyllama",
  "max_tokens": 2048,
  "temperature": 0.7
}
```

## Using Agents via API

### 1. List Available Agents

```bash
curl http://localhost:8000/api/v1/agents/
```

Response:
```json
{
  "agents": [
    {
      "name": "senior-ai-engineer",
      "description": "AI/ML implementation and optimization",
      "status": "available"
    },
    {
      "name": "code-generation-improver",
      "description": "Code quality and improvement analysis",
      "status": "available"
    }
    // ... more agents
  ]
}
```

### 2. Execute Agent Task

```bash
curl -X POST http://localhost:8000/api/v1/agents/execute \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "code-generation-improver",
    "task": "review",
    "data": {
      "code": "def calculate_sum(a, b):\n    return a + b",
      "language": "python"
    }
  }'
```

## Practical Agent Examples

### Code Review Agent

**Purpose**: Analyze code for improvements

```python
# Using in Python
import httpx

async def review_code(code_snippet):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/agents/execute",
            json={
                "agent": "code-generation-improver",
                "task": "review",
                "data": {
                    "code": code_snippet,
                    "checks": ["style", "bugs", "performance"]
                }
            }
        )
        return response.json()
```

### Security Scanning Agent

**Purpose**: Find security vulnerabilities

```python
async def security_scan(directory_path):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/agents/execute",
            json={
                "agent": "security-pentesting-specialist",
                "task": "scan",
                "data": {
                    "target": directory_path,
                    "scan_type": "comprehensive"
                }
            }
        )
        return response.json()
```

### Test Generation Agent

**Purpose**: Create unit tests automatically

```python
async def generate_tests(function_code):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/agents/execute",
            json={
                "agent": "testing-qa-validator",
                "task": "generate_tests",
                "data": {
                    "code": function_code,
                    "framework": "pytest"
                }
            }
        )
        return response.json()
```

## Agent Task Types

### Development Agents

| Agent | Tasks | Use Case |
|-------|-------|----------|
| `senior-ai-engineer` | implement, optimize, refactor | Complex code implementation |
| `code-generation-improver` | review, improve, suggest | Code quality analysis |
| `senior-backend-developer` | api_design, database_design | Backend architecture |

### Testing Agents

| Agent | Tasks | Use Case |
|-------|-------|----------|
| `testing-qa-validator` | generate_tests, validate, coverage | Automated testing |
| `security-pentesting-specialist` | scan, pentest, audit | Security assessment |

### Operations Agents

| Agent | Tasks | Use Case |
|-------|-------|----------|
| `deployment-automation-master` | deploy, rollback, monitor | CI/CD automation |
| `infrastructure-devops-manager` | provision, configure, scale | Infrastructure management |

## Configuration Best Practices

### 1. Task-Specific Configuration

```json
{
  "agent": "code-generation-improver",
  "task": "review",
  "config": {
    "focus_areas": ["security", "performance"],
    "severity_threshold": "medium",
    "include_suggestions": true
  }
}
```

### 2. Resource Limits

```json
{
  "agent": "testing-qa-validator",
  "resource_limits": {
    "max_execution_time": 300,  // seconds
    "max_memory": "1GB",
    "max_files": 100
  }
}
```

### 3. Output Formatting

```json
{
  "agent": "security-pentesting-specialist",
  "output_format": {
    "type": "json",
    "include_metadata": true,
    "group_by": "severity"
  }
}
```

## Common Workflows

### 1. Code Review Pipeline

```python
async def code_review_pipeline(file_path):
    # Step 1: Analyze code
    analysis = await analyze_code(file_path)
    
    # Step 2: Generate improvements
    improvements = await suggest_improvements(analysis)
    
    # Step 3: Create report
    report = await generate_report(improvements)
    
    return report
```

### 2. Security Assessment

```python
async def security_assessment(project_path):
    # Step 1: Code scanning
    code_scan = await scan_code(project_path)
    
    # Step 2: Dependency check
    dep_scan = await check_dependencies(project_path)
    
    # Step 3: Configuration audit
    config_audit = await audit_configs(project_path)
    
    return combine_results(code_scan, dep_scan, config_audit)
```

## Error Handling

```python
try:
    result = await execute_agent_task(agent_name, task_data)
except AgentNotAvailableError:
    # Handle agent unavailability
    pass
except TaskTimeoutError:
    # Handle timeout
    pass
except Exception as e:
    # General error handling
    logger.error(f"Agent task failed: {e}")
```

## Performance Tips

1. **Batch Operations**: Group similar tasks together
2. **Async Execution**: Use async/await for better performance
3. **Caching**: Cache frequent analysis results
4. **Resource Pooling**: Reuse agent connections

## Monitoring Agent Performance

```python
# Get agent metrics
metrics = await get_agent_metrics("code-generation-improver")
print(f"Tasks completed: {metrics['tasks_completed']}")
print(f"Average time: {metrics['avg_execution_time']}s")
print(f"Success rate: {metrics['success_rate']}%")
```

## Extending Agents

While the system comes with many pre-configured agents, you can extend their capabilities:

```python
# Custom task handler
async def custom_task_handler(agent, task_data):
    # Pre-process data
    processed_data = preprocess(task_data)
    
    # Execute agent task
    result = await agent.execute(processed_data)
    
    # Post-process results
    return postprocess(result)
```

## Troubleshooting

### Agent Not Responding

1. Check if the agent service is running:
   ```bash
   docker ps | grep agent-name
   ```

2. Check agent logs:
   ```bash
   docker logs sutazai-agent-name
   ```

3. Verify model is loaded:
   ```bash
   docker exec sutazai-ollama-tiny ollama list
   ```

### Task Timeout

- Increase timeout in configuration
- Break large tasks into smaller chunks
- Use streaming for long-running tasks

### Memory Issues

- Limit concurrent tasks
- Use smaller models for simple tasks
- Implement result pagination

## Summary

The agent system in SutazAI provides:
- **Specialized agents** for different tasks
- **Simple API** for integration
- **Local execution** for privacy
- **Flexible configuration** for customization
- **Practical workflows** for real tasks

Focus on using the right agent for each task and combining them for powerful automation workflows.