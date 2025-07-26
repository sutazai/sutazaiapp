# SutazAI AGI/ASI System - Advanced Usage Guide

## 🚀 Advanced Features & Capabilities

### 1. Multi-Model AI Interactions

#### Available Models & Their Strengths
- **DeepSeek-R1 8B** - Advanced reasoning, complex problem solving
- **Qwen2.5 3B** - Balanced performance, general tasks
- **Llama3.2 1B** - Fast responses, simple queries

#### Model Selection Strategy
```python
# For complex reasoning tasks
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze the implications of quantum computing on AGI development",
    "model": "deepseek-r1:8b"
  }'

# For quick responses
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is 2+2?",
    "model": "llama3.2:1b"
  }'
```

### 2. Self-Improvement System

The system includes autonomous code improvement capabilities:

#### Activate Self-Improvement
```python
# Analyze and improve system code
curl -X POST http://localhost:8000/api/self-improve \
  -H "Content-Type: application/json" \
  -d '{
    "target": "backend",
    "focus": "performance",
    "auto_apply": false
  }'
```

#### Monitor Improvements
```bash
# Check improvement logs
docker logs sutazai-backend | grep "improvement"

# View generated improvements
ls -la workspace/improvements/
```

### 3. Agent Orchestration

#### Deploy Additional Agents
```bash
# Start specific agent
docker-compose up -d gpt-engineer

# Scale agents
docker-compose up -d --scale aider=3
```

#### Agent Communication
```python
# Route task to specific agent
curl -X POST http://localhost:8000/api/agents/route \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Generate unit tests for user.py",
    "agent_type": "aider",
    "priority": "high"
  }'
```

### 4. Knowledge Management

#### Create Knowledge Graph
```python
# Add knowledge to graph
curl -X POST http://localhost:8000/api/knowledge/add \
  -H "Content-Type: application/json" \
  -d '{
    "entity": "AGI System",
    "properties": {
      "components": ["reasoning", "learning", "adaptation"],
      "capabilities": ["self-improvement", "multi-agent"]
    },
    "relationships": [
      {"to": "DeepSeek Model", "type": "uses"},
      {"to": "Knowledge Graph", "type": "maintains"}
    ]
  }'
```

#### Query Knowledge
```python
# Semantic search
curl -X POST http://localhost:8000/api/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does the AGI system learn?",
    "limit": 5
  }'
```

### 5. Reasoning Engine

#### Multi-Type Reasoning
```python
# Deductive reasoning
curl -X POST http://localhost:8000/api/reason \
  -H "Content-Type: application/json" \
  -d '{
    "type": "deductive",
    "premises": [
      "All AGI systems can self-improve",
      "SutazAI is an AGI system"
    ],
    "query": "Can SutazAI self-improve?"
  }'

# Causal reasoning
curl -X POST http://localhost:8000/api/reason \
  -H "Content-Type: application/json" \
  -d '{
    "type": "causal",
    "events": ["code change", "performance increase"],
    "analyze": "relationship"
  }'
```

### 6. Advanced Workflows

#### Create Complex Workflow
```python
# Multi-step AGI workflow
curl -X POST http://localhost:8000/api/workflows/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AGI Analysis Pipeline",
    "steps": [
      {
        "action": "analyze_code",
        "agent": "semgrep",
        "params": {"target": "backend/"}
      },
      {
        "action": "generate_improvements",
        "agent": "gpt-engineer",
        "params": {"based_on": "previous_analysis"}
      },
      {
        "action": "test_changes",
        "agent": "aider",
        "params": {"create_tests": true}
      },
      {
        "action": "apply_improvements",
        "agent": "self-improvement",
        "params": {"review_required": true}
      }
    ]
  }'
```

### 7. System Monitoring

#### Real-time Metrics
```bash
# System performance
curl http://localhost:8000/api/metrics

# Model performance
curl http://localhost:8000/api/models/performance

# Agent status
curl http://localhost:8000/api/agents/status
```

#### Advanced Monitoring
```bash
# Start monitoring stack
docker-compose up -d prometheus grafana

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### 8. Security & Access Control

#### API Authentication
```python
# Get API token
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "secure_password"
  }'

# Use token for requests
curl -X GET http://localhost:8000/api/protected \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 9. Batch Processing

#### Process Multiple Requests
```python
# Batch chat requests
curl -X POST http://localhost:8000/api/batch/chat \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"message": "What is AGI?", "model": "qwen2.5:3b"},
      {"message": "Explain ASI", "model": "deepseek-r1:8b"},
      {"message": "Future of AI", "model": "llama3.2:1b"}
    ],
    "parallel": true
  }'
```

### 10. Custom Extensions

#### Add Custom Agent
```python
# Register new agent
curl -X POST http://localhost:8000/api/agents/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom-agent",
    "url": "http://custom-agent:8080",
    "capabilities": ["custom_analysis"],
    "health_endpoint": "/health"
  }'
```

## 🔧 Performance Tuning

### Model Loading Optimization
```bash
# Preload models at startup
docker exec sutazai-ollama ollama run deepseek-r1:8b "test" --keepalive 0

# Adjust memory allocation
docker update --memory="8g" sutazai-ollama
```

### Database Optimization
```sql
-- Connect to PostgreSQL
docker exec -it sutazai-postgres psql -U sutazai

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM conversations WHERE model='deepseek-r1:8b';

-- Create indexes
CREATE INDEX idx_model ON conversations(model);
CREATE INDEX idx_timestamp ON conversations(created_at);
```

## 📊 Usage Examples

### Complete AGI Demonstration
```bash
# 1. Start a reasoning session
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/sessions/create | jq -r '.session_id')

# 2. Add context
curl -X POST http://localhost:8000/api/sessions/$SESSION_ID/context \
  -H "Content-Type: application/json" \
  -d '{
    "context": "You are an AGI system with self-improvement capabilities",
    "objectives": ["analyze", "reason", "improve"]
  }'

# 3. Execute complex task
curl -X POST http://localhost:8000/api/sessions/$SESSION_ID/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Analyze your own architecture and suggest improvements",
    "model": "deepseek-r1:8b",
    "use_reasoning": true,
    "use_knowledge": true
  }'
```

## 🚀 Advanced Tips

1. **Model Chaining**: Use different models for different parts of complex tasks
2. **Knowledge Persistence**: Regular knowledge graph backups maintain learning
3. **Agent Specialization**: Assign specific agents to specific task types
4. **Performance Monitoring**: Use Grafana dashboards for real-time insights
5. **Continuous Learning**: Enable auto-improvement for gradual enhancement

## 📝 Next Steps

1. Explore the API documentation at http://localhost:8000/docs
2. Experiment with different model combinations
3. Build custom workflows for your use cases
4. Monitor system performance and optimize
5. Contribute improvements back to the system

---
Your AGI/ASI system is ready for advanced experimentation!