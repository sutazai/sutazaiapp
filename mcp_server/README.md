# SutazAI MCP Server

A comprehensive Model Context Protocol (MCP) server that provides seamless integration between Claude Desktop and the SutazAI automation system/advanced automation system. This server exposes powerful tools and resources for managing AI agents, models, knowledge bases, and system orchestration.

## 🚀 Features

### 🤖 Agent Management
- **Deploy AI Agents**: Launch 23+ different AI agent types (AutoGPT, CrewAI, LocalAGI, etc.)
- **Execute Tasks**: Submit tasks to specific agents with context and priority
- **Monitor Health**: Real-time agent health monitoring and status tracking
- **Workspace Management**: Manage agent workspaces and file persistence

### 🧠 Model Operations
- **Ollama Integration**: Full control over AI models via Ollama
- **Model Management**: Pull, delete, run, and monitor models
- **Performance Tracking**: Model usage statistics and optimization

### 📚 Knowledge Base
- **Vector Search**: Query ChromaDB, Qdrant, and FAISS vector databases
- **Document Management**: Access processed documents and embeddings
- **Semantic Search**: Context-aware knowledge retrieval

### 🎯 Multi-Agent Orchestration
- **Collaborative Tasks**: Coordinate multiple agents on complex tasks
- **Flexible Strategies**: Sequential, parallel, hierarchical, and collaborative coordination
- **Progress Tracking**: Real-time orchestration session monitoring

### 📊 System Monitoring
- **Performance Metrics**: CPU, memory, disk, and network monitoring
- **Health Checks**: Continuous system health validation
- **Resource Optimization**: Intelligent resource allocation and management

## 🏗️ Architecture

```mermaid
graph TB
    Claude[Claude Desktop] --> MCP[MCP Server]
    MCP --> DB[(PostgreSQL)]
    MCP --> Redis[(Redis Cache)]
    MCP --> Backend[SutazAI Backend API]
    MCP --> Ollama[Ollama Models]
    MCP --> Vector[Vector Databases]
    
    Backend --> Agents[AI Agents]
    Vector --> ChromaDB
    Vector --> Qdrant
    Vector --> FAISS
    
    Agents --> AutoGPT
    Agents --> CrewAI
    Agents --> LocalAGI
    Agents --> More[...]
```

## 📋 Available Resources

The MCP server exposes the following resources that Claude can access:

### `sutazai://agents/list`
Lists all registered AI agents with their status, capabilities, and health information.

```json
{
  "agents": [
    {
      "agent_name": "autogpt-001",
      "agent_type": "autogpt",
      "status": "active",
      "capabilities": ["code_generation", "file_operations", "web_search"],
      "last_seen": "2024-12-01T10:30:00Z"
    }
  ],
  "total_count": 15,
  "active_count": 12
}
```

### `sutazai://models/available`
Shows all AI models available through Ollama.

```json
{
  "models": [
    {
      "name": "deepseek-r1:8b",
      "size": 4800000000,
      "modified_at": "2024-12-01T09:15:00Z"
    }
  ],
  "total_count": 8
}
```

### `sutazai://agents/tasks`
Displays current and completed agent tasks.

### `sutazai://system/metrics`
Real-time system performance and health metrics.

### `sutazai://knowledge/embeddings`
Document embeddings and knowledge base entries.

### `sutazai://agents/workspaces`
Agent workspace data and file storage information.

## 🛠️ Available Tools

### `deploy_agent`
Deploy a new AI agent with specified configuration.

**Parameters:**
- `agent_type`: Type of agent (autogpt, crewai, localagi, etc.)
- `name`: Unique name for the agent instance
- `capabilities`: List of agent capabilities
- `config`: Agent-specific configuration

**Example:**
```json
{
  "agent_type": "autogpt",
  "name": "research-assistant-001",
  "capabilities": ["research", "analysis", "reporting"],
  "config": {
    "max_iterations": 10,
    "memory_backend": "redis"
  }
}
```

### `execute_agent_task`
Execute a task using a specific AI agent.

**Parameters:**
- `agent_name`: Name of the agent to use
- `task`: Task description or instruction
- `context`: Additional context for the task
- `priority`: Task priority (low, normal, high, urgent)

**Example:**
```json
{
  "agent_name": "research-assistant-001",
  "task": "Research the latest developments in advanced computing and prepare a summary",
  "context": {
    "domain": "technology",
    "target_audience": "technical"
  },
  "priority": "high"
}
```

### `manage_model`
Manage AI models through Ollama.

**Parameters:**
- `action`: Action to perform (pull, delete, list, run, stop)
- `model_name`: Name of the model
- `parameters`: Additional parameters for the action

**Example:**
```json
{
  "action": "pull",
  "model_name": "deepseek-r1:8b",
  "parameters": {}
}
```

### `query_knowledge_base`
Query the vector knowledge base for relevant information.

**Parameters:**
- `query`: Search query
- `collection`: Specific collection to search (optional)
- `limit`: Maximum number of results (default: 10)
- `similarity_threshold`: Minimum similarity score (default: 0.7)

**Example:**
```json
{
  "query": "machine learning optimization techniques",
  "collection": "research_papers",
  "limit": 5,
  "similarity_threshold": 0.8
}
```

### `monitor_system`
Monitor system health and performance metrics.

**Parameters:**
- `metric_type`: Type of metrics (cpu, memory, disk, network, containers, agents, models)
- `time_range`: Time range for metrics (1h, 6h, 24h, 7d)

### `manage_agent_workspace`
Manage agent workspaces and data persistence.

**Parameters:**
- `action`: Action to perform (create, delete, backup, restore, list)
- `agent_name`: Agent name for workspace operations
- `workspace_data`: Workspace configuration or data

### `orchestrate_multi_agent`
Orchestrate complex tasks across multiple AI agents.

**Parameters:**
- `task_description`: Overall task description
- `agents`: List of agents and their roles
- `coordination_strategy`: Coordination strategy (sequential, parallel, hierarchical, collaborative)

**Example:**
```json
{
  "task_description": "Develop a comprehensive market analysis report",
  "agents": [
    {
      "agent_name": "data-collector",
      "role": "data_gathering",
      "subtask": "Collect market data and statistics"
    },
    {
      "agent_name": "analyst",
      "role": "analysis",
      "subtask": "Analyze trends and patterns"
    },
    {
      "agent_name": "writer",
      "role": "reporting",
      "subtask": "Generate final report"
    }
  ],
  "coordination_strategy": "sequential"
}
```

## 🚀 Installation

### Prerequisites

- Node.js 18+ 
- Docker and Docker Compose
- PostgreSQL (via SutazAI system)
- Redis (via SutazAI system)
- Running SutazAI backend system

### Quick Start

1. **Navigate to the MCP server directory:**
   ```bash
   cd /opt/sutazaiapp/mcp_server
   ```

2. **Run the automated setup:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Verify installation:**
   ```bash
   npm test
   ```

### Manual Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure environment:**
   ```bash
   cp config.example.env .env
   # Edit .env with your specific configuration
   ```

3. **Initialize database:**
   ```bash
   docker exec -i sutazai-postgres psql -U sutazai -d sutazai < database/schema.sql
   ```

4. **Start the MCP server:**
   ```bash
   docker-compose up -d mcp-server
   ```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# Database Configuration
DATABASE_URL=postgresql://sutazai:password@localhost:5432/sutazai
REDIS_URL=redis://:password@localhost:6379/0

# Service URLs
BACKEND_API_URL=http://localhost:8000
OLLAMA_URL=http://localhost:11434
CHROMADB_URL=http://localhost:8000
QDRANT_URL=http://localhost:6333

# Performance Settings
MAX_CONCURRENT_TASKS=10
TASK_TIMEOUT_SECONDS=300
CONNECTION_POOL_SIZE=20

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=/opt/sutazaiapp/logs/mcp_server.log

# Features
ENABLE_AGENT_DEPLOYMENT=true
ENABLE_MODEL_MANAGEMENT=true
ENABLE_KNOWLEDGE_QUERIES=true
ENABLE_SYSTEM_MONITORING=true
ENABLE_MULTI_AGENT_ORCHESTRATION=true
```

### Claude Desktop Configuration

The setup script automatically creates the Claude Desktop configuration:

**Linux/WSL2:**
```json
{
  "mcpServers": {
    "sutazai-mcp-server": {
      "command": "node",
      "args": ["/opt/sutazaiapp/mcp_server/index.js"],
      "env": {
        "DATABASE_URL": "postgresql://sutazai:${POSTGRES_PASSWORD}@localhost:5432/sutazai",
        "REDIS_URL": "redis://:${REDIS_PASSWORD}@localhost:6379/0",
        "BACKEND_API_URL": "http://localhost:8000",
        "OLLAMA_URL": "http://localhost:11434"
      }
    }
  }
}
```

**Configuration Locations:**
- **Linux/WSL2**: `~/.config/claude/claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/claude/claude_desktop_config.json`  
- **Windows**: `%APPDATA%/claude/claude_desktop_config.json`

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
npm test

# Run specific test categories
npm run test:unit
npm run test:integration
npm run test:performance
```

### Test with MCP Inspector

1. **Start MCP Inspector:**
   ```bash
   # MCP Inspector runs on http://localhost:6274
   # Configuration is automatically created in mcp_inspector_config.json
   ```

2. **Access the inspector:**
   Open `http://localhost:6274` in your browser

3. **Test resources and tools:**
   - Browse available resources
   - Execute tools with sample data
   - Monitor request/response flow

### Manual Testing

```bash
# Test server startup
node index.js

# Test database connectivity
docker exec sutazai-mcp-server node -e "
  const pg = require('pg');
  const client = new pg.Client({connectionString: process.env.DATABASE_URL});
  client.connect().then(() => console.log('✓ Database connected')).catch(console.error);
"

# Check agent deployment
curl -X POST http://localhost:8000/api/v1/agents/deploy \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "autogpt", "name": "test-agent"}'
```

## 🔧 Usage Examples

### 1. Deploy and Use an Agent

```javascript
// In Claude Desktop, use the MCP tools:

// Deploy an agent
deploy_agent({
  agent_type: "autogpt",
  name: "code-assistant",
  capabilities: ["code_generation", "debugging", "testing"],
  config: {
    max_iterations: 15,
    memory_backend: "redis"
  }
})

// Execute a task
execute_agent_task({
  agent_name: "code-assistant",
  task: "Create a Python function to calculate fibonacci numbers with memoization",
  context: {
    language: "python",
    optimization: "performance"
  },
  priority: "normal"
})
```

### 2. Query Knowledge Base

```javascript
// Search for relevant information
query_knowledge_base({
  query: "optimal database indexing strategies for time-series data",
  collection: "technical_docs",
  limit: 5,
  similarity_threshold: 0.8
})
```

### 3. Orchestrate Multiple Agents

```javascript
// Coordinate multiple agents for a complex task
orchestrate_multi_agent({
  task_description: "Create a full-stack web application for task management",
  agents: [
    {
      agent_name: "backend-dev",
      role: "backend_development", 
      subtask: "Design and implement REST API"
    },
    {
      agent_name: "frontend-dev",
      role: "frontend_development",
      subtask: "Create React user interface"
    },
    {
      agent_name: "devops-engineer",
      role: "deployment",
      subtask: "Set up CI/CD and deployment pipeline"
    }
  ],
  coordination_strategy: "parallel"
})
```

### 4. Monitor System Health

```javascript
// Get comprehensive system metrics
monitor_system({
  metric_type: "agents",
  time_range: "24h"
})

// Check model availability
manage_model({
  action: "list"
})
```

## 🐛 Troubleshooting

### Common Issues

**1. MCP Server Not Starting**
```bash
# Check logs
docker logs sutazai-mcp-server

# Verify database connectivity
docker exec sutazai-postgres pg_isready -U sutazai

# Check environment variables
docker exec sutazai-mcp-server env | grep DATABASE_URL
```

**2. Claude Desktop Not Connecting**
```bash
# Verify configuration file exists and is valid
cat ~/.config/claude/claude_desktop_config.json | jq .

# Restart Claude Desktop completely
pkill -f claude
# Wait 5 seconds, then restart Claude Desktop

# Check Claude Desktop logs (if available)
```

**3. Database Connection Errors**
```bash
# Check PostgreSQL service
docker ps | grep postgres

# Test connection manually
docker exec -it sutazai-postgres psql -U sutazai -d sutazai

# Verify schema is applied
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "\dt"
```

**4. Agent Deployment Failures**
```bash
# Check backend API health
curl http://localhost:8000/health

# Verify agent registry
curl http://localhost:8000/api/v1/agents/list

# Check Docker daemon access
docker ps
```

**5. Performance Issues**
```bash
# Monitor resource usage
docker stats sutazai-mcp-server

# Check connection pool
docker exec sutazai-mcp-server node -e "console.log(process.env.CONNECTION_POOL_SIZE)"

# Optimize PostgreSQL
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "VACUUM ANALYZE;"
```

### Debug Mode

Enable debug logging:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or modify .env file
echo "LOG_LEVEL=DEBUG" >> .env

# Restart MCP server
docker-compose restart mcp-server

# View detailed logs
docker logs -f sutazai-mcp-server
```

### Health Checks

```bash
# Check all service health
docker-compose ps

# Specific health check
docker exec sutazai-mcp-server node -e "process.exit(0)"

# Database health
docker exec sutazai-postgres pg_isready -U sutazai

# Redis health  
docker exec sutazai-redis redis-cli ping
```

## 📊 Performance Optimization

### Database Optimization

```sql
-- Run in PostgreSQL to optimize performance
VACUUM ANALYZE;
REINDEX DATABASE sutazai;

-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

### Connection Pooling

Adjust pool settings in `.env`:

```env
CONNECTION_POOL_SIZE=20
MAX_CONCURRENT_TASKS=10
TASK_TIMEOUT_SECONDS=300
```

### Caching Strategy

Configure Redis caching:

```env
CACHE_TTL_SECONDS=3600
ENABLE_QUERY_CACHING=true
CACHE_AGENT_STATUS=true
```

## 🔒 Security

### Authentication

The MCP server integrates with SutazAI's authentication system:

```env
# API key for backend communication
API_KEY=your-secure-api-key

# Secret key for session management
SECRET_KEY=your-secret-key
```

### Network Security

- All inter-service communication uses Docker networks
- Database connections use connection pooling with limits
- File system access is containerized and restricted

### Data Privacy

- Agent workspaces are isolated
- Sensitive data is encrypted at rest
- Logs can be configured to exclude sensitive information

## 📈 Monitoring and Metrics

### Prometheus Integration

The MCP server exposes metrics for Prometheus:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'sutazai-mcp-server'
    static_configs:
      - targets: ['mcp-server:9090']
```

### Available Metrics

- Request count and latency
- Agent deployment success rate
- Database connection pool usage
- Memory and CPU utilization
- Error rates by endpoint

### Grafana Dashboard

Import the included Grafana dashboard for visualization:

```bash
# Dashboard configuration in monitoring/grafana/dashboards/
cat monitoring/grafana/dashboards/mcp-server.json
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/sutazai/mcp-server
cd mcp-server

# Install dependencies
npm install

# Start in development mode
npm run dev

# Run tests
npm test

# Lint code
npm run lint
```

### Code Structure

```
mcp_server/
├── index.js              # Main server implementation
├── package.json           # Dependencies and scripts
├── Dockerfile            # Container configuration
├── database/
│   └── schema.sql        # Database schema
├── test.js               # Test suite
├── setup.sh              # Installation script
└── README.md             # This file
```

### Testing Guidelines

- Write tests for all new tools and resources
- Ensure database tests clean up after themselves
- Test error handling and edge cases
- Maintain > 80% code coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://github.com/modelcontextprotocol) - The foundational protocol
- [Anthropic Claude](https://claude.ai) - AI assistant platform
- [SutazAI Team](https://github.com/sutazai) - The underlying automation system/advanced automation system
- [Ollama](https://ollama.ai) - Local AI model management

## 📞 Support

For issues and questions:

1. **Check the troubleshooting section** above
2. **Run the test suite** to identify specific issues  
3. **Check logs** for detailed error information
4. **Create an issue** on the GitHub repository

**Log Locations:**
- MCP Server: `/opt/sutazaiapp/logs/mcp_server.log`
- Setup Script: `/opt/sutazaiapp/logs/mcp_setup.log`
- Docker: `docker logs sutazai-mcp-server`

---

**Built with ❤️ for the SutazAI automation system/advanced automation ecosystem** 