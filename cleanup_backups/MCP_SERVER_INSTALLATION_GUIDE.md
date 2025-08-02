# SutazAI MCP Server - Installation & Usage Guide

## 🎯 Overview

Your custom SutazAI MCP (Model Context Protocol) server is now ready! This server provides comprehensive integration between Claude Desktop and your existing SutazAI AGI/ASI system, exposing powerful tools for:

- **23+ AI Agent Types** (AutoGPT, CrewAI, LocalAGI, etc.)
- **Model Management** via Ollama
- **Vector Knowledge Base** querying
- **Multi-Agent Orchestration**
- **System Monitoring & Health Checks**

## 🚀 Quick Installation

### Option 1: Automated Setup (Recommended)

```bash
# Navigate to MCP server directory
cd /opt/sutazaiapp/mcp_server

# Run automated setup
./setup.sh
```

The setup script will:
- ✅ Install Node.js dependencies
- ✅ Initialize database schema
- ✅ Configure Docker integration  
- ✅ Set up Claude Desktop configuration
- ✅ Test all connections
- ✅ Start the MCP server

### Option 2: Manual Installation

```bash
# 1. Install dependencies
cd /opt/sutazaiapp/mcp_server
npm install

# 2. Configure environment
cp config.example.env .env
# Edit .env with your settings

# 3. Initialize database
docker exec -i sutazai-postgres psql -U sutazai -d sutazai < database/schema.sql

# 4. Start MCP server
cd /opt/sutazaiapp
docker-compose up -d mcp-server

# 5. Configure Claude Desktop manually
# Edit ~/.config/claude/claude_desktop_config.json
```

## 📋 Verification Steps

### 1. Check MCP Server Status

```bash
# Check if container is running
docker ps | grep sutazai-mcp-server

# View logs
docker logs sutazai-mcp-server

# Test database connection
docker exec sutazai-mcp-server node -e "
  const pg = require('pg');
  const client = new pg.Client({connectionString: process.env.DATABASE_URL});
  client.connect().then(() => console.log('✓ Database OK')).catch(console.error);
"
```

### 2. Run Test Suite

```bash
cd /opt/sutazaiapp/mcp_server
npm test
```

Expected output:
```
✓ Configuration validation passed
✓ Docker integration configured correctly  
✓ MCP server started successfully
✓ All 7 tools are available
✓ All 6 resources are available
✓ Performance test passed (1245ms for 10 concurrent requests)

Success Rate: 100%
✓ All tests passed! MCP server is ready for production.
```

### 3. Test with MCP Inspector

```bash
# MCP Inspector will be available at http://localhost:6274
# Configuration is created automatically in mcp_inspector_config.json
```

### 4. Verify Claude Desktop Integration

1. **Restart Claude Desktop completely**
2. **Go to Settings → Developer**
3. **Check that "sutazai-mcp-server" shows as "running"**
4. **Look for new tools and resources in Claude chat**

## 🛠️ Available Tools in Claude

Once configured, you'll have access to these tools in Claude Desktop:

### 🤖 Agent Management
```
deploy_agent - Deploy new AI agents
execute_agent_task - Run tasks on specific agents  
manage_agent_workspace - Manage agent file storage
```

### 🧠 Model Operations  
```
manage_model - Pull, run, delete Ollama models
```

### 📚 Knowledge Base
```
query_knowledge_base - Search vector databases
```

### 📊 System Monitoring
```
monitor_system - Get performance metrics
```

### 🎯 Orchestration
```
orchestrate_multi_agent - Coordinate multiple agents
```

## 📋 Available Resources in Claude

Access these data sources directly in Claude:

- `sutazai://agents/list` - All AI agents and status
- `sutazai://models/available` - Ollama models  
- `sutazai://agents/tasks` - Task history
- `sutazai://system/metrics` - Performance data
- `sutazai://knowledge/embeddings` - Document database
- `sutazai://agents/workspaces` - Agent file storage

## 💡 Usage Examples

### Deploy and Use an Agent

In Claude Desktop:

```
Deploy a new AutoGPT agent called "research-bot" with research and analysis capabilities.

Then use the research-bot to analyze the latest trends in quantum computing and prepare a technical summary.
```

Claude will use the MCP tools to:
1. Deploy the agent via `deploy_agent`
2. Execute the task via `execute_agent_task`
3. Monitor progress and return results

### Query Knowledge Base

```
Search the knowledge base for "machine learning optimization techniques" and show me the most relevant documents.
```

Claude will use `query_knowledge_base` to search your vector databases.

### Multi-Agent Coordination

```
I need to create a full-stack web application. Coordinate multiple agents: one for backend API development, one for frontend React development, and one for DevOps deployment.
```

Claude will use `orchestrate_multi_agent` to coordinate the task across multiple specialized agents.

## 🔧 Configuration Files

### Main MCP Configuration
**File**: `/opt/sutazaiapp/.mcp.json`

Contains both your existing task-master-ai and new sutazai-mcp-server configurations.

### Claude Desktop Configuration  
**File**: `~/.config/claude/claude_desktop_config.json`

Automatically created by the setup script with proper environment variables.

### MCP Server Environment
**File**: `/opt/sutazaiapp/mcp_server/.env`

Database connections, API URLs, and feature flags.

### Docker Integration
**File**: `/opt/sutazaiapp/docker-compose.yml`

The MCP server is added as a new service with proper dependencies.

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. "MCP server not starting"
```bash
# Check logs
docker logs sutazai-mcp-server

# Verify dependencies
docker ps | grep -E "(postgres|redis|backend|ollama)"

# Restart with verbose logging
docker-compose restart mcp-server
```

#### 2. "Claude can't see the MCP server"
```bash
# Verify configuration
cat ~/.config/claude/claude_desktop_config.json | jq .

# Kill Claude completely and restart
pkill -f claude
# Wait 10 seconds, then restart Claude Desktop

# Check for typos in file paths
ls -la /opt/sutazaiapp/mcp_server/index.js
```

#### 3. "Database connection errors"
```bash
# Test PostgreSQL
docker exec sutazai-postgres pg_isready -U sutazai

# Check credentials in .env
grep DATABASE_URL /opt/sutazaiapp/mcp_server/.env

# Reinitialize schema if needed  
docker exec -i sutazai-postgres psql -U sutazai -d sutazai < /opt/sutazaiapp/mcp_server/database/schema.sql
```

#### 4. "Agent deployment fails"
```bash
# Check backend API
curl http://localhost:8000/health

# Verify agent registry
curl http://localhost:8000/api/v1/agents/list

# Check Docker socket access
docker exec sutazai-mcp-server ls -la /var/run/docker.sock
```

#### 5. "Performance issues"
```bash
# Monitor resources
docker stats sutazai-mcp-server

# Adjust connection pool
echo "CONNECTION_POOL_SIZE=30" >> /opt/sutazaiapp/mcp_server/.env
docker-compose restart mcp-server

# Optimize database
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "VACUUM ANALYZE;"
```

### Debug Mode

Enable detailed logging:

```bash
# Edit environment
echo "LOG_LEVEL=DEBUG" >> /opt/sutazaiapp/mcp_server/.env

# Restart server
docker-compose restart mcp-server

# Watch logs in real-time
docker logs -f sutazai-mcp-server
```

## 📊 Monitoring & Health

### Service Health Checks

```bash
# All services status
docker-compose ps

# MCP server health
docker exec sutazai-mcp-server node -e "console.log('MCP Server is healthy')"

# Database health
docker exec sutazai-postgres pg_isready -U sutazai

# Backend API health
curl http://localhost:8000/health
```

### Performance Monitoring

The MCP server integrates with your existing Prometheus/Grafana stack:

- **Metrics endpoint**: `http://localhost:9090/metrics`
- **Grafana dashboard**: Auto-configured
- **Alerts**: Set up for critical failures

### Log Locations

- **MCP Server**: `/opt/sutazaiapp/logs/mcp_server.log`
- **Setup Script**: `/opt/sutazaiapp/logs/mcp_setup.log`  
- **Docker Logs**: `docker logs sutazai-mcp-server`
- **Database Logs**: `docker logs sutazai-postgres`

## 🔒 Security & Best Practices

### Security Features
- ✅ Container isolation
- ✅ Non-root user execution
- ✅ Environment variable encryption
- ✅ Connection pooling limits
- ✅ Rate limiting
- ✅ Audit logging

### Best Practices
1. **Rotate passwords** regularly in `.env` files
2. **Monitor resource usage** to prevent abuse
3. **Update dependencies** monthly  
4. **Backup database** before major changes
5. **Use debug mode** only for troubleshooting

## 🚀 Advanced Usage

### Custom Agent Types

To add new agent types, edit the MCP server code:

```javascript
// In index.js, add to AgentType enum:
NEW_AGENT = "new_agent"

// Add deployment logic in deployAgent function
```

### Custom Resources

Add new resource URIs:

```javascript
// In index.js, add to ListResourcesRequestSchema handler:
{
  uri: "sutazai://custom/resource",
  mimeType: "application/json",
  name: "Custom Resource",
  description: "Your custom data source"
}
```

### Performance Tuning

Optimize for your workload:

```env
# High-performance settings
MAX_CONCURRENT_TASKS=25
CONNECTION_POOL_SIZE=50
TASK_TIMEOUT_SECONDS=600
CACHE_TTL_SECONDS=1800
```

## 📞 Support & Next Steps

### Getting Help

1. **Check this guide** for common solutions
2. **Run the test suite** to identify issues
3. **Enable debug logging** for detailed information  
4. **Review logs** for specific error messages

### Next Steps

With your MCP server running, you can:

1. **Deploy AI agents** for specific tasks
2. **Build complex workflows** with multi-agent orchestration  
3. **Query your knowledge base** for contextual information
4. **Monitor system health** in real-time
5. **Scale operations** as needed

### System Integration

Your MCP server integrates seamlessly with:
- ✅ **Claude Desktop** - Direct tool and resource access
- ✅ **SutazAI Backend** - Agent management and orchestration
- ✅ **Ollama** - Local AI model operations
- ✅ **Vector Databases** - ChromaDB, Qdrant, FAISS
- ✅ **PostgreSQL** - Persistent data storage
- ✅ **Redis** - Caching and task queues
- ✅ **Docker** - Container orchestration
- ✅ **Prometheus/Grafana** - Monitoring and metrics

## 🎉 Success!

Your SutazAI MCP server is now fully operational and integrated with Claude Desktop. You have access to the full power of your AGI/ASI system through a clean, context-aware interface.

**Key Capabilities Unlocked:**
- 🤖 Deploy and manage 23+ AI agent types
- 🧠 Control AI models via Ollama  
- 📚 Query vector knowledge bases
- 🎯 Orchestrate multi-agent workflows
- 📊 Monitor system health and performance
- 🔄 Automate complex tasks with context

**Happy Building! 🚀**

---

*Built with ❤️ for the SutazAI AGI/ASI ecosystem* 