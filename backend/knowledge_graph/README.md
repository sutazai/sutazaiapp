# SutazAI Knowledge Graph System

A comprehensive knowledge graph system for the SutazAI platform that provides intelligent mapping, querying, and reasoning capabilities for AI agents, services, and system architecture.

## Overview

The Knowledge Graph System automatically extracts, maps, and maintains relationships between:
- **69 AI agents** with their capabilities and specializations
- **Service dependencies** and interactions
- **Data flow patterns** across the system
- **System architecture** components and configurations
- **Documentation** and knowledge artifacts

## Features

### 🔍 **Intelligent Discovery**
- Automatic agent capability detection
- Service dependency mapping
- Configuration analysis
- Documentation parsing
- Code structure analysis

### 🧠 **Advanced Reasoning**
- Rule-based inference engine
- Capability-based agent selection
- Performance optimization recommendations
- System health insights
- Anomaly detection

### 📊 **Interactive Visualization**
- D3.js-powered interactive graphs
- Multiple layout algorithms (force-directed, hierarchical, circular)
- Real-time filtering and search
- Export capabilities (SVG, PNG, JSON)
- Responsive design

### ⚡ **Real-time Updates**
- File system monitoring
- Agent status synchronization
- Configuration change detection
- Event-driven updates
- Conflict resolution

### 🔌 **RESTful API**
- Comprehensive query endpoints
- Natural language processing
- Agent discovery services
- System analytics
- Health monitoring

## Architecture

```
knowledge_graph/
├── __init__.py                 # Package initialization
├── schema.py                   # Graph schema and ontology
├── neo4j_manager.py           # Neo4j database operations
├── graph_builder.py           # Knowledge extraction pipeline
├── query_engine.py            # Graph querying and search
├── reasoning_engine.py        # Inference and reasoning
├── visualization.py           # Graph visualization
├── real_time_updater.py       # Real-time synchronization
├── api.py                     # FastAPI endpoints
├── manager.py                 # System coordinator
└── README.md                  # This file
```

## Quick Start

### 1. Prerequisites

```bash
# Install Neo4j (Docker recommended)
docker run \
  --name neo4j-sutazai \
  -p 7474:7474 -p 7687:7687 \
  -d \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.15.0
```

### 2. Environment Configuration

```bash
# Set environment variables
export KG_NEO4J_URI="bolt://localhost:7687"
export KG_NEO4J_USERNAME="neo4j"
export KG_NEO4J_PASSWORD="password"
export KG_NEO4J_DATABASE="sutazai"
export KG_BASE_PATH="/opt/sutazaiapp/backend"
export KG_ENABLE_REAL_TIME="true"
export KG_ENABLE_REASONING="true"
export KG_ENABLE_VISUALIZATION="true"
```

### 3. Installation

```bash
# Install dependencies
pip install -r knowledge_graph/requirements.txt

# The system will auto-initialize when the main app starts
python app/main.py
```

### 4. Access the System

- **API Documentation**: http://localhost:8000/docs#/Knowledge-Graph
- **Graph Visualizations**: http://localhost:8000/api/v1/kg/visualization/create
- **System Overview**: http://localhost:8000/api/v1/kg/system/overview

## API Usage Examples

### Agent Discovery

```python
import httpx

# Find agents with specific capabilities
response = httpx.post("http://localhost:8000/api/v1/kg/agents/find-by-capability", 
                     json={
                         "capabilities": ["code_generation", "security_analysis"],
                         "min_match": 1,
                         "include_health": True
                     })

agents = response.json()["agents"]
```

### Natural Language Queries

```python
# Ask questions in natural language
response = httpx.post("http://localhost:8000/api/v1/kg/query/natural",
                     json={
                         "query": "What agents can do code generation and have good health?"
                     })

results = response.json()
```

### Service Dependencies

```python
# Analyze service dependencies
response = httpx.get("http://localhost:8000/api/v1/kg/services/api_service/dependencies")
dependencies = response.json()
```

### Create Visualizations

```python
# Generate interactive visualization
response = httpx.post("http://localhost:8000/api/v1/kg/visualization/create",
                     json={
                         "type": "agent_capabilities",
                         "output_format": "html"
                     })

viz_path = response.json()["file_path"]
```

## Schema Overview

### Node Types

- **Agent**: AI agents with capabilities and performance metrics
- **Service**: Backend services and microservices
- **Database**: Data storage systems (PostgreSQL, Redis, Neo4j)
- **Workflow**: Process flows and orchestration patterns
- **Capability**: Functional capabilities (code generation, security, etc.)
- **Model**: AI models and configurations
- **Document**: Documentation and knowledge artifacts

### Relationship Types

- **HAS_CAPABILITY**: Agent → Capability
- **DEPENDS_ON**: Service → Service
- **ORCHESTRATES**: Agent → Service/Workflow
- **COMMUNICATES_WITH**: Agent ↔ Agent
- **READS_FROM / WRITES_TO**: Service → Database
- **USES**: Workflow → Agent/Service

## Configuration

The system uses environment variables for configuration:

```python
from knowledge_graph.manager import KnowledgeGraphConfig

# From environment
config = KnowledgeGraphConfig.from_env()

# Or programmatically
config = KnowledgeGraphConfig({
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_username": "neo4j", 
    "neo4j_password": "password",
    "base_path": "/opt/sutazaiapp/backend",
    "enable_real_time_updates": True,
    "enable_reasoning": True,
    "enable_visualization": True
})
```

## Advanced Features

### Custom Reasoning Rules

```python
from knowledge_graph.reasoning_engine import ReasoningRule

# Define custom inference rule
rule = ReasoningRule(
    rule_id="detect_overloaded_agents",
    name="Detect Overloaded Agents",
    description="Find agents with high task load",
    condition_cypher="""
        MATCH (a:Agent)
        WHERE a.current_task_count > a.max_concurrent_tasks * 0.8
        RETURN a
    """,
    action_cypher="""
        MATCH (a:Agent {id: $agent_id})
        SET a.status = 'overloaded'
    """,
    priority=3
)

reasoning_engine.rule_engine.add_rule(rule)
```

### Custom Visualizations

```python
# Create custom visualization with Cypher query
viz_manager = get_knowledge_graph_manager().get_visualization_manager()

custom_html = await viz_manager.create_custom_view(
    cypher_query="""
        MATCH (a:Agent)-[:HAS_CAPABILITY]->(c:Capability)
        WHERE c.name = 'security_analysis'
        RETURN a, c
    """,
    title="Security Analysis Agents",
    output_file="security_agents.html"
)
```

### Real-time Updates

The system automatically monitors:
- **File changes**: Python code, configurations, requirements
- **Agent events**: Registration, health changes, performance updates
- **Service changes**: Deployments, configuration updates
- **Documentation**: README files, API documentation

## Performance & Scaling

### Optimization Features

- **Batch processing**: Bulk node/relationship creation
- **Query caching**: Intelligent query result caching
- **Index optimization**: Automatic index creation for common queries
- **Connection pooling**: Efficient Neo4j connection management

### Monitoring

```python
# Get system statistics
response = httpx.get("http://localhost:8000/api/v1/kg/system/stats")
stats = response.json()

print(f"Total nodes: {stats['graph_statistics']['total_nodes']}")
print(f"Total relationships: {stats['graph_statistics']['total_relationships']}")
print(f"Query performance: {stats['neo4j_stats']['avg_query_time']}ms")
```

## Troubleshooting

### Common Issues

**1. Neo4j Connection Failed**
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Check connection
curl http://localhost:7474/browser/
```

**2. Graph Build Fails**
```bash
# Check logs
tail -f knowledge_graph.log

# Manual rebuild
curl -X POST http://localhost:8000/api/v1/kg/system/rebuild
```

**3. Real-time Updates Not Working**
```bash
# Check file permissions
ls -la /opt/sutazaiapp/backend/

# Restart real-time updater
curl -X POST http://localhost:8000/api/v1/kg/system/sync
```

### Health Checks

```bash
# System health
curl http://localhost:8000/api/v1/kg/health

# Component status
curl http://localhost:8000/api/v1/kg/system/stats
```

## Development

### Adding New Node Types

1. Update `schema.py` with new node class
2. Add to `NodeType` enum
3. Update graph builder to extract new type
4. Add visualization support

### Custom Query Templates

```python
# Add to QueryTemplate class
CUSTOM_QUERY = """
    MATCH (n:YourNodeType)
    WHERE n.property = $value
    RETURN n
"""
```

### Testing

```bash
# Run tests (when available)
pytest knowledge_graph/tests/

# Manual testing
python -c "
from knowledge_graph.manager import initialize_knowledge_graph_system
import asyncio
asyncio.run(initialize_knowledge_graph_system())
"
```

## Integration

The Knowledge Graph System integrates with:

- **Agent Registry**: Real-time agent status updates
- **Service Discovery**: Automatic service relationship mapping  
- **Monitoring System**: Performance metrics and health data
- **Configuration Management**: Automatic config change detection
- **Documentation System**: Knowledge extraction from docs

## Contributing

1. Follow the existing code patterns
2. Add comprehensive logging
3. Include error handling
4. Update documentation
5. Add appropriate tests

## License

This module is part of the SutazAI platform and follows the same licensing terms.

---

For more information, see the [SutazAI System Guide](../SUTAZAI_SYSTEM_GUIDE.md) or check the API documentation at `/docs` when the server is running.