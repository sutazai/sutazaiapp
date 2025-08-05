# SutazAI External AI Services Integration Complete

## Overview

The integration of external AI services into the SutazAI distributed architecture has been successfully completed. This document provides a comprehensive overview of the integrated services, their configurations, and usage guidelines.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway (Kong)                        │
│                    Load Balancer & Router                        │
└─────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼────────┐      ┌────────▼────────┐      ┌────────▼────────┐
│ Vector DBs     │      │ AI Frameworks   │      │ Agent Systems   │
├────────────────┤      ├─────────────────┤      ├─────────────────┤
│ • ChromaDB     │      │ • PyTorch       │      │ • Letta         │
│ • FAISS        │      │ • TensorFlow    │      │ • AutoGPT       │
│ • Qdrant       │      │ • JAX           │      │ • LocalAGI      │
└────────────────┘      └─────────────────┘      │ • TabbyML       │
                                                  └─────────────────┘
        │                         │                         │
┌───────▼────────┐      ┌────────▼────────┐      ┌────────▼────────┐
│ Workflow Tools │      │ Specialized     │      │ Service         │
├────────────────┤      │ Tools           │      │ Registry        │
│ • LangFlow     │      ├─────────────────┤      ├─────────────────┤
│ • Dify         │      │ • FinRobot      │      │ • Consul        │
│ • FlowiseAI    │      │ • Documind      │      │ • Health Checks │
│ • n8n          │      │ • GPT-Engineer  │      │ • Monitoring    │
└────────────────┘      │ • Aider         │      └─────────────────┘
                        │ • Continue      │
                        │ • Sweep         │
                        │ • Pydantic AI   │
                        │ • Mem0          │
                        └─────────────────┘
```

## Integrated Services

### 1. Vector Databases

#### ChromaDB
- **Purpose**: Persistent vector storage with metadata filtering
- **Endpoint**: `http://chromadb:8000`
- **Features**: Multi-collection support, metadata filtering, persistent storage
- **Use Cases**: Document embeddings, semantic search, RAG applications

#### FAISS
- **Purpose**: High-performance similarity search
- **Endpoint**: Local filesystem access
- **Features**: GPU acceleration, billion-scale indexing, multiple index types
- **Use Cases**: Large-scale similarity search, real-time recommendations

#### Qdrant
- **Purpose**: Advanced vector database with filtering
- **Endpoint**: `http://qdrant:6333`
- **Features**: Payload storage, advanced filtering, clustering
- **Use Cases**: Complex queries, filtered search, recommendation systems

### 2. AI Frameworks

#### PyTorch
- **Purpose**: Deep learning model training and inference
- **Features**: GPU support, automatic differentiation, JIT compilation
- **Use Cases**: Neural network training, model deployment, research

#### TensorFlow
- **Purpose**: Production-ready ML framework
- **Features**: TFLite conversion, distributed training, Tensorboard
- **Use Cases**: Production models, edge deployment, large-scale training

### 3. Agent Systems

#### Letta (MemGPT)
- **Purpose**: Autonomous agents with long-term memory
- **Endpoint**: `http://letta:8283`
- **Features**: Persistent memory, contextual conversations, task execution
- **Use Cases**: Personal assistants, complex dialogues, memory-based tasks

#### AutoGPT
- **Purpose**: Goal-oriented autonomous task execution
- **Endpoint**: `http://autogpt:8000`
- **Features**: Web browsing, file operations, plugin system
- **Use Cases**: Automated research, task automation, goal achievement

#### LocalAGI
- **Purpose**: Local AI agent execution
- **Endpoint**: `http://localagi:8080`
- **Features**: Tool integration, local model support
- **Use Cases**: Privacy-focused agents, offline capabilities

#### TabbyML
- **Purpose**: AI-powered code completion
- **Endpoint**: `http://tabbyml:8080`
- **Features**: Code generation, multiple language support
- **Use Cases**: Development assistance, code completion

### 4. Workflow Tools

#### LangFlow
- **Purpose**: Visual workflow builder for LLM applications
- **Endpoint**: `http://langflow:7860`
- **Features**: Drag-and-drop interface, component library, API generation
- **Use Cases**: Rapid prototyping, visual workflow design

#### Dify
- **Purpose**: AI application development platform
- **Endpoint**: `http://dify-api:5001`
- **Features**: App builder, dataset management, team collaboration
- **Use Cases**: AI app development, prompt engineering, knowledge management

#### FlowiseAI
- **Purpose**: Low-code AI flow builder
- **Endpoint**: `http://flowise:3000`
- **Features**: Visual builder, chatflow creation, tool integration
- **Use Cases**: Chatbot creation, workflow automation

#### n8n
- **Purpose**: Workflow automation platform
- **Endpoint**: `http://n8n:5678`
- **Features**: Node-based workflows, webhook support, extensive integrations
- **Use Cases**: Process automation, data pipelines, integration workflows

### 5. Specialized Tools

#### FinRobot
- **Purpose**: Financial analysis and trading
- **Endpoint**: `http://finrobot:8000`
- **Features**: Market data analysis, portfolio optimization, backtesting
- **Use Cases**: Trading strategies, financial analysis, risk management

#### Documind
- **Purpose**: Document processing and analysis
- **Endpoint**: `http://documind:8080`
- **Features**: OCR, NLP analysis, document extraction
- **Use Cases**: Document processing, data extraction, text analysis

#### GPT-Engineer
- **Purpose**: AI-powered code generation
- **Endpoint**: `http://gpt-engineer:8000`
- **Features**: Full project generation, iterative development
- **Use Cases**: Rapid prototyping, code generation, project scaffolding

#### Additional Tools
- **Aider**: AI pair programming assistant
- **Continue**: AI code completion and chat
- **Sweep**: AI-powered GitHub issue resolver
- **Pydantic AI**: Type-safe AI interactions
- **Mem0**: Intelligent memory layer for AI apps

## Service Configuration

All services are configured through the centralized configuration file:
```
/opt/sutazaiapp/config/services.yaml
```

### Configuration Structure
```yaml
services:
  <category>:
    <service_name>:
      enabled: true/false
      adapter: "path.to.adapter.class"
      config:
        # Service-specific configuration
      resources:
        cpu: "2"
        memory: "4Gi"
        gpu: "optional/required"
      health_check:
        endpoint: "/health"
        interval: 30
```

## Deployment

### Deploy All Services
```bash
./scripts/deploy-ai-services.sh
```

### Deploy Specific Category
```bash
./scripts/deploy-ai-services.sh --category vector_databases
```

### Deploy Single Service
```bash
./scripts/deploy-ai-services.sh --service chromadb
```

### Dry Run Mode
```bash
./scripts/deploy-ai-services.sh --dry-run
```

## API Gateway Routes

All services are accessible through the unified API gateway:

- **Vector Databases**: `/api/v1/vectors/{service_name}`
- **AI Frameworks**: `/api/v1/ml/{service_name}`
- **Agent Systems**: `/api/v1/agents/{service_name}`
- **Workflow Tools**: `/api/v1/workflows/{service_name}`
- **Specialized Tools**: `/api/v1/tools/{service_name}`

## Service Discovery

Services are automatically registered with Consul for:
- Health monitoring
- Load balancing
- Service discovery
- Failover handling

## Resource Management

### GPU Allocation
- Fair-share strategy with 1.2x oversubscription
- Priority-based allocation for critical services
- Automatic GPU detection and assignment

### Memory Management
- Enforced memory limits with swap accounting
- Container-level resource isolation
- Automatic OOM prevention

### CPU Allocation
- Base share: 1024
- Priority multiplier: 2x for critical services
- Dynamic scaling based on load

## Monitoring and Health

### Health Checks
- Automatic health monitoring every 30 seconds
- Service-specific health endpoints
- Failure threshold: 3 consecutive failures

### Metrics Collection
- Prometheus metrics on port 9090
- Service-specific metrics exposed at `/metrics`
- Grafana dashboards for visualization

### Distributed Tracing
- Jaeger integration for request tracing
- End-to-end visibility across services
- Performance bottleneck identification

## Usage Examples

### Vector Search with ChromaDB
```python
from services.adapters.vector_db.chromadb_adapter import ChromaDBAdapter

adapter = ChromaDBAdapter(config)
await adapter.initialize()

# Add vectors
result = await adapter.add_vectors(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["doc1", "doc2"],
    metadatas=[{"type": "article"}, {"type": "blog"}]
)

# Search
results = await adapter.search_vectors(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5
)
```

### PyTorch Model Inference
```python
from services.adapters.ai_frameworks.pytorch_adapter import PyTorchAdapter

adapter = PyTorchAdapter(config)
await adapter.initialize()

# Load model
await adapter.load_model("my_model", "/models/model.pt")

# Run inference
result = await adapter.inference(
    model_name="my_model",
    input_data=input_tensor
)
```

### Create Letta Agent
```python
from services.adapters.agent_systems.letta_adapter import LettaAdapter

adapter = LettaAdapter(config)
await adapter.initialize()

# Create agent
agent = await adapter.create_agent(
    agent_id="assistant-1",
    persona="helpful assistant",
    human="user"
)

# Send message
response = await adapter.send_message(
    agent_id="assistant-1",
    message="What can you help me with?"
)
```

## Best Practices

1. **Service Selection**
   - Choose the right service for your use case
   - Consider resource requirements
   - Evaluate latency requirements

2. **Resource Optimization**
   - Share resources between compatible services
   - Use GPU only when necessary
   - Monitor resource usage regularly

3. **Error Handling**
   - Implement retry logic with exponential backoff
   - Use circuit breakers for failing services
   - Log errors comprehensively

4. **Security**
   - Use service-to-service authentication
   - Encrypt sensitive data in transit
   - Regularly update service dependencies

## Troubleshooting

### Service Not Starting
1. Check logs: `docker logs sutazai-<service-name>`
2. Verify configuration in `services.yaml`
3. Ensure required resources are available
4. Check network connectivity

### Performance Issues
1. Monitor resource usage with `docker stats`
2. Check service metrics in Grafana
3. Review distributed traces in Jaeger
4. Optimize configuration parameters

### Integration Errors
1. Verify API endpoints are correct
2. Check authentication credentials
3. Review adapter implementation
4. Test with minimal configuration

## Future Enhancements

1. **Additional Services**
   - More specialized AI tools
   - Enhanced workflow capabilities
   - Advanced monitoring solutions

2. **Performance Optimizations**
   - Service mesh implementation
   - Advanced caching strategies
   - Predictive scaling

3. **Enhanced Security**
   - Zero-trust networking
   - Advanced encryption
   - Compliance automation

## Support and Maintenance

- **Documentation**: `/opt/sutazaiapp/docs/`
- **Logs**: `/opt/sutazaiapp/logs/`
- **Monitoring**: http://localhost:3000 (Grafana)
- **Service Registry**: http://localhost:8500 (Consul)

For issues or questions, consult the service-specific documentation or raise an issue in the project repository.

---

*Integration completed on: 2025-08-04*
*Version: 1.0.0*