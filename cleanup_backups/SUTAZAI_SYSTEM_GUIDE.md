# SutazAI System Guide

## Overview

SutazAI is a comprehensive, well-structured AI system designed for enterprise and research applications. It provides a unified platform for AI model management, agent orchestration, knowledge management, and neural processing.

## Architecture

### Core Components

1. **SutazAI Core (`sutazai_core.py`)**
   - Main system orchestrator
   - Component lifecycle management
   - Event system and monitoring
   - Resource management

2. **System Manager (`system_manager.py`)**
   - FastAPI integration layer
   - Request tracking and metrics
   - Configuration management
   - Component coordination

3. **Enhanced Main (`enhanced_main.py`)**
   - FastAPI application with integrated system
   - RESTful API endpoints
   - Security and middleware
   - Error handling

### System Features

- **Model Management**: Load, unload, and manage AI models
- **Agent Orchestration**: Create and manage AI agents
- **Knowledge Management**: Vector storage and semantic search
- **Neural Processing**: Biological modeling and neuromorphic computing
- **Security**: Ethical verification and content filtering
- **Monitoring**: System metrics and health checks
- **Web Learning**: Content scraping and knowledge extraction

## Configuration

### System Configuration (`config/system.json`)

```json
{
  "system_name": "SutazAI",
  "version": "1.0.0",
  "environment": "production",
  "directories": {
    "base_dir": "/opt/sutazaiapp/backend",
    "data_dir": "/opt/sutazaiapp/backend/data",
    "models_dir": "/opt/sutazaiapp/backend/data/models"
  },
  "features": {
    "enable_neural_processing": true,
    "enable_agent_orchestration": true,
    "enable_knowledge_management": true
  }
}
```

### Model Configuration (`config/models.json`)

```json
{
  "models": {
    "llama3-8b": {
      "name": "Llama 3 8B ChatQA (Ollama)",
      "type": "ollama",
      "framework": "langchain",
      "parameters": {
        "model": "llama3-chatqa",
        "temperature": 0.7,
        "num_ctx": 8192
      }
    }
  }
}
```

## API Endpoints

### System Management

- `GET /health` - System health check
- `GET /system/status` - Detailed system status
- `GET /system/metrics` - System metrics and statistics
- `GET /system/config` - System configuration
- `POST /system/config` - Update system configuration
- `POST /system/command` - Execute system commands

### Model Management

- `GET /models` - List available models
- `POST /models/{model_name}/load` - Load a model
- `POST /models/{model_name}/unload` - Unload a model

### Agent Management

- `GET /agents` - List available agents
- `POST /agents/{agent_type}/create` - Create agent instance
- `POST /chat` - Chat with agents

### Component Management

- `GET /components` - List all components
- `GET /components/{component_name}` - Get component status

## Usage Examples

### Starting the System

```python
from system_manager import initialize_system_manager, startup_system_manager

# Initialize system manager
system_manager = initialize_system_manager()

# Start the system
success = await startup_system_manager()
if success:
    print("SutazAI system started successfully")
```

### Using the API

```bash
# Check system health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Load a model
curl -X POST http://localhost:8000/models/llama3-8b/load

# Chat with an agent
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "agent": "llama3-8b"
  }'
```

### System Commands

```python
# Execute system commands
await system_manager.execute_command("model.list")
await system_manager.execute_command("agent.create", agent_type="assistant")
await system_manager.execute_command("system.health")
```

## Component Details

### Model Manager
- Manages AI model lifecycle
- Supports multiple frameworks (Ollama, HuggingFace, GPT4All)
- Handles model caching and optimization
- Provides model versioning

### Agent Framework
- Creates and manages AI agents
- Supports multiple agent types
- Handles agent communication and collaboration
- Provides task execution and monitoring

### Knowledge Engine
- Vector storage and retrieval
- Semantic search capabilities
- Document processing and indexing
- Knowledge graph construction

### Security Manager
- Ethical content verification
- Access control and authentication
- Audit logging and monitoring
- Content filtering and sanitization

## Monitoring and Metrics

### System Metrics
- CPU, memory, and disk usage
- Network I/O statistics
- Active models and agents
- Request rate and response times
- Error rates and uptime

### Health Checks
- Component status monitoring
- Resource usage validation
- Performance threshold checks
- Automated recovery procedures

## Development

### Running the System

```bash
# Start with enhanced main
python enhanced_main.py

# Or with uvicorn
uvicorn enhanced_main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables

```bash
export HOST=0.0.0.0
export PORT=8000
export DEBUG=false
export LOG_LEVEL=info
export ENABLE_DOCS=true
```

### Testing

```bash
# Run system tests
python -m pytest tests/

# Check system health
curl http://localhost:8000/health

# Verify components
curl http://localhost:8000/components
```

## Security

### Features
- Rate limiting and request throttling
- CORS configuration
- Content sanitization
- Ethical AI verification
- Access control and authentication

### Best Practices
- Use HTTPS in production
- Configure proper CORS origins
- Enable authentication for sensitive endpoints
- Monitor and audit system access
- Regular security updates

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "enhanced_main.py"]
```

### Production Configuration

```json
{
  "environment": "production",
  "debug_mode": false,
  "security": {
    "enable_security": true,
    "enable_audit": true,
    "enable_encryption": true
  },
  "monitoring": {
    "enable_monitoring": true,
    "metrics_interval": 300,
    "log_level": "WARNING"
  }
}
```

## Troubleshooting

### Common Issues

1. **System Won't Start**
   - Check configuration files
   - Verify directory permissions
   - Review system logs

2. **Model Loading Fails**
   - Ensure model files exist
   - Check memory availability
   - Verify model configuration

3. **Agent Creation Fails**
   - Check agent configuration
   - Verify required dependencies
   - Review agent framework logs

### Log Locations

- System logs: `/opt/sutazaiapp/backend/logs/`
- Component logs: Individual component log files
- Error logs: Centralized error logging

### Performance Optimization

- Adjust worker count based on CPU cores
- Configure memory limits appropriately
- Use model quantization for memory efficiency
- Enable caching for frequently used models

## Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure system: Edit `config/system.json`
4. Run tests: `python -m pytest`
5. Start development server: `python enhanced_main.py`

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Include comprehensive docstrings
- Write unit tests for new features
- Use async/await for I/O operations

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For support and questions:
- Check the documentation
- Review system logs
- Use the health check endpoints
- Contact the development team

---

*This guide provides a comprehensive overview of the SutazAI system. For detailed implementation information, refer to the source code and inline documentation.*