# SutazAI Backend

A comprehensive AGI/ASI backend system built with FastAPI, integrating multiple AI services, vector databases, and intelligent agents.

## Architecture Overview

### Core Components

- **FastAPI Backend**: High-performance async API server
- **Multiple AI Services**: Ollama, DeepSeek, Llama, CodeLlama, Mistral, and more
- **Vector Databases**: ChromaDB, Qdrant, FAISS for embeddings
- **Agent Orchestration**: AutoGPT, LocalAGI, TabbyML, AgentZero, BigAGI
- **Document Processing**: Documind for intelligent document analysis
- **Code Generation**: GPT-Engineer, Aider for automated coding
- **Web Automation**: Browser-Use, Skyvern for web interactions
- **Financial Analysis**: FinRobot for financial data processing
- **ML Frameworks**: PyTorch, TensorFlow, JAX integration

### Key Features

- 🤖 **Multi-Agent Orchestration**: Coordinate multiple AI agents
- 🧠 **Model Management**: Load and manage various AI models
- 📊 **Vector Storage**: Efficient similarity search and embeddings
- 📄 **Document Processing**: Extract and analyze document content
- 💻 **Code Generation**: Automated code creation and improvement
- 🌐 **Web Automation**: Intelligent web scraping and interaction
- 📈 **Financial Analysis**: Advanced financial data processing
- 🔍 **Monitoring**: Comprehensive health checks and metrics
- 🔒 **Security**: JWT authentication and secure API endpoints

## Project Structure

```
backend/
├── main_complete.py          # Main application entry point
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── .env                    # Environment variables
├── core/                   # Core system modules
│   ├── config.py           # Configuration management
│   ├── database.py         # Database connections
│   ├── cache.py            # Redis caching
│   ├── security.py         # Authentication & security
│   ├── monitoring.py       # Metrics and health checks
│   └── logging_config.py   # Logging configuration
├── services/               # Business logic services
│   ├── agent_orchestrator.py
│   ├── model_manager.py
│   ├── vector_store.py
│   ├── document_processor.py
│   ├── code_generator.py
│   ├── web_automation.py
│   ├── financial_analyzer.py
│   ├── workflow_engine.py
│   └── backup_manager.py
└── api/                    # API endpoints
    └── v1/
        ├── health.py
        ├── agents.py
        ├── models.py
        ├── documents.py
        ├── chat.py
        ├── workflows.py
        └── admin.py
```

## Quick Start

### 1. Environment Setup

```bash
# Create environment file
python3 setup_env.py

# Test structure
python3 test_structure.py
```

### 2. Docker Deployment

```bash
# Start all services
docker-compose -f ../docker-compose-complete.yml up -d

# Or start minimal services
docker-compose -f ../docker-compose.yml up -d
```

### 3. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend
python main_complete.py
```

## API Endpoints

### Health & Status
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed system health
- `GET /status` - Comprehensive system status

### Agents
- `GET /api/v1/agents` - List active agents
- `POST /api/v1/agents/create` - Create new agent
- `GET /api/v1/agents/{agent_id}/status` - Get agent status
- `POST /api/v1/agents/{agent_id}/execute` - Execute task
- `DELETE /api/v1/agents/{agent_id}` - Stop agent

### Models
- `GET /api/v1/models` - List available models
- `GET /api/v1/models/loaded` - List loaded models
- `POST /api/v1/models/load` - Load model
- `POST /api/v1/models/generate` - Generate text
- `POST /api/v1/models/chat` - Chat completion
- `DELETE /api/v1/models/{model_id}` - Unload model

### Documents
- `GET /api/v1/documents` - List documents
- `POST /api/v1/documents/upload` - Upload document
- `POST /api/v1/documents/process` - Process document
- `GET /api/v1/documents/{doc_id}` - Get document
- `GET /api/v1/documents/{doc_id}/text` - Get extracted text

### Chat
- `POST /api/v1/chat` - Chat with AI
- `GET /api/v1/chat/models` - Available chat models
- `GET /api/v1/chat/history` - Chat history

### Workflows
- `GET /api/v1/workflows` - List workflows
- `POST /api/v1/workflows/create` - Create workflow
- `POST /api/v1/workflows/{id}/execute` - Execute workflow
- `GET /api/v1/workflows/{id}/status` - Workflow status

### Admin
- `GET /api/v1/admin/system/status` - System status
- `GET /api/v1/admin/system/metrics` - System metrics
- `POST /api/v1/admin/system/restart` - Restart services
- `GET /api/v1/admin/logs` - System logs

## Configuration

### Environment Variables

```bash
# Core Settings
DEBUG_MODE=true
SECRET_KEY=your_secret_key_32_chars_minimum
JWT_SECRET=your_jwt_secret_32_chars_minimum

# Database URLs
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://localhost:6379
MONGODB_URL=mongodb://user:pass@host:27017/db

# Vector Databases
CHROMADB_URL=http://localhost:8001
QDRANT_URL=http://localhost:6333
FAISS_URL=http://localhost:8088

# AI Services
OLLAMA_URL=http://localhost:11434
AUTOGPT_URL=http://localhost:8010
LOCALAGI_URL=http://localhost:8011
# ... (see .env for full list)
```

## Service Integration

### Ollama Models
- DeepSeek-Coder (6.7B, 33B)
- Llama 2 (7B, 13B)
- CodeLlama (7B, 13B)
- Mistral (7B)
- Phi (2.7B)
- Qwen (7B)
- Gemma (7B)

### Vector Databases
- **ChromaDB**: Document embeddings and similarity search
- **Qdrant**: High-performance vector search with filtering
- **FAISS**: CPU-optimized similarity search

### AI Agents
- **AutoGPT**: Autonomous reasoning and task execution
- **LocalAGI**: Local AI inference and management
- **TabbyML**: Code completion and programming assistance
- **AgentZero**: Task automation and workflow execution
- **BigAGI**: Multi-modal AI capabilities

## Development

### Testing

```bash
# Test structure
python3 test_structure.py

# Test startup (requires dependencies)
python3 start.py
```

### Debugging

```bash
# Enable debug mode
export DEBUG_MODE=true

# Check logs
tail -f /logs/sutazai.log

# Monitor health
curl http://localhost:8000/health/detailed
```

## Docker Services

The system integrates with 30+ Docker services:

- **Databases**: PostgreSQL, Redis, MongoDB
- **Vector Stores**: ChromaDB, Qdrant, FAISS
- **AI Models**: Ollama, PyTorch, TensorFlow, JAX
- **Agents**: AutoGPT, LocalAGI, TabbyML, etc.
- **Tools**: Browser-Use, Skyvern, Documind, etc.
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Web**: Nginx, Open-WebUI

## Security

- JWT-based authentication
- API key management
- Input sanitization
- Rate limiting
- CSRF protection
- Secure file uploads

## Monitoring

- Health checks for all services
- System metrics collection
- Performance monitoring
- Error tracking
- Audit logging

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Test with `python3 test_structure.py`
5. Ensure all services integrate properly

## License

This project is part of the SutazAI AGI/ASI system.