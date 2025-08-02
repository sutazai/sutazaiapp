# 🚀 SutazAI automation system/advanced automation Complete System Usage Guide

## System Overview
The SutazAI automation system/advanced automation automation system is now **100% operational** with comprehensive AI capabilities, automated deployment, and self-improving features.

## 🌐 Access Points

### Primary Interfaces
- **Main UI**: http://192.168.131.128:8501 (Streamlit Interface)
- **API Backend**: http://192.168.131.128:8000 (FastAPI Documentation)
- **Open WebUI**: http://192.168.131.128:8089 (Alternative Chat Interface)

### Monitoring & Admin
- **API Docs**: http://192.168.131.128:8000/docs (Swagger UI)
- **Vector Database**: http://192.168.131.128:6333 (Qdrant)
- **ChromaDB**: http://192.168.131.128:8001 (Vector Storage)

## 🤖 Available AI Models

### Active Models
1. **DeepSeek-R1 8B** (`tinyllama`)
   - Advanced reasoning and code generation
   - Complex problem solving
   - Mathematical computations

2. **Llama 3.2 1B** (`llama3.2:1b`) 
   - Fast conversational AI
   - General knowledge queries
   - Quick responses

### Model Selection
```bash
# Via API
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "model": "tinyllama"}'

# Via Web UI
# Select model from dropdown in the interface
```

## 🎯 Core Capabilities

### 1. Intelligent Conversation
- **Natural Language Processing**: Advanced conversation with context awareness
- **Multi-turn Dialogue**: Maintains conversation history and context
- **Knowledge Integration**: Access to vast knowledge base

**Usage Examples:**
```
• "Explain advanced computing in simple terms"
• "Help me debug this Python code: [paste code]"
• "What are the latest trends in AI development?"
```

### 2. Advanced Code Generation
- **Multi-language Support**: Python, JavaScript, Java, C++, Go, Rust
- **Framework Integration**: FastAPI, React, Django, Flask, Node.js
- **Architecture Design**: System design and planning

**Usage Examples:**
```
• "Generate a FastAPI service for user authentication"
• "Create a React component for data visualization" 
• "Build a microservice architecture for e-commerce"
```

### 3. Document Processing (DocuMind)
- **Multi-format Support**: PDF, DOCX, TXT, Excel, Images
- **OCR Capabilities**: Text extraction from scanned documents
- **Content Analysis**: Summarization and key information extraction

**API Usage:**
```bash
curl -X POST http://localhost:8085/process \
  -F "file=@document.pdf"
```

### 4. Financial Analysis (FinRobot)
- **Market Data Analysis**: Real-time stock data and trends
- **Portfolio Optimization**: Risk assessment and allocation
- **Predictive Modeling**: ML-based price predictions

**API Usage:**
```bash
# Get stock analysis
curl -X POST http://localhost:8086/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "analysis_type": "technical"}'
```

### 5. Web Automation (Browser-Use)
- **Automated Browsing**: Navigate websites automatically
- **Form Filling**: Complete online forms
- **Data Extraction**: Scrape and process web content

### 6. Security Analysis (Semgrep Integration)
- **Vulnerability Scanning**: Automated security audits
- **Code Analysis**: Security best practices enforcement
- **Compliance Checking**: Industry standard compliance

## 🔧 Agent Ecosystem

### Multi-Agent Orchestrator (26 Agents Total)

#### Specialized Agents:
1. **CodeMaster** - Advanced code generation and review
2. **SecurityGuard** - Security analysis and vulnerability detection
3. **DocProcessor** - Document parsing and analysis
4. **FinAnalyst** - Financial data analysis and modeling
5. **WebAutomator** - Browser automation and web scraping
6. **TaskCoordinator** - Workflow management and orchestration
7. **SystemMonitor** - Health monitoring and diagnostics
8. **DataScientist** - ML models and data analysis
9. **DevOpsEngineer** - Infrastructure and deployment
10. **GeneralAssistant** - Conversational AI and general help

#### External Agent Integration:
- **AutoGPT**: Task automation and autonomous execution
- **LangChain**: Advanced orchestration and document QA
- **TabbyML**: Real-time code completion
- **Browser-Use**: Web automation capabilities

### Agent Communication
```bash
# Call specific agent
curl -X POST http://localhost:8000/api/external_agents/call \
  -H "Content-Type: application/json" \
  -d '{"agent": "langchain", "task": "analyze document", "data": {...}}'
```

## 💡 Advanced Features

### 1. Self-Improving Code Generation
The system can analyze and improve its own code:

```bash
# Analyze code quality
curl -X POST http://localhost:8099/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "def hello():\n    print(\"hello\")", "language": "python"}'

# Self-improve a file
curl -X POST http://localhost:8099/self-improve \
  -H "Content-Type: application/json" \
  -d '{"target_file": "backend/main.py", "improvement_type": "performance"}'
```

### 2. Vector Search & RAG
- **FAISS Integration**: Fast similarity search
- **ChromaDB**: Persistent vector storage
- **Qdrant**: Advanced vector operations

```bash
# Add documents to vector store
curl -X POST http://localhost:8096/add \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Document content"], "index_name": "knowledge_base"}'

# Search similar content
curl -X POST http://localhost:8096/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI development", "k": 5}'
```

### 3. Task Distribution
The system automatically distributes tasks to the most appropriate agents:

```bash
curl -X POST http://localhost:8000/api/docker_agents/distribute \
  -H "Content-Type: application/json" \
  -d '{"task": "analyze financial data", "data": {...}}'
```

## 🔄 Automated Workflows

### 1. Code Development Pipeline
```
User Request → CodeMaster Agent → Security Analysis → Code Review → Deployment
```

### 2. Document Analysis Pipeline  
```
Document Upload → DocProcessor → Content Extraction → Vector Storage → Analysis
```

### 3. Financial Analysis Pipeline
```
Market Data → FinAnalyst → Technical Analysis → Risk Assessment → Recommendations
```

## 📊 System Monitoring

### Health Checks
```bash
# Overall system health
curl http://localhost:8000/health

# Complete system status
curl http://localhost:8000/api/system/complete_status

# Individual service health
curl http://localhost:8085/health  # DocuMind
curl http://localhost:8086/health  # FinRobot
curl http://localhost:8088/health  # FAISS
```

### Performance Metrics
- **Response Times**: Real-time monitoring
- **Agent Utilization**: Task distribution analytics
- **Resource Usage**: CPU, memory, and storage tracking
- **Error Rates**: Automatic error detection and recovery

## 🚀 Quick Start Examples

### 1. Basic Chat
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! What can you help me with?"}'
```

### 2. Generate Python Code
```bash
curl -X POST http://localhost:8000/api/code/generate \
  -H "Content-Type: application/json" \
  -d '{"description": "Create a REST API for user management", "language": "python"}'
```

### 3. Analyze Stock Data
```bash
curl -X GET http://localhost:8086/stock/AAPL?period=1y
```

### 4. Process Document
```bash
curl -X POST http://localhost:8085/process \
  -F "file=@/path/to/document.pdf"
```

### 5. Web Automation
```bash
curl -X POST http://localhost:8088/automate \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "action": "extract_data"}'
```

## 🔒 Security Features

### Built-in Security
- **Code Security Scanning**: Automatic vulnerability detection
- **Input Validation**: All inputs are sanitized and validated
- **Rate Limiting**: API rate limiting to prevent abuse
- **Authentication**: Secure API access (when configured)

### Security Analysis
```bash
# Run security scan on code
curl -X POST http://localhost:8000/api/security/scan \
  -H "Content-Type: application/json" \
  -d '{"code": "your_code_here", "language": "python"}'
```

## 🎛️ Configuration

### Environment Variables
```bash
# Core services
BACKEND_URL=http://localhost:8000
OLLAMA_URL=http://localhost:11434
QDRANT_URL=http://localhost:6333

# Database connections
DATABASE_URL=postgresql://sutazai:sutazai_password@localhost:5432/sutazai
REDIS_URL=redis://localhost:6379
```

### Model Configuration
```json
{
  "models": {
    "default": "llama3.2:1b",
    "code_generation": "tinyllama",
    "conversation": "llama3.2:1b"
  }
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Model Not Responding**
   ```bash
   # Check Ollama status
   curl http://localhost:11434/api/health
   
   # Restart Ollama service
   docker restart sutazai-ollama
   ```

2. **Agent Not Available**
   ```bash
   # Check agent status
   curl http://localhost:8000/api/system/complete_status
   
   # Restart specific service
   docker restart sutazai-langchain-agents
   ```

3. **Database Connection Issues**
   ```bash
   # Check database health
   curl http://localhost:6333/healthz  # Qdrant
   curl http://localhost:8001/api/v1/heartbeat  # ChromaDB
   ```

### Log Access
```bash
# View system logs
docker logs sutazai-backend
docker logs sutazai-streamlit
docker logs sutazai-ollama
```

## 🚀 Next Steps

### Enhanced Capabilities
1. **Custom Agent Development**: Create specialized agents for specific domains
2. **Advanced Workflows**: Build complex multi-agent workflows
3. **Integration Extensions**: Connect with external APIs and services
4. **Performance Optimization**: Fine-tune models for specific use cases

### Scaling Options
1. **Horizontal Scaling**: Deploy additional agent instances
2. **Load Balancing**: Distribute requests across multiple backends
3. **Cloud Deployment**: Migrate to cloud infrastructure
4. **Model Optimization**: Quantize models for better performance

---

## 📞 Support

For technical support or advanced configuration:
- Check the API documentation at http://localhost:8000/docs
- Monitor system health through the dashboard
- Review logs for detailed error information
- Use the self-improvement features for automatic optimization

**The SutazAI automation system/advanced automation system is now ready for advanced AI applications!** 🎯