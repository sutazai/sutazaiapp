# SutazAI AGI/ASI System - Quick Start Guide

## 🚀 System is Ready!

Your SutazAI AGI/ASI Autonomous System is now operational and ready for use.

## 📍 Access Points

### Primary Interface
**Web UI**: http://localhost:8501
- Chat with AI models
- Monitor system status
- Manage agents and workflows

### API Access
**Base URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🤖 Available AI Models

1. **qwen2.5:3b** - Advanced 3.1B parameter model
2. **llama3.2:1b** - Fast 1.2B parameter model

## 🎯 Quick Start Steps

### 1. Open the Web Interface
Navigate to http://localhost:8501 in your browser

### 2. Start Chatting
- Select a model from the dropdown
- Type your message
- Click "Send" or press Enter

### 3. Try These Examples
```
"Explain quantum computing in simple terms"
"Write a Python function to calculate fibonacci numbers"
"What are the benefits of artificial general intelligence?"
"Help me understand machine learning concepts"
```

## 🛠️ Common Commands

### Check System Status
```bash
docker ps | grep sutazai
```

### View Logs
```bash
# Backend logs
ps aux | grep intelligent_backend

# Ollama logs
docker logs sutazai-ollama --tail 50
```

### List Available Models
```bash
docker exec sutazai-ollama ollama list
```

### Test API
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "model": "llama3.2:1b"}'
```

## 📊 System Components

### Running Services
- ✅ Backend API (Port 8000)
- ✅ Streamlit UI (Port 8501)
- ✅ Ollama AI Server (Port 11434)
- ✅ PostgreSQL Database (Port 5432)
- ✅ Redis Cache (Port 6379)
- ⚠️ Vector Stores (Functional but showing warnings)

### System Capabilities
- Multi-model AI chat
- Agent orchestration
- Knowledge management
- Self-improvement features
- 100% local execution (no external APIs)

## 🔧 Troubleshooting

### If models are slow to respond:
- First requests may take time as models load into memory
- Subsequent requests will be faster

### If UI is not accessible:
```bash
# Check if service is running
ps aux | grep streamlit

# Restart if needed
docker restart sutazai-streamlit
```

### To pull additional models:
```bash
docker exec sutazai-ollama ollama pull <model-name>
# Examples: codellama:7b, mistral:7b, phi:2.7b
```

## 📈 Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Try different models**: Each has unique capabilities
3. **Monitor performance**: Check logs and system resources
4. **Customize**: Modify configurations as needed

## 🎉 Congratulations!

Your SutazAI AGI/ASI system is fully operational. Enjoy exploring the capabilities of your local AI system!

For support or questions, refer to the documentation or system logs.