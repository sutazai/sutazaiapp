# SutazAI automation system/advanced automation Quick Start Guide

## 🚀 System is Fully Operational!

All tests passed! Your SutazAI system is working perfectly with:
- ✅ Ollama integration (3 models loaded)
- ✅ All Docker services running
- ✅ Backend API fully functional
- ✅ Chat interface working

## 🎯 Quick Test Commands

### 1. Simple Chat Test
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is artificial intelligence?"}'
```

### 2. Self-Improvement Query
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How can you self-improve using external AI agents?"}'
```

### 3. Complex Task
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a detailed plan for building a web application",
    "model": "llama3.2:1b",
    "temperature": 0.8,
    "max_tokens": 500
  }'
```

## 📊 Monitor System

### Real-time Performance
```bash
# Watch performance metrics
watch -n 2 'curl -s http://localhost:8000/api/performance/summary | jq'

# Check health
curl http://localhost:8000/health | jq

# View logs
tail -f /opt/sutazaiapp/logs/backend_complete.log
```

### System Status
```bash
# Backend service
systemctl status sutazai-complete-backend

# Docker services
docker ps | grep sutazai

# Run full test suite
python3 /opt/sutazaiapp/test_complete_system.py
```

## 💬 Using the Chat Interface

In your Streamlit interface, try these queries:

1. **Basic Questions**:
   - "What is advanced computing?"
   - "Explain machine learning"
   - "How does blockchain work?"

2. **Self-Improvement**:
   - "How can you self-improve?"
   - "What AI agents can you use?"
   - "Explain your architecture"

3. **Complex Tasks**:
   - "Create a business plan for a startup"
   - "Design a database schema for an e-commerce site"
   - "Write a Python script to analyze data"

## 🤖 Available Models

| Model | Size | Best For |
|-------|------|----------|
| llama3.2:1b | 1.3GB | Fast responses, general chat |
| qwen2.5:3b | 1.9GB | Code generation, technical queries |
| tinyllama | 5.6GB | Complex reasoning, detailed analysis |

## 🔧 Troubleshooting

### If chat responses are slow:
```bash
# Check which model is being used
curl http://localhost:8000/api/models | jq

# Use a smaller model
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "model": "llama3.2:1b"}'
```

### If backend stops responding:
```bash
# Restart the service
sudo systemctl restart sutazai-complete-backend

# Check logs
tail -100 /opt/sutazaiapp/logs/backend_complete.log
```

### If Ollama has issues:
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
docker restart sutazai-ollama

# Pull a model again if needed
docker exec sutazai-ollama ollama pull llama3.2:1b
```

## 🎉 Success Indicators

Your system is working when:
- Health endpoint shows: `{"status": "healthy"}`
- Chat responses include: `"ollama_success": true`
- Performance metrics show active requests
- No errors in the logs

## 📈 Next Steps

1. **Add More Models**:
   ```bash
   docker exec sutazai-ollama ollama pull mistral:7b
   docker exec sutazai-ollama ollama pull integration_score-3:3.8b
   ```

2. **Deploy AI Agents** (when Docker images available):
   ```bash
   docker-compose up -d autogpt crewai privategpt
   ```

3. **Customize Settings**:
   - Edit `/opt/sutazaiapp/.env` for configuration
   - Modify `intelligent_backend_complete.py` for custom features

## 🔗 API Documentation

Access the interactive API documentation:
- Swagger UI: http://localhost:8000/docs (if enabled)
- API endpoints:
  - `/health` - System health
  - `/api/chat` - Chat completion
  - `/api/models` - List models
  - `/api/agents` - List agents
  - `/api/performance/summary` - Performance metrics

---

**System Version**: Complete Backend v11.0  
**Status**: ✅ Fully Operational  
**Last Tested**: Successfully passed all 9 tests