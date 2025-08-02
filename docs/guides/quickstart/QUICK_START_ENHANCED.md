# SutazAI Enhanced Coordinator v2.0 - Quick Start Guide

## 🚀 Quick Deployment (5 minutes)

### Option 1: Complete System with Coordinator (Recommended)
```bash
# Clone the repository
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp

# Deploy everything including Enhanced Coordinator
DEPLOY_BRAIN=true ./scripts/deploy_complete_system.sh
```

### Option 2: Deploy Coordinator Separately
```bash
# First, deploy main system
./scripts/deploy_complete_system.sh

# Then deploy Enhanced Coordinator
./scripts/deploy_coordinator_enhanced.sh
```

## 🧪 Verify Installation

### Quick Test
```bash
# Run comprehensive validation
./scripts/validate_and_optimize_system.sh
```

### Detailed Testing
```bash
# Test Enhanced Coordinator specifically
./scripts/test_enhanced_coordinator.sh
```

## 🎯 Key Access Points

### Main System
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Enhanced Coordinator
- **Coordinator API**: http://localhost:8888
- **JARVIS**: http://localhost:8026
- **Monitoring**: http://localhost:3001
- **Prometheus**: http://localhost:9091

## 💬 Quick Examples

### 1. Basic Coordinator Request
```bash
curl -X POST http://localhost:8888/process \
  -H 'Content-Type: application/json' \
  -d '{"input": "Create a Python function to calculate fibonacci numbers"}'
```

### 2. JARVIS Multi-Modal Request
```bash
curl -X POST http://localhost:8026/execute \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Analyze the current system status and optimize performance",
    "mode": "multi-modal",
    "system_command": true
  }'
```

### 3. Test Universal Learning
```bash
curl -X POST http://localhost:8888/process \
  -H 'Content-Type: application/json' \
  -d '{"input": "Demonstrate your learning capabilities by solving a novel problem"}'
```

## 📊 Monitor System Health

### Real-time Monitoring
1. **Grafana Dashboard**: http://localhost:3001
   - Username: `admin`
   - Password: `admin`

2. **Live Logs**:
   ```bash
   # View all logs
   tail -f /workspace/logs/*.log
   
   # View Coordinator-specific logs
   tail -f /workspace/coordinator/logs/*.log
   ```

3. **System Status**:
   ```bash
   # Check all services
   docker ps
   
   # Check Coordinator status
   curl http://localhost:8888/status
   ```

## 🔧 Common Operations

### Check Available Models
```bash
docker exec sutazai-ollama ollama list
```

### Pull Additional Models
```bash
docker exec sutazai-ollama ollama pull llama2:13b
docker exec sutazai-ollama ollama pull mistral:7b
```

### Restart Coordinator System
```bash
cd /workspace/coordinator
docker-compose -f docker-compose-enhanced.yml restart
```

### View Agent Status
```bash
curl http://localhost:8888/agents
```

## 🛠️ Configuration

### Basic Configuration
Edit `/workspace/coordinator/config/coordinator_enhanced_config.yaml`:
```yaml
# Adjust concurrent agents
agents:
  max_concurrent: 10  # Increase for more parallelism

# Change quality threshold
quality:
  min_score: 0.85  # Lower for faster but less accurate results

# Enable/disable self-improvement
self_improvement:
  auto_improve: true  # Set to false to disable
```

### Resource Limits
Edit `/workspace/coordinator/docker-compose-enhanced.yml`:
```yaml
services:
  coordinator-core:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
```

## 🚨 Troubleshooting

### Coordinator Not Starting
```bash
# Check logs
docker logs sutazai-coordinator-core

# Verify prerequisites
docker ps | grep -E "ollama|redis|postgresql|qdrant|chromadb"
```

### Agent Connection Issues
```bash
# Restart specific agent
docker restart sutazai-jarvis

# Check agent health
curl http://localhost:8026/health
```

### Memory Issues
```bash
# Check memory usage
docker stats

# Clean up resources
docker system prune -f --volumes
```

### Model Download Failures
```bash
# Manually pull models
docker exec -it sutazai-ollama bash
ollama pull tinyllama
```

## 📈 Performance Tips

1. **GPU Acceleration**: Ensure NVIDIA drivers and CUDA are installed
2. **Memory Optimization**: Allocate at least 32GB RAM to Docker
3. **Model Selection**: Use smaller models (7B) for faster responses
4. **Concurrent Limits**: Adjust `max_concurrent_agents` based on resources

## 🔐 Security Checklist

- [ ] Change default passwords in `.env`
- [ ] Secure `.env` file: `chmod 600 .env`
- [ ] Enable firewall for production
- [ ] Use HTTPS for external access
- [ ] Regular backups of PostgreSQL data

## 📚 Next Steps

1. **Explore the API Documentation**: http://localhost:8000/docs
2. **Read the Architecture Guide**: `/workspace/docs/ENHANCED_COORDINATOR_SUMMARY.md`
3. **Try Advanced Features**:
   - Test self-improvement capabilities
   - Experiment with multi-agent coordination
   - Monitor learning progress

## 🆘 Support

- **Logs**: `/workspace/logs/`
- **Reports**: `/workspace/reports/`
- **Documentation**: `/workspace/docs/`
- **GitHub Issues**: https://github.com/sutazai/sutazaiapp/issues

## 🎉 Congratulations!

You now have a fully operational automation system/advanced automation system with:
- ✅ Universal Learning Machine
- ✅ 30+ Specialized Agents
- ✅ Advanced ML/Deep Learning
- ✅ Self-Improvement Capabilities
- ✅ Multi-Modal Processing (JARVIS)
- ✅ Enterprise-Grade Monitoring

The system will continuously learn and improve. Check back regularly to see its progress!

---

**Pro Tip**: The Coordinator gets smarter with each interaction. Feed it diverse tasks to accelerate learning!