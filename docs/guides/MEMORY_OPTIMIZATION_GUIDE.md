# SutazAI Memory Optimization Guide

## 🚨 Immediate Fix for OOM Issues

If you're experiencing Out of Memory (OOM) kills like the ones you showed, run this immediately:

```bash
./fix-memory-issues.sh
```

This script will:
- Stop all containers to prevent further OOM kills
- Clean up Docker resources
- Add 8GB swap space
- Optimize system memory settings
- Start minimal services with strict memory limits

## 📊 Memory Optimization Summary

### Problem Analysis
Your system was experiencing:
- Frequent OOM kills on Ollama (6-8GB memory usage)
- Streamlit OOM kills (12GB+ memory usage)
- No swap space (causing aggressive OOM killer behavior)
- Multiple large models loaded simultaneously

### Solution Implemented

#### 1. **Docker Compose with Memory Limits** (`docker-compose-optimized.yml`)
- Strict memory limits for all containers
- Ollama limited to 4GB max
- Streamlit limited to 1GB max
- PostgreSQL optimized for 512MB
- Redis limited to 256MB with LRU eviction

#### 2. **Ollama Memory Management** (`scripts/ollama-startup.sh`)
- Single model loading (OLLAMA_MAX_LOADED_MODELS=1)
- Automatic model unloading after 5 minutes
- Memory-aware model selection (prefer smaller models)
- Continuous cleanup routine

#### 3. **System Optimization**
- 8GB swap space addition
- vm.overcommit_memory=1 (prevent aggressive OOM kills)
- vm.swappiness=10 (prefer RAM over swap)
- Increased file descriptor limits

#### 4. **Health Monitoring** (`scripts/health-monitor.py`)
- Real-time memory monitoring
- Automatic service recovery
- Memory pressure detection
- Emergency cleanup when memory > 95%

#### 5. **Model Management** (`scripts/model-manager.py`)
- Intelligent model loading based on available memory
- Automatic unloading of unused models
- Priority-based model selection
- Memory usage tracking

## 🚀 Deployment Options

### Option 1: Emergency Fix (Immediate)
```bash
./fix-memory-issues.sh
```
- Minimal resource usage
- Basic functionality
- Stable operation

### Option 2: Optimized Deployment (Recommended)
```bash
./deploy-optimized.sh
```
- Full feature set with memory optimization
- Comprehensive monitoring
- Auto-scaling and recovery

### Option 3: Full Production Deployment
```bash
./deploy.sh
```
- Complete AGI/ASI system
- All AI agents and services
- Maximum functionality (requires 32GB+ RAM)

## 🔧 Configuration Details

### Memory Allocation Strategy

| Service | Memory Limit | Purpose |
|---------|-------------|---------|
| Ollama | 4GB | AI model serving |
| Streamlit | 1GB | Web interface |
| PostgreSQL | 1GB | Database |
| Redis | 512MB | Caching |
| ChromaDB | 1GB | Vector database |
| Qdrant | 1GB | Vector database |
| Backend | 1GB | API services |

### Model Loading Strategy

| Available Memory | Models Loaded |
|-----------------|---------------|
| < 2GB | llama3.2:1b only |
| 2-4GB | llama3.2:1b + qwen2.5-coder:1.5b |
| 4-8GB | Priority + starcoder2:3b |
| > 8GB | All models as needed |

## 📈 Monitoring and Maintenance

### Real-time Monitoring
```bash
# System memory
free -h

# Docker container memory
docker stats

# Service health
docker-compose -f docker-compose-optimized.yml ps

# System monitor
python3 scripts/health-monitor.py
```

### Log Files
- `/var/log/sutazai-health.log` - Health monitoring
- `/var/log/sutazai-model-manager.log` - Model management
- `/var/log/sutazai-system.log` - System logging
- `/var/log/sutazai/` - Detailed metrics and reports

### Automated Tasks
- Health checks every 60 seconds
- Model optimization every 5 minutes
- System cleanup every 30 minutes
- Log rotation every 24 hours

## 🛠️ Troubleshooting

### High Memory Usage
1. Check running models: `curl http://localhost:11434/api/ps`
2. Unload unused models: `curl -X POST http://localhost:11434/api/generate -d '{"model":"MODEL_NAME","keep_alive":0}'`
3. Restart memory-intensive services: `docker-compose restart streamlit-frontend`

### OOM Kills Still Occurring
1. Increase swap space: `sudo fallocate -l 16G /swapfile2`
2. Reduce container limits in docker-compose-optimized.yml
3. Use emergency mode: `./fix-memory-issues.sh`

### Service Not Starting
1. Check memory availability: `free -h`
2. Check container logs: `docker-compose logs SERVICE_NAME`
3. Verify dependencies: `docker-compose ps`

## 🔄 Recovery Procedures

### Automatic Recovery
The health monitor automatically:
- Restarts failed services
- Unloads models on memory pressure
- Scales down non-critical services
- Performs emergency cleanup

### Manual Recovery
```bash
# Stop all services
docker-compose -f docker-compose-optimized.yml down

# Clean up resources
docker system prune -f

# Restart with memory fix
./fix-memory-issues.sh
```

## 📋 Best Practices

### Memory Management
1. **Monitor regularly**: Use `docker stats` and `free -h`
2. **Load models on demand**: Don't keep unused models in memory
3. **Use smaller models**: Prefer efficient models for production
4. **Scale gradually**: Add services incrementally

### System Maintenance
1. **Regular cleanup**: Run cleanup scripts weekly
2. **Log rotation**: Monitor log file sizes
3. **Update models**: Keep models optimized and current
4. **Backup configurations**: Save working configurations

### Performance Optimization
1. **CPU allocation**: Monitor CPU usage with `htop`
2. **Disk I/O**: Monitor with `iotop`
3. **Network**: Monitor with `iftop`
4. **Service dependencies**: Optimize startup order

## 🚀 Scaling Guidelines

### Horizontal Scaling (Multiple Instances)
- Use Docker Swarm or Kubernetes
- Load balance with Nginx
- Share state via Redis/PostgreSQL

### Vertical Scaling (More Resources)
- 32GB+ RAM for full AGI system
- NVMe SSD for model storage
- GPU for accelerated inference

### Hybrid Approach
- Core services on main server
- AI agents on separate instances
- Shared vector databases

## 📞 Support and Maintenance

### Automated Alerts
Configure webhooks in health monitor for:
- Memory usage > 90%
- Service failures
- Model loading issues
- System errors

### Manual Checks
Weekly checks:
- Memory usage trends
- Log file sizes
- Service performance
- Model efficiency

### Optimization Schedule
- Daily: Cleanup and health checks
- Weekly: Performance analysis
- Monthly: Configuration review
- Quarterly: System updates

## 🎯 Expected Performance

### With Optimizations
- Memory usage: 60-80% of available RAM
- Stable operation without OOM kills
- Response times: < 2 seconds for most operations
- Uptime: > 99.5%

### Resource Requirements
- **Minimum**: 16GB RAM, 4 CPU cores, 100GB disk
- **Recommended**: 32GB RAM, 8 CPU cores, 500GB SSD
- **Optimal**: 64GB RAM, 16 CPU cores, 1TB NVMe SSD

## ✅ Verification Steps

After deployment, verify:

1. **No OOM kills**: `dmesg | grep -i "killed process"`
2. **Memory within limits**: `docker stats`
3. **Services healthy**: All services showing "healthy" status
4. **Swap usage reasonable**: `free -h` shows < 50% swap usage
5. **Application responsive**: UI loads within 5 seconds

## 🔗 Quick Reference

### Key Commands
```bash
# Emergency fix
./fix-memory-issues.sh

# Optimized deployment
./deploy-optimized.sh

# Check memory
free -h && docker stats --no-stream

# View logs
docker-compose logs -f --tail=50

# Health check
curl http://localhost:8000/health

# Model status
curl http://localhost:11434/api/ps
```

### Important Files
- `docker-compose-optimized.yml` - Main deployment
- `scripts/health-monitor.py` - Health monitoring
- `scripts/model-manager.py` - Model management
- `fix-memory-issues.sh` - Emergency fix
- `deploy-optimized.sh` - Full deployment

This optimization guide should resolve your OOM issues and provide a stable, scalable SutazAI deployment.