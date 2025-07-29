# SutazAI Enterprise AGI/ASI System - Complete Deployment Guide

## 🚀 Enhanced WSL2 2025 Edition

This deployment script has been completely rewritten to fix all critical issues identified in the deployment logs and implement the most advanced 2025 best practices for WSL2, Docker, and AI system deployment.

## 🛠️ Key Improvements

### 1. **WSL2-Specific Fixes**
- ✅ Advanced Docker daemon startup recovery for WSL2
- ✅ Multiple DNS resolution methods (systemd-resolved + fallback)
- ✅ Network optimization with sysctl settings
- ✅ Detection and suggestion of experimental WSL2 features
- ✅ Automatic detection of WSL2 environment

### 2. **Docker Daemon Intelligence**
- ✅ Smart Docker daemon recovery with multiple strategies
- ✅ Direct dockerd startup for WSL2 environments
- ✅ Socket permission fixes
- ✅ Optimized daemon.json configuration
- ✅ DNS server configuration for containers

### 3. **Enhanced Error Handling**
- ✅ Exponential backoff retry logic
- ✅ Comprehensive debug logging
- ✅ Graceful fallback mechanisms
- ✅ Intelligent auto-correction system
- ✅ Detailed error reporting

### 4. **GPU/CPU Detection**
- ✅ Automatic GPU capability detection
- ✅ CPU-only fallback with optimizations
- ✅ Dynamic compose file selection
- ✅ Environment variable configuration
- ✅ Clear warnings for unsupported features

### 5. **Network Resilience**
- ✅ Package installation with retry logic
- ✅ Ubuntu 24.04+ compatibility (PEP 668)
- ✅ Multiple DNS servers configuration
- ✅ Network connectivity verification
- ✅ Port conflict resolution

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10/11 with WSL2 enabled
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: Minimum 50GB free space
- **CPU**: 4+ cores recommended
- **Network**: Internet connection for package downloads

### Software Requirements
- WSL2 installed and configured
- Ubuntu 20.04+ or compatible distro
- Docker permissions (user in docker group or root)

## 🚀 Quick Start

1. **Make the script executable:**
   ```bash
   chmod +x deploy_complete_system.sh
   ```

2. **Run the deployment:**
   ```bash
   sudo ./deploy_complete_system.sh
   ```

3. **Monitor the deployment:**
   The script provides real-time progress indicators and detailed logging.

## 🎯 Advanced WSL2 Configuration (Optional)

For optimal performance, create `C:\Users\<YourUsername>\.wslconfig`:

```ini
[experimental]
# Network improvements
networkingMode=mirrored    # Better network compatibility
dnsTunneling=true          # Improved DNS resolution
firewall=true              # Windows firewall integration
autoProxy=true             # Automatic proxy configuration

# Performance improvements
autoMemoryReclaim=gradual  # Memory optimization
sparseVhd=true            # Disk space optimization

[wsl2]
# Resource limits (adjust based on your system)
memory=16GB
processors=8
swap=8GB
```

After creating this file, restart WSL:
```powershell
wsl --shutdown
```

## 🔧 Script Commands

```bash
# Deploy the complete system (default)
./deploy_complete_system.sh

# Stop all services
./deploy_complete_system.sh stop

# Restart the system
./deploy_complete_system.sh restart

# Check system status
./deploy_complete_system.sh status

# View logs
./deploy_complete_system.sh logs

# Run health checks only
./deploy_complete_system.sh health

# Show help
./deploy_complete_system.sh help
```

## 📊 Deployment Phases

1. **Network Infrastructure Setup**
   - WSL2 detection and diagnostics
   - DNS resolution configuration
   - Docker daemon optimization

2. **Package Installation**
   - Critical system packages
   - Python environment setup
   - Optional development tools

3. **GPU Capability Detection**
   - NVIDIA/AMD GPU detection
   - CPU-only fallback
   - Environment optimization

4. **Port Conflict Resolution**
   - Automatic port scanning
   - Dynamic port remapping
   - Compose override generation

5. **Environment Configuration**
   - Secure password generation
   - Environment variable setup
   - Configuration file creation

6. **Directory Structure Setup**
   - Complete directory tree
   - Permission configuration
   - .gitkeep file creation

7. **Core Infrastructure**
   - PostgreSQL database
   - Redis cache
   - Neo4j graph database

8. **Vector Databases**
   - ChromaDB
   - Qdrant
   - FAISS

9. **AI Model Management**
   - Ollama server
   - Model downloads
   - LiteLLM proxy

10. **Backend Services**
    - Enterprise API
    - Health checks
    - API testing

11. **Frontend Services**
    - Web interface
    - Service validation

12. **AI Agent Ecosystem**
    - 50+ AI agents
    - Core agents deployment
    - Specialized services

13. **Monitoring Stack**
    - Prometheus
    - Grafana
    - Loki
    - Promtail

14. **System Initialization**
    - Knowledge graph setup
    - Agent registration
    - Service integration

15. **Integration Testing**
    - API connectivity
    - Database testing
    - Service validation

16. **Comprehensive Health Check**
    - All service endpoints
    - Container status
    - System metrics

## 🐛 Troubleshooting

### Docker Daemon Issues
If Docker fails to start:
1. The script will automatically attempt recovery
2. Check logs: `cat logs/deployment_*.log`
3. Manual start: `sudo dockerd &`

### DNS Resolution Problems
If you can't reach the internet:
1. Check `/etc/resolv.conf` has valid nameservers
2. Try: `ping 8.8.8.8` (should work even if DNS fails)
3. Restart networking: `sudo systemctl restart systemd-resolved`

### Port Conflicts
The script automatically handles port conflicts by:
1. Detecting occupied ports
2. Creating override configurations
3. Remapping to alternative ports

### Memory Issues
If running low on memory:
1. Enable WSL2 memory reclaim (see Advanced Configuration)
2. Reduce concurrent agents in `.env`
3. Monitor with: `docker system df`

## 📝 Important Files

- **Deployment Log**: `logs/deployment_YYYYMMDD_HHMMSS.log`
- **Debug Log**: `logs/deployment_debug_YYYYMMDD_HHMMSS.log`
- **Environment Config**: `.env`
- **Docker Compose**: `docker-compose.yml`
- **Port Overrides**: `docker-compose.port-optimized.yml` (if needed)

## 🔒 Security Notes

- All passwords are randomly generated
- Environment file has restricted permissions (600)
- Services are configured for local access only
- No external API keys required for base functionality

## 📞 Support

For issues or questions:
1. Check the deployment logs first
2. Review the troubleshooting section
3. File issues on the GitHub repository
4. Join the community Discord/Slack

## 🎉 Success Indicators

A successful deployment will show:
- ✅ 70%+ services healthy
- ✅ Backend API responding
- ✅ Frontend accessible
- ✅ AI models downloaded
- ✅ Monitoring stack operational

Access the system at:
- **Main Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Grafana Monitoring**: http://localhost:3000

---

**Note**: This deployment script represents the state-of-the-art in WSL2 Docker deployments as of 2025, incorporating all known fixes and optimizations for maximum reliability and performance.