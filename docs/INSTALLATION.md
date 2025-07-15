# SutazAI Installation Guide

## Prerequisites

### System Requirements

#### Minimum Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **CPU**: 8 cores (Intel i7 or AMD Ryzen 7)
- **RAM**: 16 GB
- **Storage**: 100 GB SSD
- **Python**: 3.8+
- **Docker**: 20.10+

#### Recommended Specifications
- **Operating System**: Ubuntu 22.04 LTS
- **CPU**: 16+ cores (Intel i9 or AMD Ryzen 9) 
- **RAM**: 32+ GB
- **Storage**: 500+ GB NVMe SSD
- **GPU**: NVIDIA RTX 4090 or similar (for ML acceleration)

### Software Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y     python3     python3-pip     python3-venv     docker.io     docker-compose     git     curl     wget     build-essential     sqlite3
```

## Installation Methods

### Method 1: Automated Installation (Recommended)

```bash
# Download and run the installation script
curl -sSL https://install.sutazai.com | bash

# Or manual download
wget https://raw.githubusercontent.com/sutazai/sutazaiapp/main/install.sh
chmod +x install.sh
./install.sh
```

### Method 2: Manual Installation

#### Step 1: Clone Repository
```bash
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp
```

#### Step 2: Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

#### Step 3: Configure Docker
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Test Docker installation
docker --version
docker-compose --version
```

#### Step 4: Initialize System
```bash
# Run quick deployment
python3 quick_deploy.py

# Or step-by-step initialization
python3 scripts/init_db.py
python3 scripts/init_ai.py
python3 security_fix.py
```

#### Step 5: Start the System
```bash
# Make startup script executable
chmod +x start.sh

# Start SutazAI
./start.sh
```

### Method 3: Docker Installation

#### Using Docker Compose
```bash
# Clone repository
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp

# Start with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
```

#### Manual Docker Setup
```bash
# Build the image
docker build -t sutazai:latest .

# Run the container
docker run -d     --name sutazai     -p 8000:8000     -v $(pwd)/data:/opt/sutazaiapp/data     -v $(pwd)/models:/opt/sutazaiapp/models     sutazai:latest
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Core Configuration
SUTAZAI_ROOT=/opt/sutazaiapp
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=sqlite:///data/sutazai.db
DATABASE_POOL_SIZE=20

# AI Configuration
AI_MODEL_PATH=/opt/sutazaiapp/models
VECTOR_DB_PATH=/opt/sutazaiapp/data/vectors
OLLAMA_HOST=http://localhost:11434

# Security Configuration
SECRET_KEY=your-very-secure-secret-key
ENCRYPTION_KEY=your-encryption-key
AUTHORIZED_USERS=chrissuta01@gmail.com

# Performance Configuration
MAX_WORKERS=8
CACHE_SIZE=1000
MEMORY_LIMIT=8192

# Monitoring Configuration
ENABLE_MONITORING=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30
```

### Database Configuration
```python
# config/database.py
DATABASE_CONFIG = {
    "type": "sqlite",
    "path": "/opt/sutazaiapp/data/sutazai.db",
    "pool_size": 20,
    "max_overflow": 40,
    "pool_timeout": 30,
    "optimization": {
        "wal_mode": True,
        "cache_size": -64000,  # 64MB
        "synchronous": "NORMAL",
        "journal_mode": "WAL"
    }
}
```

### AI Model Configuration
```python
# config/models.py
MODEL_CONFIG = {
    "local_models": {
        "code_llama": {
            "path": "models/code-llama-7b",
            "type": "code_generation",
            "enabled": True
        },
        "mistral": {
            "path": "models/mistral-7b",
            "type": "chat",
            "enabled": True
        }
    },
    "vector_db": {
        "chromadb": {
            "path": "data/vectors/chromadb",
            "collection": "sutazai_knowledge"
        },
        "faiss": {
            "path": "data/vectors/faiss",
            "index_type": "IVF"
        }
    }
}
```

## Post-Installation Setup

### 1. Verify Installation
```bash
# Run system tests
python3 scripts/test_system.py

# Check system health
curl http://localhost:8000/health

# Test API endpoints
curl http://localhost:8000/api/v1/status
```

### 2. Download AI Models
```bash
# Download local models
python3 scripts/download_models.py

# Initialize Ollama models
ollama pull llama2
ollama pull codellama
ollama pull mistral
```

### 3. Initialize Data
```bash
# Create initial user
python3 scripts/create_user.py --email chrissuta01@gmail.com --admin

# Import knowledge base
python3 scripts/import_knowledge.py --source data/knowledge/

# Initialize vector databases
python3 scripts/init_vectors.py
```

### 4. Configure Security
```bash
# Generate security keys
python3 scripts/generate_keys.py

# Setup SSL certificates (optional)
python3 scripts/setup_ssl.py

# Run security audit
python3 security_audit.py
```

## Service Management

### Systemd Service (Linux)
Create a systemd service file:

```bash
# Create service file
sudo nano /etc/systemd/system/sutazai.service
```

```ini
[Unit]
Description=SutazAI AGI/ASI System
After=network.target

[Service]
Type=forking
User=sutazai
Group=sutazai
WorkingDirectory=/opt/sutazaiapp
Environment=PYTHONPATH=/opt/sutazaiapp
ExecStart=/opt/sutazaiapp/start.sh
ExecStop=/bin/kill -TERM $MAINPID
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable sutazai
sudo systemctl start sutazai

# Check status
sudo systemctl status sutazai
```

### Docker Service Management
```bash
# Start service
docker-compose up -d

# Stop service
docker-compose down

# Restart service
docker-compose restart

# View logs
docker-compose logs -f
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using port 8000
sudo netstat -tulpn | grep :8000

# Kill process using port
sudo kill -9 $(sudo lsof -t -i:8000)
```

#### Database Connection Issues
```bash
# Check database file permissions
ls -la data/sutazai.db

# Reset database
rm data/sutazai.db
python3 scripts/init_db.py
```

#### Docker Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Restart session or run
newgrp docker
```

#### Memory Issues
```bash
# Check memory usage
free -h

# Adjust memory limits in .env
MEMORY_LIMIT=4096
MAX_WORKERS=4
```

### Log Analysis
```bash
# View application logs
tail -f logs/sutazai.log

# View error logs
tail -f logs/error.log

# View Docker logs
docker-compose logs -f sutazai
```

### Performance Issues
```bash
# Run performance analysis
python3 performance_optimization.py

# Monitor system resources
htop

# Check disk usage
df -h
du -sh data/
```

## Upgrading

### Manual Upgrade
```bash
# Backup current installation
cp -r /opt/sutazaiapp /opt/sutazaiapp.backup

# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Run migration scripts
python3 scripts/migrate.py

# Restart system
./restart.sh
```

### Automated Upgrade
```bash
# Run upgrade script
python3 scripts/upgrade.py

# Or use the upgrade command
./sutazai upgrade
```

## Uninstallation

### Remove SutazAI
```bash
# Stop services
sudo systemctl stop sutazai
sudo systemctl disable sutazai

# Remove service file
sudo rm /etc/systemd/system/sutazai.service

# Remove application
sudo rm -rf /opt/sutazaiapp

# Remove user data (optional)
rm -rf ~/.sutazai
```

### Clean Docker Installation
```bash
# Stop and remove containers
docker-compose down --volumes

# Remove images
docker rmi sutazai:latest

# Remove volumes
docker volume prune
```

## Support

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Review the logs in `logs/` directory
3. Submit an issue on [GitHub](https://github.com/sutazai/sutazaiapp/issues)
4. Join our [Discord community](https://discord.gg/sutazai)

---

**Installation completed successfully!** 🎉

Access SutazAI at: http://localhost:8000
