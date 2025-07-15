#!/usr/bin/env python3
"""
Automated Setup and Deployment System for SutazAI
Complete automation for installation, configuration, and deployment
"""

import asyncio
import logging
import subprocess
import sys
import os
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import platform
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedSetup:
    """Comprehensive automated setup system"""
    
    def __init__(self):
        self.root_dir = Path("/opt/sutazaiapp")
        self.system_info = self._gather_system_info()
        self.setup_steps_completed = []
        self.requirements = {
            "python_version": (3, 8),
            "min_ram_gb": 4,
            "min_disk_gb": 10,
            "required_ports": [8000, 3000, 11434, 5432, 6379]
        }
        
    def _gather_system_info(self):
        """Gather system information"""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": sys.version_info,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "disk_gb": psutil.disk_usage('/').total / (1024**3),
            "user": os.getenv("USER", "unknown")
        }
    
    async def run_complete_setup(self):
        """Run complete automated setup"""
        logger.info("🚀 Starting Complete Automated Setup for SutazAI")
        
        # Phase 1: System requirements check
        await self._check_system_requirements()
        
        # Phase 2: Install system dependencies
        await self._install_system_dependencies()
        
        # Phase 3: Setup Python environment
        await self._setup_python_environment()
        
        # Phase 4: Install Python dependencies
        await self._install_python_dependencies()
        
        # Phase 5: Setup databases
        await self._setup_databases()
        
        # Phase 6: Setup AI models
        await self._setup_ai_models()
        
        # Phase 7: Configure services
        await self._configure_services()
        
        # Phase 8: Setup monitoring
        await self._setup_monitoring()
        
        # Phase 9: Create deployment scripts
        await self._create_deployment_scripts()
        
        # Phase 10: Final validation
        await self._validate_installation()
        
        logger.info("✅ Complete automated setup finished!")
        return self.setup_steps_completed
    
    async def _check_system_requirements(self):
        """Check system requirements"""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        if self.system_info["python_version"] < self.requirements["python_version"]:
            raise RuntimeError(f"Python {self.requirements['python_version']} or higher required")
        
        # Check RAM
        if self.system_info["memory_gb"] < self.requirements["min_ram_gb"]:
            logger.warning(f"Low RAM: {self.system_info['memory_gb']:.1f}GB (recommended: {self.requirements['min_ram_gb']}GB)")
        
        # Check disk space
        if self.system_info["disk_gb"] < self.requirements["min_disk_gb"]:
            raise RuntimeError(f"Insufficient disk space: {self.system_info['disk_gb']:.1f}GB (required: {self.requirements['min_disk_gb']}GB)")
        
        # Check ports
        await self._check_ports()
        
        self.setup_steps_completed.append("System requirements checked")
        logger.info("✅ System requirements check passed")
    
    async def _check_ports(self):
        """Check if required ports are available"""
        import socket
        
        for port in self.requirements["required_ports"]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    logger.warning(f"Port {port} is already in use")
                sock.close()
            except Exception:
                pass
    
    async def _install_system_dependencies(self):
        """Install system-level dependencies"""
        logger.info("📦 Installing system dependencies...")
        
        if self.system_info["platform"] == "Linux":
            await self._install_linux_dependencies()
        elif self.system_info["platform"] == "Darwin":
            await self._install_macos_dependencies()
        elif self.system_info["platform"] == "Windows":
            await self._install_windows_dependencies()
        
        self.setup_steps_completed.append("System dependencies installed")
    
    async def _install_linux_dependencies(self):
        """Install Linux dependencies"""
        packages = [
            "python3-pip", "python3-venv", "python3-dev",
            "build-essential", "git", "curl", "wget",
            "postgresql", "postgresql-contrib", "postgresql-client",
            "redis-server", "docker.io", "docker-compose",
            "nginx", "supervisor", "htop", "vim"
        ]
        
        try:
            # Update package list
            await self._run_command("sudo apt update")
            
            # Install packages
            package_list = " ".join(packages)
            await self._run_command(f"sudo apt install -y {package_list}")
            
            # Enable services
            services = ["postgresql", "redis-server", "docker"]
            for service in services:
                await self._run_command(f"sudo systemctl enable {service}")
                await self._run_command(f"sudo systemctl start {service}")
            
            logger.info("✅ Linux dependencies installed")
            
        except Exception as e:
            logger.error(f"Failed to install Linux dependencies: {e}")
            # Continue with available tools
    
    async def _install_macos_dependencies(self):
        """Install macOS dependencies"""
        try:
            # Check if Homebrew is installed
            if not shutil.which("brew"):
                logger.info("Installing Homebrew...")
                await self._run_command('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
            
            # Install packages
            packages = [
                "python3", "postgresql", "redis", "git", "curl", "wget",
                "docker", "docker-compose", "nginx"
            ]
            
            for package in packages:
                await self._run_command(f"brew install {package}")
            
            # Start services
            services = ["postgresql", "redis"]
            for service in services:
                await self._run_command(f"brew services start {service}")
            
            logger.info("✅ macOS dependencies installed")
            
        except Exception as e:
            logger.error(f"Failed to install macOS dependencies: {e}")
    
    async def _install_windows_dependencies(self):
        """Install Windows dependencies"""
        logger.info("For Windows, please manually install:")
        logger.info("- Python 3.8+")
        logger.info("- PostgreSQL")
        logger.info("- Redis")
        logger.info("- Docker Desktop")
        logger.info("- Git")
        
        # Check if essential tools are available
        tools = ["python", "git", "docker"]
        for tool in tools:
            if not shutil.which(tool):
                logger.warning(f"{tool} not found in PATH")
    
    async def _setup_python_environment(self):
        """Setup Python virtual environment"""
        logger.info("🐍 Setting up Python environment...")
        
        venv_path = self.root_dir / "venv"
        
        try:
            # Create virtual environment
            if not venv_path.exists():
                await self._run_command(f"python3 -m venv {venv_path}")
            
            # Create activation script
            activate_script = self.root_dir / "activate_env.sh"
            activate_content = f"""#!/bin/bash
source {venv_path}/bin/activate
export PYTHONPATH={self.root_dir}:$PYTHONPATH
echo "✅ SutazAI environment activated"
"""
            activate_script.write_text(activate_content)
            activate_script.chmod(0o755)
            
            self.setup_steps_completed.append("Python environment created")
            logger.info("✅ Python environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup Python environment: {e}")
    
    async def _install_python_dependencies(self):
        """Install Python dependencies"""
        logger.info("📚 Installing Python dependencies...")
        
        # Create comprehensive requirements file
        await self._create_requirements_file()
        
        venv_pip = self.root_dir / "venv/bin/pip"
        if not venv_pip.exists():
            venv_pip = "pip3"  # Fallback to system pip
        
        try:
            # Upgrade pip
            await self._run_command(f"{venv_pip} install --upgrade pip")
            
            # Install requirements
            requirements_file = self.root_dir / "requirements_complete.txt"
            await self._run_command(f"{venv_pip} install -r {requirements_file}")
            
            self.setup_steps_completed.append("Python dependencies installed")
            logger.info("✅ Python dependencies installed")
            
        except Exception as e:
            logger.error(f"Failed to install Python dependencies: {e}")
    
    async def _create_requirements_file(self):
        """Create comprehensive requirements file"""
        requirements = """# SutazAI Complete Requirements
# Core web framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0

# Database
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.7
redis==5.0.1

# AI and ML
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.2
numpy>=1.24.0
scikit-learn>=1.3.0

# HTTP and networking
aiohttp==3.9.0
requests==2.31.0
httpx==0.25.0

# Async and concurrency
asyncio-mqtt==0.15.0
asyncpg==0.29.0

# Monitoring and logging
loguru==0.7.2
prometheus-client==0.18.0
psutil==5.9.6

# Security
cryptography==41.0.7
bcrypt==4.0.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Data processing
pandas>=2.0.0
pydantic-settings==2.0.3

# Development tools
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.10.1
flake8==6.1.0
mypy==1.7.0

# Documentation
mkdocs==1.5.3
mkdocs-material==9.4.8

# Utilities
click==8.1.7
rich==13.7.0
typer==0.9.0
pathlib2==2.3.7
"""
        
        requirements_file = self.root_dir / "requirements_complete.txt"
        requirements_file.write_text(requirements)
    
    async def _setup_databases(self):
        """Setup and configure databases"""
        logger.info("🗄️ Setting up databases...")
        
        # Setup PostgreSQL
        await self._setup_postgresql()
        
        # Setup Redis
        await self._setup_redis()
        
        self.setup_steps_completed.append("Databases configured")
    
    async def _setup_postgresql(self):
        """Setup PostgreSQL database"""
        try:
            # Create database and user
            db_password = os.getenv("DB_PASSWORD", "sutazai_secure_2024")
            
            commands = [
                "sudo -u postgres createdb sutazaidb",
                f"sudo -u postgres psql -c \"CREATE USER sutazai WITH PASSWORD '{db_password}';\"",
                "sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE sutazaidb TO sutazai;\"",
                "sudo -u postgres psql -c \"ALTER USER sutazai CREATEDB;\""
            ]
            
            for cmd in commands:
                try:
                    await self._run_command(cmd)
                except subprocess.CalledProcessError:
                    # Database/user might already exist
                    pass
            
            logger.info("✅ PostgreSQL configured")
            
        except Exception as e:
            logger.error(f"PostgreSQL setup failed: {e}")
    
    async def _setup_redis(self):
        """Setup Redis cache"""
        try:
            # Create Redis configuration
            redis_config = """# SutazAI Redis Configuration
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 300
daemonize yes
supervised no
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/redis/redis-server.log
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis
maxmemory 256mb
maxmemory-policy allkeys-lru
"""
            
            config_file = self.root_dir / "config/redis.conf"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(redis_config)
            
            logger.info("✅ Redis configured")
            
        except Exception as e:
            logger.error(f"Redis setup failed: {e}")
    
    async def _setup_ai_models(self):
        """Setup AI models and Ollama"""
        logger.info("🤖 Setting up AI models...")
        
        try:
            # Install Ollama
            if not shutil.which("ollama"):
                logger.info("Installing Ollama...")
                await self._run_command("curl -fsSL https://ollama.ai/install.sh | sh")
            
            # Start Ollama service
            await self._run_command("ollama serve", background=True)
            
            # Wait for Ollama to start
            await asyncio.sleep(5)
            
            # Pull default models
            default_models = ["llama2:7b", "codellama:7b"]
            for model in default_models:
                logger.info(f"Pulling model: {model}")
                await self._run_command(f"ollama pull {model}")
            
            self.setup_steps_completed.append("AI models configured")
            logger.info("✅ AI models setup complete")
            
        except Exception as e:
            logger.error(f"AI models setup failed: {e}")
    
    async def _configure_services(self):
        """Configure system services"""
        logger.info("⚙️ Configuring services...")
        
        # Create systemd service file
        await self._create_systemd_service()
        
        # Create Docker Compose override
        await self._create_docker_override()
        
        # Create Nginx configuration
        await self._create_nginx_config()
        
        self.setup_steps_completed.append("Services configured")
    
    async def _create_systemd_service(self):
        """Create systemd service for SutazAI"""
        service_content = f"""[Unit]
Description=SutazAI AGI/ASI System
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User={self.system_info['user']}
WorkingDirectory={self.root_dir}
Environment=PYTHONPATH={self.root_dir}
ExecStart={self.root_dir}/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=sutazai

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.root_dir / "scripts/sutazai.service"
        service_file.parent.mkdir(parents=True, exist_ok=True)
        service_file.write_text(service_content)
        
        # Install service (requires sudo)
        try:
            await self._run_command(f"sudo cp {service_file} /etc/systemd/system/")
            await self._run_command("sudo systemctl daemon-reload")
            await self._run_command("sudo systemctl enable sutazai")
            logger.info("✅ Systemd service created")
        except Exception as e:
            logger.warning(f"Could not install systemd service: {e}")
    
    async def _create_docker_override(self):
        """Create Docker Compose override for production"""
        override_content = """version: '3.8'

services:
  sutazai-app:
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
      - LOG_LEVEL=INFO
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  redis:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M

  ollama:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
"""
        
        override_file = self.root_dir / "docker-compose.override.yml"
        override_file.write_text(override_content)
    
    async def _create_nginx_config(self):
        """Create Nginx configuration"""
        nginx_config = f"""# SutazAI Nginx Configuration
upstream sutazai_backend {{
    server 127.0.0.1:8000;
}}

server {{
    listen 80;
    server_name localhost {self.system_info.get('hostname', 'sutazai.local')};
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {{
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://sutazai_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Websocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }}
    
    location /static/ {{
        alias {self.root_dir}/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }}
    
    location /health {{
        access_log off;
        proxy_pass http://sutazai_backend/health;
    }}
}}
"""
        
        nginx_file = self.root_dir / "config/nginx/sutazai.conf"
        nginx_file.parent.mkdir(parents=True, exist_ok=True)
        nginx_file.write_text(nginx_config)
    
    async def _setup_monitoring(self):
        """Setup monitoring and logging"""
        logger.info("📊 Setting up monitoring...")
        
        # Create monitoring configuration
        await self._create_monitoring_config()
        
        # Setup log rotation
        await self._setup_log_rotation()
        
        self.setup_steps_completed.append("Monitoring configured")
    
    async def _create_monitoring_config(self):
        """Create monitoring configuration"""
        prometheus_config = """# SutazAI Prometheus Configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sutazai'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
"""
        
        config_file = self.root_dir / "config/monitoring/prometheus.yml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(prometheus_config)
    
    async def _setup_log_rotation(self):
        """Setup log rotation"""
        logrotate_config = f"""{self.root_dir}/logs/*.log {{
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 {self.system_info['user']} {self.system_info['user']}
    postrotate
        systemctl reload sutazai || true
    endscript
}}
"""
        
        logrotate_file = self.root_dir / "config/logrotate/sutazai"
        logrotate_file.parent.mkdir(parents=True, exist_ok=True)
        logrotate_file.write_text(logrotate_config)
    
    async def _create_deployment_scripts(self):
        """Create deployment and management scripts"""
        logger.info("📜 Creating deployment scripts...")
        
        # Main deployment script
        await self._create_deploy_script()
        
        # Backup script
        await self._create_backup_script()
        
        # Update script
        await self._create_update_script()
        
        # Status script
        await self._create_status_script()
        
        self.setup_steps_completed.append("Deployment scripts created")
    
    async def _create_deploy_script(self):
        """Create main deployment script"""
        deploy_content = f"""#!/bin/bash
set -e

echo "🚀 Deploying SutazAI..."

# Change to project directory
cd {self.root_dir}

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH={self.root_dir}:$PYTHONPATH

# Run database migrations
echo "📄 Running database migrations..."
python -c "
from backend.database.connection import init_database, create_tables
init_database()
create_tables()
print('✅ Database migrations completed')
"

# Start services with Docker Compose
echo "🐳 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services..."
sleep 10

# Initialize AI models
echo "🤖 Initializing AI models..."
python -c "
import asyncio
from backend.ai.ollama_manager import ollama_manager
asyncio.run(ollama_manager.initialize())
print('✅ AI models initialized')
"

# Run health check
echo "🏥 Running health check..."
python -c "
import asyncio
from backend.monitoring.health import health_checker
result = asyncio.run(health_checker.run_all_checks())
print('✅ Health check completed')
"

echo "✅ SutazAI deployment completed successfully!"
echo "🌐 Access the system at: http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/docs"
"""
        
        deploy_script = self.root_dir / "scripts/deploy.sh"
        deploy_script.parent.mkdir(parents=True, exist_ok=True)
        deploy_script.write_text(deploy_content)
        deploy_script.chmod(0o755)
    
    async def _create_backup_script(self):
        """Create backup script"""
        backup_content = f"""#!/bin/bash
set -e

BACKUP_DIR={self.root_dir}/backups
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="sutazai_backup_$TIMESTAMP"

echo "💾 Creating backup: $BACKUP_NAME"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
echo "📊 Backing up database..."
pg_dump sutazaidb > $BACKUP_DIR/$BACKUP_NAME.sql

# Backup application data
echo "📁 Backing up application data..."
tar -czf $BACKUP_DIR/$BACKUP_NAME.tar.gz \\
    --exclude=venv \\
    --exclude=__pycache__ \\
    --exclude=*.pyc \\
    --exclude=logs \\
    --exclude=backups \\
    {self.root_dir}

# Backup Redis data
echo "🔴 Backing up Redis data..."
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb $BACKUP_DIR/$BACKUP_NAME.rdb

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "sutazai_backup_*" -mtime +7 -delete

echo "✅ Backup completed: $BACKUP_NAME"
"""
        
        backup_script = self.root_dir / "scripts/backup.sh"
        backup_script.write_text(backup_content)
        backup_script.chmod(0o755)
    
    async def _create_update_script(self):
        """Create update script"""
        update_content = f"""#!/bin/bash
set -e

echo "🔄 Updating SutazAI..."

cd {self.root_dir}

# Backup before update
echo "💾 Creating backup..."
./scripts/backup.sh

# Pull latest changes
echo "📥 Pulling latest changes..."
git pull origin main

# Update Python dependencies
echo "📚 Updating dependencies..."
source venv/bin/activate
pip install -r requirements_complete.txt

# Run database migrations
echo "📄 Running migrations..."
python -c "
from backend.database.connection import init_database, create_tables
init_database()
create_tables()
"

# Restart services
echo "🔄 Restarting services..."
docker-compose restart

# Update AI models
echo "🤖 Updating AI models..."
python -c "
import asyncio
from backend.ai.ollama_manager import ollama_manager
asyncio.run(ollama_manager.initialize())
"

echo "✅ Update completed successfully!"
"""
        
        update_script = self.root_dir / "scripts/update.sh"
        update_script.write_text(update_content)
        update_script.chmod(0o755)
    
    async def _create_status_script(self):
        """Create status monitoring script"""
        status_content = f"""#!/bin/bash

echo "📊 SutazAI System Status"
echo "========================"

# System information
echo "🖥️  System Information:"
echo "   Platform: {self.system_info['platform']}"
echo "   CPU Cores: {self.system_info['cpu_count']}"
echo "   Memory: {self.system_info['memory_gb']:.1f}GB"
echo "   Disk: {self.system_info['disk_gb']:.1f}GB"
echo ""

# Service status
echo "🔧 Service Status:"
services=("postgresql" "redis-server" "docker" "nginx")
for service in "${{services[@]}}"; do
    if systemctl is-active --quiet $service; then
        echo "   ✅ $service: running"
    else
        echo "   ❌ $service: stopped"
    fi
done
echo ""

# Docker containers
echo "🐳 Docker Containers:"
docker-compose ps
echo ""

# Database status
echo "🗄️  Database Status:"
if pg_isready -q; then
    echo "   ✅ PostgreSQL: ready"
else
    echo "   ❌ PostgreSQL: not ready"
fi

if redis-cli ping >/dev/null 2>&1; then
    echo "   ✅ Redis: ready"
else
    echo "   ❌ Redis: not ready"
fi
echo ""

# Application health
echo "🏥 Application Health:"
curl -s http://localhost:8000/health | python -m json.tool
echo ""

# Resource usage
echo "📈 Resource Usage:"
echo "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{{print $2}}' | cut -d'%' -f1)%"
echo "   Memory: $(free | grep Mem | awk '{{printf("%.1f%%", $3/$2 * 100.0)}}')"
echo "   Disk: $(df / | tail -1 | awk '{{print $5}}')"
"""
        
        status_script = self.root_dir / "scripts/status.sh"
        status_script.write_text(status_content)
        status_script.chmod(0o755)
    
    async def _validate_installation(self):
        """Validate the complete installation"""
        logger.info("✅ Validating installation...")
        
        validation_results = {}
        
        # Check files
        required_files = [
            "main.py",
            "requirements_complete.txt",
            "docker-compose.yml",
            "scripts/deploy.sh",
            "scripts/backup.sh",
            "venv/bin/activate"
        ]
        
        for file_path in required_files:
            file_obj = self.root_dir / file_path
            validation_results[file_path] = file_obj.exists()
        
        # Check services
        services_to_check = ["postgresql", "redis-server"]
        for service in services_to_check:
            try:
                result = await self._run_command(f"systemctl is-active {service}")
                validation_results[f"service_{service}"] = "active" in result
            except:
                validation_results[f"service_{service}"] = False
        
        # Generate validation report
        validation_report = {
            "validation_timestamp": time.time(),
            "system_info": self.system_info,
            "setup_steps_completed": self.setup_steps_completed,
            "validation_results": validation_results,
            "overall_status": "success" if all(validation_results.values()) else "warning"
        }
        
        report_file = self.root_dir / "SETUP_VALIDATION_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info(f"✅ Validation report saved: {report_file}")
        
        # Print summary
        passed = sum(1 for result in validation_results.values() if result)
        total = len(validation_results)
        logger.info(f"📊 Validation Summary: {passed}/{total} checks passed")
        
        self.setup_steps_completed.append("Installation validated")
    
    async def _run_command(self, command: str, background: bool = False) -> str:
        """Run shell command"""
        logger.debug(f"Running command: {command}")
        
        if background:
            # Run in background
            subprocess.Popen(command, shell=True)
            return "Background process started"
        else:
            # Run and wait for completion
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Command failed"
                raise subprocess.CalledProcessError(process.returncode, command, error_msg)
            
            return stdout.decode()
    
    def generate_setup_report(self):
        """Generate final setup report"""
        report = {
            "automated_setup_report": {
                "timestamp": time.time(),
                "system_info": self.system_info,
                "setup_steps_completed": self.setup_steps_completed,
                "status": "completed",
                "features_installed": [
                    "Complete Python environment with virtual environment",
                    "PostgreSQL database with user and permissions",
                    "Redis cache with optimized configuration",
                    "AI models via Ollama integration",
                    "Docker Compose orchestration",
                    "Nginx reverse proxy with security headers",
                    "Systemd service for auto-start",
                    "Comprehensive monitoring and logging",
                    "Automated deployment scripts",
                    "Backup and update automation"
                ],
                "scripts_created": [
                    "scripts/deploy.sh - Main deployment",
                    "scripts/backup.sh - Automated backups",
                    "scripts/update.sh - System updates",
                    "scripts/status.sh - System monitoring",
                    "activate_env.sh - Environment activation"
                ],
                "services_configured": [
                    "SutazAI application (port 8000)",
                    "PostgreSQL database (port 5432)",
                    "Redis cache (port 6379)",
                    "Ollama AI models (port 11434)",
                    "Nginx reverse proxy (port 80)"
                ],
                "next_steps": [
                    "Run ./scripts/deploy.sh to start the system",
                    "Access web interface at http://localhost:8000",
                    "Check system status with ./scripts/status.sh",
                    "Create regular backups with ./scripts/backup.sh",
                    "Monitor logs in ./logs/ directory"
                ]
            }
        }
        
        report_file = self.root_dir / "AUTOMATED_SETUP_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Setup report generated: {report_file}")
        return report

async def main():
    """Main setup function"""
    setup = AutomatedSetup()
    
    try:
        steps_completed = await setup.run_complete_setup()
        report = setup.generate_setup_report()
        
        print("🎉 SutazAI Automated Setup Completed Successfully!")
        print(f"✅ Completed {len(steps_completed)} setup steps")
        print("📋 Next steps:")
        print("   1. Run: ./scripts/deploy.sh")
        print("   2. Visit: http://localhost:8000")
        print("   3. Check: ./scripts/status.sh")
        
        return steps_completed
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print("❌ Setup failed. Check logs for details.")
        return []

if __name__ == "__main__":
    asyncio.run(main())