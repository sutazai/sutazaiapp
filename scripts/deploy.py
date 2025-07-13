#!/usr/bin/env python3
"""
SutazAI Enterprise Deployment Script
Complete automated deployment and setup
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/sutazaiapp")

class SutazAIDeployer:
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.venv_path = self.project_root / "venv"
        self.logs_dir = self.project_root / "logs"
        
    def run_command(self, command, cwd=None, check=True):
        """Run shell command with logging"""
        try:
            logger.info(f"Running: {command}")
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=cwd or self.project_root,
                capture_output=True, 
                text=True,
                check=check
            )
            
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
            
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def setup_environment(self):
        """Setup Python virtual environment"""
        logger.info("🐍 Setting up Python environment...")
        
        # Create virtual environment if it doesn't exist
        if not self.venv_path.exists():
            if not self.run_command(f"python3 -m venv {self.venv_path}"):
                logger.error("Failed to create virtual environment")
                return False
        
        # Activate and upgrade pip
        activate_cmd = f"source {self.venv_path}/bin/activate"
        if not self.run_command(f"{activate_cmd} && pip install --upgrade pip"):
            logger.error("Failed to upgrade pip")
            return False
        
        # Install requirements
        if not self.run_command(f"{activate_cmd} && pip install -r requirements.txt"):
            logger.error("Failed to install requirements")
            return False
        
        logger.info("✅ Python environment setup complete")
        return True
    
    def setup_directories(self):
        """Setup necessary directories"""
        logger.info("📁 Setting up directories...")
        
        directories = [
            "data", "logs", "cache", "models/ollama", 
            "temp", "run", "backup"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ Created: {dir_name}")
        
        return True
    
    def setup_database(self):
        """Setup database"""
        logger.info("🗄️ Setting up database...")
        
        activate_cmd = f"source {self.venv_path}/bin/activate"
        if not self.run_command(f"{activate_cmd} && python scripts/setup_database.py"):
            logger.error("Database setup failed")
            return False
        
        logger.info("✅ Database setup complete")
        return True
    
    def validate_system(self):
        """Validate system requirements"""
        logger.info("🔍 Validating system...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 8:
            logger.error(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        
        # Check disk space (need at least 1GB)
        stat = os.statvfs(self.project_root)
        free_space = stat.f_bavail * stat.f_frsize / (1024**3)  # GB
        if free_space < 1:
            logger.error(f"Insufficient disk space: {free_space:.1f}GB available, need 1GB+")
            return False
        
        logger.info("✅ System validation passed")
        return True
    
    def start_services(self):
        """Start all services"""
        logger.info("🚀 Starting SutazAI services...")
        
        # Use the start_all.sh script
        if not self.run_command("./bin/start_all.sh"):
            logger.error("Failed to start services")
            return False
        
        # Wait for services to be ready
        time.sleep(10)
        
        # Verify services
        return self.verify_deployment()
    
    def verify_deployment(self):
        """Verify deployment is working"""
        logger.info("✅ Verifying deployment...")
        
        services = [
            ("Backend API", "http://127.0.0.1:8000/health"),
            ("Ollama AI", "http://127.0.0.1:11434/health"),
            ("Web UI", "http://127.0.0.1:3000"),
        ]
        
        all_good = True
        for service_name, url in services:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"  ✅ {service_name}: OK")
                else:
                    logger.error(f"  ❌ {service_name}: HTTP {response.status_code}")
                    all_good = False
            except Exception as e:
                logger.error(f"  ❌ {service_name}: {e}")
                all_good = False
        
        # Test chat API
        try:
            chat_response = requests.post(
                "http://127.0.0.1:8000/api/chat",
                json={"message": "Test deployment"},
                timeout=10
            )
            if chat_response.status_code == 200:
                logger.info("  ✅ Chat API: Working")
            else:
                logger.error(f"  ❌ Chat API: HTTP {chat_response.status_code}")
                all_good = False
        except Exception as e:
            logger.error(f"  ❌ Chat API: {e}")
            all_good = False
        
        return all_good
    
    def create_systemd_service(self):
        """Create systemd service for auto-start"""
        logger.info("⚙️ Creating systemd service...")
        
        service_content = f"""[Unit]
Description=SutazAI Enterprise System
After=network.target

[Service]
Type=forking
User=root
WorkingDirectory={self.project_root}
ExecStart={self.project_root}/bin/start_all.sh
ExecStop={self.project_root}/bin/stop_all.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        try:
            service_file = Path("/etc/systemd/system/sutazai.service")
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            # Reload systemd and enable service
            self.run_command("systemctl daemon-reload")
            self.run_command("systemctl enable sutazai.service")
            
            logger.info("✅ Systemd service created and enabled")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create systemd service: {e}")
            return False
    
    def deploy(self):
        """Main deployment function"""
        logger.info("🚀 Starting SutazAI Enterprise Deployment...")
        
        steps = [
            ("System Validation", self.validate_system),
            ("Directory Setup", self.setup_directories),
            ("Environment Setup", self.setup_environment),
            ("Database Setup", self.setup_database),
            ("Service Startup", self.start_services),
            ("Systemd Service", self.create_systemd_service),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 {step_name}...")
            if not step_func():
                logger.error(f"❌ {step_name} failed!")
                return False
            logger.info(f"✅ {step_name} completed")
        
        logger.info("🎉 SutazAI Enterprise Deployment Complete!")
        logger.info("🌐 Access Points:")
        logger.info("  • Main Dashboard: http://127.0.0.1:3000")
        logger.info("  • Chat Interface: http://127.0.0.1:3000/chat.html")
        logger.info("  • API Documentation: http://127.0.0.1:8000/docs")
        logger.info("  • System Health: http://127.0.0.1:8000/health")
        
        return True

def main():
    """Main function"""
    deployer = SutazAIDeployer()
    success = deployer.deploy()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())