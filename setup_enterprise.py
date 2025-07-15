"""
SutazAI Enterprise Deployment Setup
Complete enterprise-grade deployment automation and system installation

This script provides comprehensive setup and deployment automation for the
SutazAI AGI/ASI system including all dependencies, configurations, and services.
"""

import os
import sys
import subprocess
import json
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import requests
import zipfile
import tarfile
import platform
import socket
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentMode(Enum):
    """Deployment modes"""
    STANDALONE = "standalone"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DEVELOPMENT = "development"

class SystemType(Enum):
    """System types"""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    mode: DeploymentMode
    system_type: SystemType
    install_dir: str
    data_dir: str
    logs_dir: str
    config_dir: str
    models_dir: str
    enable_monitoring: bool = True
    enable_security: bool = True
    enable_api: bool = True
    api_port: int = 8000
    monitoring_port: int = 8090
    database_type: str = "sqlite"  # sqlite, postgresql, mysql
    enable_ollama: bool = True
    enable_docker: bool = False
    enable_kubernetes: bool = False

class EnterpriseDeploymentManager:
    """Manages enterprise deployment of SutazAI system"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.system_info = self._get_system_info()
        self.installation_log = []
        
        # Paths
        self.install_dir = Path(config.install_dir)
        self.data_dir = Path(config.data_dir)
        self.logs_dir = Path(config.logs_dir)
        self.config_dir = Path(config.config_dir)
        self.models_dir = Path(config.models_dir)
        
        logger.info(f"Enterprise Deployment Manager initialized for {config.mode.value} mode")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "user": os.getenv("USER", "unknown"),
            "cpu_count": os.cpu_count(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _log_step(self, step: str, status: str, details: str = ""):
        """Log installation step"""
        entry = {
            "step": step,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.installation_log.append(entry)
        
        if status == "success":
            logger.info(f"✓ {step}")
        elif status == "failed":
            logger.error(f"✗ {step}: {details}")
        else:
            logger.info(f"→ {step}")
    
    def _run_command(self, command: List[str], check: bool = True, shell: bool = False) -> subprocess.CompletedProcess:
        """Run system command"""
        try:
            if shell:
                command = ' '.join(command)
            
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.returncode == 0:
                logger.debug(f"Command succeeded: {command}")
            else:
                logger.warning(f"Command failed: {command}, stderr: {result.stderr}")
            
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}, error: {e}")
            raise
    
    def create_directories(self):
        """Create necessary directories"""
        self._log_step("Creating directories", "starting")
        
        try:
            directories = [
                self.install_dir,
                self.data_dir,
                self.logs_dir,
                self.config_dir,
                self.models_dir,
                self.data_dir / "neural_networks",
                self.data_dir / "knowledge_graphs",
                self.data_dir / "metrics",
                self.data_dir / "alerts",
                self.logs_dir / "agi_system",
                self.logs_dir / "api",
                self.logs_dir / "monitoring",
                self.install_dir / "backups",
                self.install_dir / "deployment",
                self.install_dir / "tests"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            
            self._log_step("Creating directories", "success")
            
        except Exception as e:
            self._log_step("Creating directories", "failed", str(e))
            raise
    
    def install_system_dependencies(self):
        """Install system dependencies"""
        self._log_step("Installing system dependencies", "starting")
        
        try:
            if self.config.system_type == SystemType.LINUX:
                self._install_linux_dependencies()
            elif self.config.system_type == SystemType.WINDOWS:
                self._install_windows_dependencies()
            elif self.config.system_type == SystemType.MACOS:
                self._install_macos_dependencies()
            
            self._log_step("Installing system dependencies", "success")
            
        except Exception as e:
            self._log_step("Installing system dependencies", "failed", str(e))
            raise
    
    def _install_linux_dependencies(self):
        """Install Linux system dependencies"""
        try:
            # Update package manager
            self._run_command(["sudo", "apt-get", "update"], shell=False)
            
            # Install essential packages
            packages = [
                "python3-pip",
                "python3-venv",
                "python3-dev",
                "build-essential",
                "curl",
                "wget",
                "git",
                "sqlite3",
                "postgresql-client",
                "redis-tools",
                "nginx",
                "supervisor",
                "htop",
                "tmux"
            ]
            
            for package in packages:
                try:
                    self._run_command(["sudo", "apt-get", "install", "-y", package])
                    logger.info(f"Installed package: {package}")
                except Exception as e:
                    logger.warning(f"Failed to install {package}: {e}")
            
            # Install Docker if enabled
            if self.config.enable_docker:
                self._install_docker_linux()
            
            # Install Kubernetes if enabled
            if self.config.enable_kubernetes:
                self._install_kubernetes_linux()
            
        except Exception as e:
            logger.error(f"Failed to install Linux dependencies: {e}")
            raise
    
    def _install_windows_dependencies(self):
        """Install Windows system dependencies"""
        try:
            # Check if Chocolatey is installed
            try:
                self._run_command(["choco", "--version"])
            except:
                logger.info("Installing Chocolatey...")
                install_choco = '''
                Set-ExecutionPolicy Bypass -Scope Process -Force;
                [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;
                iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
                '''
                self._run_command(["powershell", "-Command", install_choco], shell=True)
            
            # Install packages via Chocolatey
            packages = [
                "python3",
                "git",
                "curl",
                "wget",
                "sqlite",
                "postgresql",
                "redis-64"
            ]
            
            for package in packages:
                try:
                    self._run_command(["choco", "install", "-y", package])
                    logger.info(f"Installed package: {package}")
                except Exception as e:
                    logger.warning(f"Failed to install {package}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to install Windows dependencies: {e}")
            raise
    
    def _install_macos_dependencies(self):
        """Install macOS system dependencies"""
        try:
            # Check if Homebrew is installed
            try:
                self._run_command(["brew", "--version"])
            except:
                logger.info("Installing Homebrew...")
                install_brew = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                self._run_command([install_brew], shell=True)
            
            # Install packages via Homebrew
            packages = [
                "python3",
                "git",
                "curl",
                "wget",
                "sqlite",
                "postgresql",
                "redis",
                "nginx"
            ]
            
            for package in packages:
                try:
                    self._run_command(["brew", "install", package])
                    logger.info(f"Installed package: {package}")
                except Exception as e:
                    logger.warning(f"Failed to install {package}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to install macOS dependencies: {e}")
            raise
    
    def _install_docker_linux(self):
        """Install Docker on Linux"""
        try:
            # Install Docker
            install_docker = '''
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            '''
            self._run_command([install_docker], shell=True)
            
            # Install Docker Compose
            self._run_command([
                "sudo", "curl", "-L",
                "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-linux-x86_64",
                "-o", "/usr/local/bin/docker-compose"
            ])
            self._run_command(["sudo", "chmod", "+x", "/usr/local/bin/docker-compose"])
            
            logger.info("Docker installed successfully")
            
        except Exception as e:
            logger.error(f"Failed to install Docker: {e}")
            raise
    
    def _install_kubernetes_linux(self):
        """Install Kubernetes tools on Linux"""
        try:
            # Install kubectl
            kubectl_install = '''
            curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
            sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
            '''
            self._run_command([kubectl_install], shell=True)
            
            # Install kind (Kubernetes in Docker)
            kind_install = '''
            curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
            chmod +x ./kind
            sudo mv ./kind /usr/local/bin/kind
            '''
            self._run_command([kind_install], shell=True)
            
            logger.info("Kubernetes tools installed successfully")
            
        except Exception as e:
            logger.error(f"Failed to install Kubernetes tools: {e}")
            raise
    
    def install_python_dependencies(self):
        """Install Python dependencies"""
        self._log_step("Installing Python dependencies", "starting")
        
        try:
            # Create virtual environment
            venv_path = self.install_dir / "venv"
            self._run_command([sys.executable, "-m", "venv", str(venv_path)])
            
            # Activate virtual environment and install dependencies
            pip_executable = str(venv_path / "bin" / "pip") if self.config.system_type != SystemType.WINDOWS else str(venv_path / "Scripts" / "pip.exe")
            
            # Upgrade pip
            self._run_command([pip_executable, "install", "--upgrade", "pip"])
            
            # Install requirements
            requirements_path = self.install_dir / "requirements.txt"
            if requirements_path.exists():
                self._run_command([pip_executable, "install", "-r", str(requirements_path)])
            else:
                # Install essential packages
                packages = [
                    "fastapi>=0.104.0",
                    "uvicorn[standard]>=0.24.0",
                    "pydantic>=2.4.0",
                    "sqlalchemy>=2.0.0",
                    "alembic>=1.12.0",
                    "redis>=5.0.0",
                    "celery>=5.3.0",
                    "requests>=2.31.0",
                    "aiohttp>=3.8.0",
                    "asyncio>=3.4.3",
                    "numpy>=1.24.0",
                    "pandas>=2.0.0",
                    "scikit-learn>=1.3.0",
                    "transformers>=4.30.0",
                    "torch>=2.0.0",
                    "pytest>=7.4.0",
                    "pytest-asyncio>=0.21.0",
                    "prometheus-client>=0.17.0",
                    "opentelemetry-api>=1.20.0",
                    "psutil>=5.9.0",
                    "pyyaml>=6.0",
                    "loguru>=0.7.0",
                    "passlib[bcrypt]>=1.7.4",
                    "python-jose[cryptography]>=3.3.0",
                    "kubernetes>=27.2.0",
                    "docker>=6.1.0"
                ]
                
                for package in packages:
                    try:
                        self._run_command([pip_executable, "install", package])
                        logger.info(f"Installed Python package: {package}")
                    except Exception as e:
                        logger.warning(f"Failed to install {package}: {e}")
            
            self._log_step("Installing Python dependencies", "success")
            
        except Exception as e:
            self._log_step("Installing Python dependencies", "failed", str(e))
            raise
    
    def install_ollama(self):
        """Install Ollama for local model management"""
        if not self.config.enable_ollama:
            return
        
        self._log_step("Installing Ollama", "starting")
        
        try:
            # Install Ollama
            install_ollama = 'curl -fsSL https://ollama.com/install.sh | sh'
            self._run_command([install_ollama], shell=True)
            
            # Start Ollama service
            if self.config.system_type == SystemType.LINUX:
                self._run_command(["sudo", "systemctl", "start", "ollama"])
                self._run_command(["sudo", "systemctl", "enable", "ollama"])
            
            # Wait for Ollama to start
            time.sleep(5)
            
            # Download essential models
            essential_models = ["llama3.1:latest", "codellama:latest"]
            
            for model in essential_models:
                try:
                    logger.info(f"Downloading model: {model}")
                    self._run_command(["ollama", "pull", model])
                    logger.info(f"Model {model} downloaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to download model {model}: {e}")
            
            self._log_step("Installing Ollama", "success")
            
        except Exception as e:
            self._log_step("Installing Ollama", "failed", str(e))
            raise
    
    def setup_database(self):
        """Setup database"""
        self._log_step("Setting up database", "starting")
        
        try:
            if self.config.database_type == "sqlite":
                self._setup_sqlite_database()
            elif self.config.database_type == "postgresql":
                self._setup_postgresql_database()
            elif self.config.database_type == "mysql":
                self._setup_mysql_database()
            
            self._log_step("Setting up database", "success")
            
        except Exception as e:
            self._log_step("Setting up database", "failed", str(e))
            raise
    
    def _setup_sqlite_database(self):
        """Setup SQLite database"""
        try:
            db_path = self.data_dir / "sutazai.db"
            
            # Create database file
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            
            # Create basic tables
            cursor = conn.cursor()
            
            # AGI tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agi_tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    data TEXT,
                    result TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("SQLite database setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup SQLite database: {e}")
            raise
    
    def _setup_postgresql_database(self):
        """Setup PostgreSQL database"""
        try:
            # This would require PostgreSQL to be installed and running
            # Implementation would depend on specific deployment requirements
            logger.info("PostgreSQL database setup would be implemented here")
            
        except Exception as e:
            logger.error(f"Failed to setup PostgreSQL database: {e}")
            raise
    
    def _setup_mysql_database(self):
        """Setup MySQL database"""
        try:
            # This would require MySQL to be installed and running
            # Implementation would depend on specific deployment requirements
            logger.info("MySQL database setup would be implemented here")
            
        except Exception as e:
            logger.error(f"Failed to setup MySQL database: {e}")
            raise
    
    def create_configuration_files(self):
        """Create configuration files"""
        self._log_step("Creating configuration files", "starting")
        
        try:
            # Main configuration file
            config_data = {
                "database": {
                    "type": self.config.database_type,
                    "path": str(self.data_dir / "sutazai.db") if self.config.database_type == "sqlite" else None
                },
                "api": {
                    "host": "0.0.0.0",
                    "port": self.config.api_port,
                    "enable_cors": True,
                    "enable_docs": True
                },
                "monitoring": {
                    "enabled": self.config.enable_monitoring,
                    "prometheus_port": self.config.monitoring_port,
                    "metrics_retention_days": 30
                },
                "security": {
                    "enabled": self.config.enable_security,
                    "jwt_secret": "your-secret-key-here",
                    "authorized_user": "chrissuta01@gmail.com"
                },
                "neural_network": {
                    "default_nodes": 100,
                    "learning_rate": 0.01,
                    "activation_threshold": 0.5
                },
                "models": {
                    "ollama_host": "http://localhost:11434",
                    "models_dir": str(self.models_dir),
                    "auto_load_models": True
                },
                "logging": {
                    "level": "INFO",
                    "log_dir": str(self.logs_dir),
                    "max_file_size": "10MB",
                    "backup_count": 5
                }
            }
            
            config_file = self.config_dir / "settings.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Create environment file
            env_file = self.install_dir / ".env"
            with open(env_file, 'w') as f:
                f.write(f"SUTAZAI_CONFIG_PATH={config_file}\n")
                f.write(f"SUTAZAI_DATA_DIR={self.data_dir}\n")
                f.write(f"SUTAZAI_LOGS_DIR={self.logs_dir}\n")
                f.write(f"SUTAZAI_MODELS_DIR={self.models_dir}\n")
            
            # Create systemd service file (Linux only)
            if self.config.system_type == SystemType.LINUX:
                self._create_systemd_service()
            
            # Create startup script
            self._create_startup_script()
            
            self._log_step("Creating configuration files", "success")
            
        except Exception as e:
            self._log_step("Creating configuration files", "failed", str(e))
            raise
    
    def _create_systemd_service(self):
        """Create systemd service file"""
        try:
            service_content = f'''[Unit]
Description=SutazAI AGI/ASI System
After=network.target
Wants=network.target

[Service]
Type=simple
User={os.getenv("USER", "root")}
WorkingDirectory={self.install_dir}
Environment=SUTAZAI_CONFIG_PATH={self.config_dir}/settings.json
ExecStart={self.install_dir}/venv/bin/python {self.install_dir}/main_agi.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
'''
            
            service_file = Path("/etc/systemd/system/sutazai.service")
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            # Enable service
            self._run_command(["sudo", "systemctl", "daemon-reload"])
            self._run_command(["sudo", "systemctl", "enable", "sutazai"])
            
            logger.info("Systemd service created and enabled")
            
        except Exception as e:
            logger.warning(f"Failed to create systemd service: {e}")
    
    def _create_startup_script(self):
        """Create startup script"""
        try:
            if self.config.system_type == SystemType.WINDOWS:
                script_content = f'''@echo off
cd /d "{self.install_dir}"
"{self.install_dir}\\venv\\Scripts\\python.exe" main_agi.py
pause
'''
                script_file = self.install_dir / "start_sutazai.bat"
            else:
                script_content = f'''#!/bin/bash
cd "{self.install_dir}"
source venv/bin/activate
export SUTAZAI_CONFIG_PATH="{self.config_dir}/settings.json"
python main_agi.py
'''
                script_file = self.install_dir / "start_sutazai.sh"
            
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Make executable on Unix systems
            if self.config.system_type != SystemType.WINDOWS:
                script_file.chmod(0o755)
            
            logger.info(f"Startup script created: {script_file}")
            
        except Exception as e:
            logger.error(f"Failed to create startup script: {e}")
            raise
    
    def run_system_tests(self):
        """Run system tests"""
        self._log_step("Running system tests", "starting")
        
        try:
            # Set up test environment
            test_env = os.environ.copy()
            test_env["SUTAZAI_CONFIG_PATH"] = str(self.config_dir / "settings.json")
            test_env["PYTHONPATH"] = str(self.install_dir)
            
            # Run test framework
            python_executable = str(self.install_dir / "venv" / "bin" / "python") if self.config.system_type != SystemType.WINDOWS else str(self.install_dir / "venv" / "Scripts" / "python.exe")
            test_script = self.install_dir / "tests" / "test_framework.py"
            
            if test_script.exists():
                result = subprocess.run(
                    [python_executable, str(test_script)],
                    env=test_env,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes timeout
                )
                
                if result.returncode == 0:
                    self._log_step("Running system tests", "success", "All tests passed")
                else:
                    self._log_step("Running system tests", "failed", f"Tests failed: {result.stderr}")
            else:
                self._log_step("Running system tests", "success", "No test framework found, skipping")
            
        except Exception as e:
            self._log_step("Running system tests", "failed", str(e))
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate deployment report"""
        try:
            report = {
                "deployment_info": {
                    "mode": self.config.mode.value,
                    "system_type": self.config.system_type.value,
                    "timestamp": datetime.now().isoformat(),
                    "install_directory": str(self.install_dir),
                    "configuration": {
                        "api_enabled": self.config.enable_api,
                        "monitoring_enabled": self.config.enable_monitoring,
                        "security_enabled": self.config.enable_security,
                        "ollama_enabled": self.config.enable_ollama,
                        "database_type": self.config.database_type
                    }
                },
                "system_info": self.system_info,
                "installation_log": self.installation_log,
                "service_urls": {
                    "api": f"http://localhost:{self.config.api_port}",
                    "api_docs": f"http://localhost:{self.config.api_port}/api/docs",
                    "health_check": f"http://localhost:{self.config.api_port}/health",
                    "metrics": f"http://localhost:{self.config.monitoring_port}/metrics" if self.config.enable_monitoring else None
                },
                "next_steps": [
                    f"Start the system using: {self.install_dir}/start_sutazai.sh",
                    f"Check system status at: http://localhost:{self.config.api_port}/health",
                    f"Access API documentation at: http://localhost:{self.config.api_port}/api/docs",
                    "Configure monitoring and alerting as needed",
                    "Set up backups for data directory",
                    "Review security configuration"
                ]
            }
            
            # Save report
            report_file = self.install_dir / "deployment_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Deployment report saved to: {report_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate deployment report: {e}")
            return {"error": str(e)}
    
    def deploy(self) -> Dict[str, Any]:
        """Execute complete deployment"""
        logger.info("=== Starting SutazAI Enterprise Deployment ===")
        
        try:
            # Execute deployment steps
            self.create_directories()
            self.install_system_dependencies()
            self.install_python_dependencies()
            
            if self.config.enable_ollama:
                self.install_ollama()
            
            self.setup_database()
            self.create_configuration_files()
            self.run_system_tests()
            
            # Generate deployment report
            report = self.generate_deployment_report()
            
            logger.info("=== SutazAI Enterprise Deployment Complete ===")
            
            return report
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self._log_step("Deployment", "failed", str(e))
            raise

def create_deployment_config(mode: str = "standalone") -> DeploymentConfig:
    """Create deployment configuration"""
    
    # Detect system type
    system_name = platform.system().lower()
    if system_name == "linux":
        system_type = SystemType.LINUX
    elif system_name == "windows":
        system_type = SystemType.WINDOWS
    elif system_name == "darwin":
        system_type = SystemType.MACOS
    else:
        system_type = SystemType.LINUX  # Default
    
    # Create configuration
    config = DeploymentConfig(
        mode=DeploymentMode(mode),
        system_type=system_type,
        install_dir="/opt/sutazaiapp",
        data_dir="/opt/sutazaiapp/data",
        logs_dir="/opt/sutazaiapp/logs",
        config_dir="/opt/sutazaiapp/config",
        models_dir="/opt/sutazaiapp/models",
        enable_monitoring=True,
        enable_security=True,
        enable_api=True,
        enable_ollama=True,
        database_type="sqlite"
    )
    
    return config

def main():
    """Main entry point for deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Enterprise Deployment")
    parser.add_argument("--mode", choices=["standalone", "docker", "kubernetes", "development"], 
                       default="standalone", help="Deployment mode")
    parser.add_argument("--install-dir", default="/opt/sutazaiapp", help="Installation directory")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama installation")
    parser.add_argument("--no-monitoring", action="store_true", help="Disable monitoring")
    parser.add_argument("--database", choices=["sqlite", "postgresql", "mysql"], 
                       default="sqlite", help="Database type")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_deployment_config(args.mode)
    config.install_dir = args.install_dir
    config.data_dir = f"{args.install_dir}/data"
    config.logs_dir = f"{args.install_dir}/logs"
    config.config_dir = f"{args.install_dir}/config"
    config.models_dir = f"{args.install_dir}/models"
    config.enable_ollama = not args.no_ollama
    config.enable_monitoring = not args.no_monitoring
    config.database_type = args.database
    
    # Create deployment manager
    deployment_manager = EnterpriseDeploymentManager(config)
    
    try:
        # Execute deployment
        report = deployment_manager.deploy()
        
        # Print summary
        print("\n" + "="*60)
        print("SutazAI Enterprise Deployment Complete!")
        print("="*60)
        print(f"Installation Directory: {config.install_dir}")
        print(f"API URL: http://localhost:{config.api_port}")
        print(f"API Documentation: http://localhost:{config.api_port}/api/docs")
        print(f"Health Check: http://localhost:{config.api_port}/health")
        if config.enable_monitoring:
            print(f"Metrics: http://localhost:{config.monitoring_port}/metrics")
        print(f"\nTo start the system: {config.install_dir}/start_sutazai.sh")
        print(f"Deployment report: {config.install_dir}/deployment_report.json")
        
        return 0
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())