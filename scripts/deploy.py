#!/usr/bin/env python3.11
"""
SutazAI Deployment Script

This script handles the deployment process between the Code Server (192.168.100.28)
and Deployment Server (192.168.100.100), including:
- Code synchronization
- Environment setup
- Service deployment
- Health checks
- Rollback capabilities
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("/opt/sutazaiapp/logs/deploy.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SutazAI.Deploy")

class Deployer:
    def __init__(self):
        self.code_server = "192.168.100.28"
        self.deploy_server = "192.168.100.100"
        self.project_root = "/opt/sutazaiapp"
        self.backup_dir = f"{self.project_root}/backups"
        self.deploy_user = "sutazaiapp_dev"
        self.ssh_key = f"/home/{self.deploy_user}/.ssh/sutazai_deploy"
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)

    def validate_environment(self) -> bool:
        """Validate deployment environment."""
        logger.info("Validating deployment environment...")
        
        # Check SSH key
        if not os.path.exists(self.ssh_key):
            logger.error(f"SSH key not found: {self.ssh_key}")
            return False
            
        # Check project directory
        if not os.path.exists(self.project_root):
            logger.error(f"Project directory not found: {self.project_root}")
            return False
            
        # Check virtual environment
        venv_path = f"{self.project_root}/venv"
        if not os.path.exists(venv_path):
            logger.error(f"Virtual environment not found: {venv_path}")
            return False
            
        return True

    def create_backup(self) -> Optional[str]:
        """Create a backup of the current deployment."""
        logger.info("Creating deployment backup...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = f"{self.backup_dir}/{backup_name}"
        
        try:
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            
            # Copy project files
            subprocess.run(
                ["rsync", "-av", "--exclude=venv", "--exclude=__pycache__",
                 f"{self.project_root}/", f"{backup_path}/"],
                check=True,
            )
            
            # Backup database if exists
            if os.path.exists(f"{self.project_root}/data/db"):
                subprocess.run(
                    ["pg_dump", "-U", "sutazai", "sutazai", "-f", f"{backup_path}/db_backup.sql"],
                    check=True,
                )
            
            logger.info(f"Backup created successfully: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None

    def sync_code(self) -> bool:
        """Synchronize code between servers."""
        logger.info("Synchronizing code between servers...")
        
        try:
            # Pull latest changes on code server
            subprocess.run(
                ["git", "-C", self.project_root, "pull", "origin", "main"],
                check=True,
            )
            
            # Sync to deployment server
            subprocess.run(
                ["rsync", "-av", "--delete", "--exclude=venv", "--exclude=__pycache__",
                 f"{self.project_root}/", f"{self.deploy_user}@{self.deploy_server}:{self.project_root}/"],
                check=True,
            )
            
            logger.info("Code synchronization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error synchronizing code: {e}")
            return False

    def setup_deployment_server(self) -> bool:
        """Set up the deployment server environment."""
        logger.info("Setting up deployment server...")
        
        try:
            # Create virtual environment on deployment server
            subprocess.run(
                ["ssh", f"{self.deploy_user}@{self.deploy_server}",
                 f"cd {self.project_root} && python3.11 -m venv venv"],
                check=True,
            )
            
            # Install dependencies
            subprocess.run(
                ["ssh", f"{self.deploy_user}@{self.deploy_server}",
                 f"cd {self.project_root} && source venv/bin/activate && pip install -r requirements.txt"],
                check=True,
            )
            
            # Set up environment variables
            env_file = f"{self.project_root}/.env"
            if os.path.exists(env_file):
                subprocess.run(
                    ["scp", env_file, f"{self.deploy_user}@{self.deploy_server}:{self.project_root}/"],
                    check=True,
                )
            
            logger.info("Deployment server setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up deployment server: {e}")
            return False

    def deploy_services(self) -> bool:
        """Deploy services on the deployment server."""
        logger.info("Deploying services...")
        
        try:
            # Stop existing services
            subprocess.run(
                ["ssh", f"{self.deploy_user}@{self.deploy_server}",
                 "sudo systemctl stop sutazai-backend sutazai-web"],
                check=True,
            )
            
            # Deploy backend
            subprocess.run(
                ["ssh", f"{self.deploy_user}@{self.deploy_server}",
                 f"cd {self.project_root} && source venv/bin/activate && python -m backend.main"],
                check=True,
            )
            
            # Deploy web UI
            subprocess.run(
                ["ssh", f"{self.deploy_user}@{self.deploy_server}",
                 f"cd {self.project_root}/web_ui && npm install && npm run build"],
                check=True,
            )
            
            # Start services
            subprocess.run(
                ["ssh", f"{self.deploy_user}@{self.deploy_server}",
                 "sudo systemctl start sutazai-backend sutazai-web"],
                check=True,
            )
            
            logger.info("Services deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying services: {e}")
            return False

    def run_health_checks(self) -> bool:
        """Run health checks on deployed services."""
        logger.info("Running health checks...")
        
        try:
            # Check backend health
            result = subprocess.run(
                ["curl", "-f", f"http://{self.deploy_server}:8000/health"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("Backend health check failed")
                return False
                
            # Check web UI
            result = subprocess.run(
                ["curl", "-f", f"http://{self.deploy_server}:3000"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("Web UI health check failed")
                return False
                
            logger.info("Health checks passed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running health checks: {e}")
            return False

    def rollback(self, backup_path: str) -> bool:
        """Rollback to a previous deployment."""
        logger.info(f"Rolling back to backup: {backup_path}")
        
        try:
            # Stop services
            subprocess.run(
                ["ssh", f"{self.deploy_user}@{self.deploy_server}",
                 "sudo systemctl stop sutazai-backend sutazai-web"],
                check=True,
            )
            
            # Restore files
            subprocess.run(
                ["rsync", "-av", "--delete",
                 f"{backup_path}/", f"{self.deploy_user}@{self.deploy_server}:{self.project_root}/"],
                check=True,
            )
            
            # Restore database if backup exists
            if os.path.exists(f"{backup_path}/db_backup.sql"):
                subprocess.run(
                    ["scp", f"{backup_path}/db_backup.sql",
                     f"{self.deploy_user}@{self.deploy_server}:{self.project_root}/"],
                    check=True,
                )
                subprocess.run(
                    ["ssh", f"{self.deploy_user}@{self.deploy_server}",
                     f"psql -U sutazai sutazai < {self.project_root}/db_backup.sql"],
                    check=True,
                )
            
            # Start services
            subprocess.run(
                ["ssh", f"{self.deploy_user}@{self.deploy_server}",
                 "sudo systemctl start sutazai-backend sutazai-web"],
                check=True,
            )
            
            logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False

    def deploy(self) -> bool:
        """Run the complete deployment process."""
        logger.info("Starting deployment process...")
        
        if not self.validate_environment():
            return False
            
        backup_path = self.create_backup()
        if not backup_path:
            return False
            
        if not self.sync_code():
            return False
            
        if not self.setup_deployment_server():
            return False
            
        if not self.deploy_services():
            return False
            
        if not self.run_health_checks():
            logger.error("Deployment failed health checks, initiating rollback...")
            return self.rollback(backup_path)
            
        logger.info("Deployment completed successfully")
        return True

def main():
    """Main entry point."""
    deployer = Deployer()
    success = deployer.deploy()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 