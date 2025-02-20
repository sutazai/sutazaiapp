#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import json
import shutil
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Imports using the added path
from config_manager import SutazAIConfigManager
from advanced_monitoring import SutazAIMonitor
from ssh_key_manager import SutazAISSHKeyManager

class SutazAISystemSyncOptimizer:
    def __init__(self, project_root="/opt/sutazai_project/SutazAI"):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(project_root, 'logs', 'system_sync.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.project_root = project_root
        self.config_manager = SutazAIConfigManager()
        self.monitor = SutazAIMonitor()
        self.ssh_key_manager = SutazAISSHKeyManager()

        # Synchronization configuration
        self.sync_config = self.config_manager.get_config().get('sync', {})

    def comprehensive_system_check(self):
        """
        Perform a comprehensive system-wide check and optimization
        """
        self.logger.info("🔍 Starting Comprehensive System Check and Optimization")

        try:
            # System Health Monitoring
            system_metrics = self.monitor.get_system_metrics()
            self.logger.info(f"System Metrics: {json.dumps(system_metrics, indent=2)}")

            # Check and optimize critical resources
            self._optimize_resources(system_metrics)

            # Validate and repair file system
            self._validate_file_system()

            # Check dependencies
            self._check_dependencies()

            # Perform security hardening
            self._security_hardening()

            self.logger.info("✅ Comprehensive System Check Completed Successfully")
        except Exception as e:
            self.logger.error(f"System Check Failed: {e}")
            self.monitor.send_notification(
                "System Check Failure", 
                f"Comprehensive system check encountered an error: {e}", 
                is_critical=True
            )

    def _optimize_resources(self, system_metrics):
        """
        Optimize system resources based on current metrics
        """
        try:
            # CPU Optimization
            if system_metrics['cpu']['usage'] > 80:
                self.logger.warning("High CPU Usage Detected - Initiating Optimization")
                subprocess.run(['nice', '-n', '19', 'python3', '-m', 'compileall', self.project_root])

            # Memory Optimization
            if system_metrics['memory']['percent_used'] > 85:
                self.logger.warning("High Memory Usage Detected - Clearing Caches")
                subprocess.run(['sync'])
                subprocess.run(['echo', '3', '>', '/proc/sys/vm/drop_caches'])

            # Disk Space Optimization
            if system_metrics['disk']['percent_used'] > 85:
                self.logger.warning("Low Disk Space - Cleaning Temporary Files")
                self._clean_temporary_files()

        except Exception as e:
            self.logger.error(f"Resource Optimization Failed: {e}")

    def _validate_file_system(self):
        """
        Validate and repair file system integrity
        """
        try:
            # Check for broken symlinks
            for root, _, files in os.walk(self.project_root):
                for file in files:
                    filepath = os.path.join(root, file)
                    if os.path.islink(filepath) and not os.path.exists(os.readlink(filepath)):
                        self.logger.warning(f"Broken symlink found: {filepath}")
                        os.unlink(filepath)

            # Repair file permissions
            subprocess.run(['find', self.project_root, '-type', 'd', '-exec', 'chmod', '755', '{}', '+'])
            subprocess.run(['find', self.project_root, '-type', 'f', '-exec', 'chmod', '644', '{}', '+'])

        except Exception as e:
            self.logger.error(f"File System Validation Failed: {e}")

    def _check_dependencies(self):
        """
        Check and update project dependencies
        """
        try:
            # Update pip packages
            subprocess.run([sys.executable, '-m', 'pip', 'list', '--outdated'])
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
            
            # Update requirements
            requirements_file = os.path.join(self.project_root, 'requirements.txt')
            if os.path.exists(requirements_file):
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file, '--upgrade'])

        except Exception as e:
            self.logger.error(f"Dependency Check Failed: {e}")

    def _security_hardening(self):
        """
        Implement basic security hardening measures
        """
        try:
            # Rotate SSH keys periodically
            for server in ['code_server', 'deploy_server']:
                self.ssh_key_manager.rotate_ssh_keys(server)

            # Remove old log files
            log_dir = os.path.join(self.project_root, 'logs')
            for log_file in os.listdir(log_dir):
                log_path = os.path.join(log_dir, log_file)
                if os.path.isfile(log_path):
                    file_age = time.time() - os.path.getctime(log_path)
                    if file_age > 30 * 24 * 3600:  # 30 days
                        os.unlink(log_path)

        except Exception as e:
            self.logger.error(f"Security Hardening Failed: {e}")

    def _clean_temporary_files(self):
        """
        Clean up temporary and unnecessary files
        """
        try:
            # Remove Python cache files
            subprocess.run(['find', self.project_root, '-type', 'd', '-name', '__pycache__', '-exec', 'rm', '-rf', '{}', '+'])
            subprocess.run(['find', self.project_root, '-type', 'f', '-name', '*.pyc', '-delete'])
            subprocess.run(['find', self.project_root, '-type', 'f', '-name', '*.pyo', '-delete'])
            subprocess.run(['find', self.project_root, '-type', 'f', '-name', '*.pyd', '-delete'])

        except Exception as e:
            self.logger.error(f"Temporary File Cleanup Failed: {e}")

    def synchronize_servers(self):
        """
        Synchronize code between servers
        """
        try:
            # Get server configurations
            servers = self.config_manager.get_config().get('servers', {})
            code_server = servers.get('code_server', {})
            deploy_server = servers.get('deploy_server', {})

            # Perform synchronization
            exclude_patterns = self.sync_config.get('exclude_patterns', [])
            rsync_command = [
                'rsync', '-avz', '--delete',
                *[f'--exclude={pattern}' for pattern in exclude_patterns],
                f'{code_server.get("hostname")}:{self.project_root}/',
                f'{deploy_server.get("hostname")}:{self.project_root}/'
            ]

            result = subprocess.run(rsync_command, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Server Synchronization Successful")
            else:
                error_msg = f"Server Synchronization Failed: {result.stderr}"
                self.logger.error(error_msg)
                self.monitor.send_notification("Sync Failure", error_msg, is_critical=True)

        except Exception as e:
            error_msg = f"Server Synchronization Error: {e}"
            self.logger.error(error_msg)
            self.monitor.send_notification("Sync Error", error_msg, is_critical=True)

def main():
    sync_optimizer = SutazAISystemSyncOptimizer()
    
    try:
        # Perform comprehensive system check
        sync_optimizer.comprehensive_system_check()

        # Synchronize servers
        sync_optimizer.synchronize_servers()

        # Send success notification
        sync_optimizer.monitor.send_notification(
            "System Sync and Optimization Complete", 
            "SutazAI system synchronization and optimization finished successfully."
        )

    except Exception as e:
        sync_optimizer.logger.error(f"System Sync and Optimization Failed: {e}")
        sync_optimizer.monitor.send_notification(
            "System Sync Failure", 
            f"Comprehensive system synchronization failed: {e}", 
            is_critical=True
        )

if __name__ == "__main__":
    main() 