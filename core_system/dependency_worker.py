#!/usr/bin/env python3
"""
SutazAI Dependency Management Background Worker

Continuously monitors and manages project dependencies
in an autonomous, intelligent manner.
"""

import os
import sys
import time
import logging
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dependency_manager import AutonomousDependencyManager
from workers.base_worker import SutazAiWorker

class DependencyManagementWorker(SutazAiWorker):
    """
    Autonomous background worker for continuous dependency management
    
    Responsibilities:
    - Periodic dependency updates
    - Vulnerability scanning
    - Dependency health monitoring
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        interval: int = 86400,  # 24 hours
        log_dir: Optional[str] = None
    ):
        """
        Initialize Dependency Management Worker
        
        Args:
            base_dir (str): Base project directory
            interval (int): Interval between dependency checks (seconds)
            log_dir (Optional[str]): Custom log directory
        """
        super().__init__("DependencyManagementWorker", interval)
        
        # Configure logging
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(base_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
            filename=os.path.join(self.log_dir, 'dependency_worker.log')
        )
        self.logger = logging.getLogger('SutazAI.DependencyWorker')
        
        # Initialize dependency manager
        self.dependency_manager = AutonomousDependencyManager(
            base_dir=base_dir,
            log_dir=self.log_dir
        )
    
    def execute(self):
        """
        Primary execution method for dependency management
        
        Performs:
        - Dependency update checks
        - Vulnerability scanning
        - Dependency health assessment
        """
        try:
            self.logger.info("Starting autonomous dependency management cycle")
            
            # Run comprehensive dependency management
            self.dependency_manager.autonomous_dependency_management()
            
            # Additional logging and tracking
            self.log_dependency_status()
        
        except Exception as e:
            self.logger.error(f"Dependency management cycle failed: {e}")
    
    def log_dependency_status(self):
        """
        Log detailed dependency management status
        """
        try:
            # Get current installed packages
            installed_packages = self.dependency_manager.get_installed_packages()
            
            # Log package count and details
            self.logger.info(f"Total installed packages: {len(installed_packages)}")
            
            # Optional: More detailed logging
            for pkg, ver in installed_packages.items():
                self.logger.debug(f"Package: {pkg}, Version: {ver}")
        
        except Exception as e:
            self.logger.warning(f"Failed to log dependency status: {e}")
    
    def run_continuous_monitoring(self):
        """
        Run continuous dependency monitoring
        Allows for more granular control and potential dynamic interval adjustment
        """
        try:
            while True:
                self.execute()
                
                # Sleep for configured interval
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            self.logger.info("Dependency worker gracefully interrupted")
        except Exception as e:
            self.logger.error(f"Continuous monitoring failed: {e}")
            sys.exit(1)

def main():
    """
    Main entry point for dependency management worker
    """
    try:
        # Initialize and start dependency management worker
        dependency_worker = DependencyManagementWorker()
        dependency_worker.run_continuous_monitoring()
    
    except Exception as e:
        print(f"Dependency worker initialization failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 