#!/usr/bin/env python3
import logging
import subprocess
import time
from typing import List, Tuple

from setup_directories import create_directory_structure
from error_recovery import rollback_deployment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEPENDENCIES = ["python3.11", "postgresql", "uvicorn", "streamlit"]
SERVICES = [
    ("model_server", "uvicorn backend.model_server:app --host 0.0.0.0 --port 8001"),
    ("api_server", "uvicorn backend.api_routes:app --host 0.0.0.0 --port 8000"),
    ("frontend", "streamlit run frontend/app.py --server.port 8501")
]

def check_dependencies():
    """Verify system dependencies"""
    missing = []
    for dep in DEPENDENCIES:
        try:
            subprocess.run(["which", dep], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            missing.append(dep)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        raise RuntimeError(f"Missing required dependencies: {', '.join(missing)}")
    
    logger.info("All dependencies satisfied")

def deploy_services():
    """Deploy all system services"""
    logger.info("Deploying system services")
    
    for name, cmd in SERVICES:
        try:
            # Convert command string to list for security
            cmd_parts = cmd.split()
            subprocess.Popen(
                cmd_parts, 
                stdout=open(f"/opt/SUTAZAI/logs/{name}.log", "a"),
                stderr=subprocess.STDOUT
            )
            logger.info(f"Started {name} service")
            time.sleep(2)  # Stagger service starts
        except Exception as e:
            logger.error(f"Failed to start {name}: {str(e)}")
            raise

def main():
    try:
        logger.info("Starting deployment...")
        check_dependencies()
        create_directory_structure()
        deploy_services()
        logger.info("Deployment completed successfully")
    except Exception as e:
        logger.critical(f"Deployment failed: {str(e)}")
        rollback_deployment()
        exit(1)

if __name__ == "__main__":
    main()
