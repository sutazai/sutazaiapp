#!/usr/bin/env python3
"""
Performance Issue Resolver for SutazAI

This script identifies and resolves common performance issues in the SutazAI system.
Key features:
- Identifies zombie processes and leftover bash processes
- Kills runaway bash processes with high CPU usage
- Ensures proper shutdown of backend and related processes
- Cleans up temporary files and caches
"""

import os
import sys
import time
import signal
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/opt/sutazaiapp/logs/performance_fix.log"),
    ],
)
logger = logging.getLogger(__name__)

def kill_high_cpu_processes():
    """Find and kill processes with excessive CPU usage."""
    logger.info("Checking for high CPU usage processes...")
    
    try:
        # Find processes with high CPU usage
        high_cpu_cmd = "ps -eo pid,pcpu,pmem,user,args --sort=-pcpu | head -n 10"
        result = subprocess.run(high_cpu_cmd, shell=True, capture_output=True, text=True)
        logger.info(f"Top CPU processes:\n{result.stdout}")
        
        # Look for bash processes with high CPU
        bash_cmd = "ps aux | grep bash | grep -v grep"
        result = subprocess.run(bash_cmd, shell=True, capture_output=True, text=True)
        
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
                
            pid = parts[1]
            cpu_usage = float(parts[2])
            
            # Kill bash processes with high CPU usage
            if cpu_usage > 80.0:
                logger.info(f"Killing high CPU bash process: {pid} (CPU: {cpu_usage}%)")
                subprocess.run(f"kill -9 {pid}", shell=True)
                
    except Exception as e:
        logger.error(f"Error killing high CPU processes: {e}")

def restart_backend_service():
    """Stop and restart the backend service."""
    logger.info("Restarting backend service...")
    
    try:
        # Kill any existing backend processes
        subprocess.run("pkill -f 'python -m backend.backend_main'", shell=True)
        time.sleep(2)  # Allow time for processes to terminate
        
        # Start backend service in background
        logger.info("Starting backend service...")
        subprocess.Popen(
            "cd /opt/sutazaiapp && python -m backend.backend_main > /opt/sutazaiapp/logs/backend_restart.log 2>&1",
            shell=True, 
            start_new_session=True
        )
        
    except Exception as e:
        logger.error(f"Error restarting backend service: {e}")

def clean_caches():
    """Clean up temporary files and caches."""
    logger.info("Cleaning caches and temporary files...")
    
    try:
        cache_dirs = [
            "/opt/sutazaiapp/.mypy_cache",
            "/opt/sutazaiapp/.ruff_cache",
            "/tmp/sutazai*",
        ]
        
        for cache_dir in cache_dirs:
            subprocess.run(f"rm -rf {cache_dir}", shell=True)
            
        logger.info("Cache cleanup completed")
        
    except Exception as e:
        logger.error(f"Error cleaning caches: {e}")

def main():
    """Main entry point for the performance issue resolver."""
    logger.info("Starting performance issue resolver...")
    
    # Kill high CPU processes
    kill_high_cpu_processes()
    
    # Clean caches
    clean_caches()
    
    # Restart backend service
    restart_backend_service()
    
    logger.info("Performance issue resolution completed")
    
if __name__ == "__main__":
    main() 