#!/usr/bin/env python3
"""
Master Monitoring Script - Orchestrates all monitoring operations
Usage: python monitoring-master.py [health|alerts|logs|all]
"""

import os
import sys
import subprocess
from datetime import datetime

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] MONITOR: {message}")

def run_scripts(category):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    category_dir = os.path.join(script_dir, category)
    
    if not os.path.exists(category_dir):
        log(f"Category directory not found: {category}")
        return
    
    for script in os.listdir(category_dir):
        script_path = os.path.join(category_dir, script)
        if os.path.isfile(script_path) and os.access(script_path, os.X_OK):
            log(f"Running {script}")
            try:
                subprocess.run([script_path], check=True)
            except subprocess.CalledProcessError as e:
                log(f"Error running {script}: {e}")

def main():
    operation = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    log(f"Starting monitoring operation: {operation}")
    
    if operation == "health":
        run_scripts("health-checks")
    elif operation == "alerts":
        run_scripts("alerts")
    elif operation == "logs":
        run_scripts("logging")
    elif operation == "all":
        for category in ["health-checks", "alerts", "logging"]:
            run_scripts(category)
    else:
        log(f"Unknown operation: {operation}")
        sys.exit(1)
    
    log(f"Monitoring operation complete: {operation}")

if __name__ == "__main__":
    main()
