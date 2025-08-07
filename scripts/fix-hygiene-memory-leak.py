#!/usr/bin/env python3
"""
Purpose: Emergency fix for hygiene monitoring memory leak
Usage: python fix-hygiene-memory-leak.py
Requirements: psutil, aiohttp
"""

import asyncio
import psutil
import signal
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HygieneMemoryLeakFixer:
    def __init__(self):
        self.monitoring_pids = []
        self.memory_threshold_mb = 500  # Restart if process exceeds 500MB
        self.check_interval = 60  # Check every minute
        
    def find_hygiene_processes(self):
        """Find all hygiene monitoring processes"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
            try:
                info = proc.info
                cmdline = ' '.join(info['cmdline'] or [])
                
                # Check if this is a hygiene monitoring process
                if ('hygiene' in cmdline and 'backend' in cmdline) or \
                   ('monitoring' in cmdline and 'static_monitor' in cmdline):
                    processes.append({
                        'pid': info['pid'],
                        'name': info['name'],
                        'cmdline': cmdline,
                        'memory_mb': info['memory_info'].rss / (1024 * 1024)
                    })
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return processes
    
    def restart_hygiene_backend(self):
        """Restart the hygiene backend service"""
        logger.info("Restarting hygiene backend service...")
        
        try:
            # First, try docker restart
            result = subprocess.run(
                ['docker', 'restart', 'hygiene-backend'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Successfully restarted hygiene-backend container")
                return True
            else:
                logger.error(f"Failed to restart container: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error restarting service: {e}")
            
        return False
    
    def kill_runaway_process(self, pid):
        """Kill a specific process that's using too much memory"""
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            
            # Wait for graceful termination
            try:
                proc.wait(timeout=10)
            except psutil.TimeoutExpired:
                # Force kill if needed
                proc.kill()
                
            logger.info(f"Terminated process {pid}")
            return True
            
        except psutil.NoSuchProcess:
            logger.warning(f"Process {pid} no longer exists")
            return True
        except Exception as e:
            logger.error(f"Failed to kill process {pid}: {e}")
            return False
    
    async def monitor_and_fix(self):
        """Monitor processes and fix memory leaks"""
        logger.info("Starting hygiene memory leak monitor...")
        
        while True:
            try:
                processes = self.find_hygiene_processes()
                
                for proc in processes:
                    memory_mb = proc['memory_mb']
                    pid = proc['pid']
                    
                    logger.info(f"Process {pid} using {memory_mb:.1f}MB - {proc['cmdline'][:60]}...")
                    
                    if memory_mb > self.memory_threshold_mb:
                        logger.warning(f"Process {pid} exceeds memory threshold ({memory_mb:.1f}MB > {self.memory_threshold_mb}MB)")
                        
                        # If it's the backend, restart the container
                        if 'hygiene-backend' in proc['cmdline']:
                            self.restart_hygiene_backend()
                        else:
                            # Otherwise kill the specific process
                            self.kill_runaway_process(pid)
                
                # Also check docker stats
                self.check_docker_memory()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    def check_docker_memory(self):
        """Check Docker container memory usage"""
        try:
            result = subprocess.run([
                'docker', 'stats', '--no-stream', '--format',
                '{{.Container}}\t{{.Name}}\t{{.MemUsage}}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'hygiene' in line:
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            container_id = parts[0]
                            name = parts[1]
                            mem_usage = parts[2]
                            
                            # Parse memory usage
                            if 'GiB' in mem_usage:
                                mem_value = float(mem_usage.split('GiB')[0].strip())
                                if mem_value > 0.5:  # Alert if over 500MB
                                    logger.warning(f"Container {name} using high memory: {mem_usage}")
                                    
        except Exception as e:
            logger.error(f"Error checking docker stats: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, exiting...")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    fixer = HygieneMemoryLeakFixer()
    
    # Show current status
    processes = fixer.find_hygiene_processes()
    logger.info(f"Found {len(processes)} hygiene monitoring processes")
    
    # Start monitoring
    await fixer.monitor_and_fix()

if __name__ == '__main__':
    asyncio.run(main())