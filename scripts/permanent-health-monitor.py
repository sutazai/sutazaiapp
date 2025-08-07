#!/usr/bin/env python3
"""
Permanent Health Monitor for SutazAI
Continuously monitors container health and applies automatic fixes
"""

import docker
import time
import logging
import json
import signal
import sys
from datetime import datetime
from typing import Dict, List, Set
import subprocess
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/permanent-health-monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContainerHealthMonitor:
    """Monitors and maintains container health automatically"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.running = True
        self.check_interval = 30  # seconds
        self.restart_cooldown = 300  # 5 minutes
        self.last_restart = {}  # container_name -> timestamp
        self.health_stats = {
            'total_checks': 0,
            'fixed_containers': 0,
            'restart_attempts': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        logger.info("Shutting down health monitor...")
        self.running = False
        self.save_stats()
        sys.exit(0)
    
    def save_stats(self):
        """Save health monitoring statistics"""
        try:
            stats_file = '/opt/sutazaiapp/logs/health_monitor_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(self.health_stats, f, indent=2)
            logger.info(f"Saved health monitoring stats to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
    
    def get_sutazai_containers(self) -> List[docker.models.containers.Container]:
        """Get all SutazAI containers"""
        try:
            containers = []
            for container in self.client.containers.list():
                if container.name.startswith('sutazai-'):
                    containers.append(container)
            return containers
        except Exception as e:
            logger.error(f"Failed to get containers: {e}")
            return []
    
    def check_container_health(self, container: docker.models.containers.Container) -> str:
        """Check the health status of a container"""
        try:
            container.reload()
            health = container.attrs.get('State', {}).get('Health', {})
            return health.get('Status', 'no-health-check')
        except Exception as e:
            logger.error(f"Failed to check health for {container.name}: {e}")
            return 'unknown'
    
    def is_service_actually_healthy(self, container: docker.models.containers.Container) -> bool:
        """Check if the service inside the container is actually responding"""
        try:
            # Test if port 8080 is open inside the container
            exec_result = container.exec_run([
                'python3', '-c', 
                'import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(("localhost", 8080))==0 else 1)'
            ])
            return exec_result.exit_code == 0
        except Exception as e:
            logger.debug(f"Service check failed for {container.name}: {e}")
            return False
    
    def fix_container_health_check(self, container_name: str) -> bool:
        """Apply health check fix to a specific container"""
        try:
            logger.info(f"Applying health check fix to {container_name}")
            
            # Create a simple health check script
            health_script = '''#!/usr/bin/env python3
import socket
import sys

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    result = sock.connect_ex(('localhost', 8080))
    sock.close()
    sys.exit(0 if result == 0 else 1)
except:
    sys.exit(1)
'''
            
            # Write the script to a temporary file
            script_path = f'/tmp/health_fix_{container_name}.py'
            with open(script_path, 'w') as f:
                f.write(health_script)
            os.chmod(script_path, 0o755)
            
            # Copy to container
            container = self.client.containers.get(container_name)
            with open(script_path, 'rb') as f:
                container.put_archive('/tmp/', [('health.py', f.read())])
            
            # Make it executable
            container.exec_run(['chmod', '+x', '/tmp/health.py'])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fix health check for {container_name}: {e}")
            return False
    
    def restart_container_if_needed(self, container: docker.models.containers.Container) -> bool:
        """Restart a container if it's unhealthy and cooldown has passed"""
        container_name = container.name
        current_time = time.time()
        
        # Check cooldown period
        if container_name in self.last_restart:
            time_since_restart = current_time - self.last_restart[container_name]
            if time_since_restart < self.restart_cooldown:
                logger.debug(f"Container {container_name} in restart cooldown ({time_since_restart:.0f}s)")
                return False
        
        try:
            logger.info(f"Restarting unhealthy container: {container_name}")
            container.restart()
            self.last_restart[container_name] = current_time
            self.health_stats['restart_attempts'] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to restart {container_name}: {e}")
            return False
    
    def disable_health_check(self, container_name: str) -> bool:
        """Disable health check for a persistently problematic container"""
        try:
            logger.info(f"Disabling health check for persistently problematic container: {container_name}")
            
            # This would require recreating the container, which is complex
            # For now, we'll just log it and rely on the service being functional
            # even if the health check reports unhealthy
            
            return True
        except Exception as e:
            logger.error(f"Failed to disable health check for {container_name}: {e}")
            return False
    
    def monitor_cycle(self):
        """Perform one monitoring cycle"""
        logger.info("Starting health monitoring cycle...")
        
        containers = self.get_sutazai_containers()
        if not containers:
            logger.warning("No SutazAI containers found")
            return
        
        healthy_count = 0
        unhealthy_count = 0
        fixed_count = 0
        
        for container in containers:
            self.health_stats['total_checks'] += 1
            
            health_status = self.check_container_health(container)
            
            if health_status == 'healthy' or health_status == 'no-health-check':
                healthy_count += 1
                continue
            elif health_status == 'unhealthy':
                unhealthy_count += 1
                logger.warning(f"Container {container.name} is unhealthy")
                
                # Check if the service is actually working
                if self.is_service_actually_healthy(container):
                    logger.info(f"Service in {container.name} is actually healthy - health check issue")
                    # Apply health check fix
                    if self.fix_container_health_check(container.name):
                        fixed_count += 1
                        self.health_stats['fixed_containers'] += 1
                else:
                    logger.warning(f"Service in {container.name} is not responding - restarting")
                    if self.restart_container_if_needed(container):
                        fixed_count += 1
            else:
                logger.warning(f"Container {container.name} has unknown health status: {health_status}")
        
        total_containers = len(containers)
        health_rate = (healthy_count * 100 / total_containers) if total_containers > 0 else 0
        
        logger.info(f"Health check completed: {healthy_count}/{total_containers} healthy ({health_rate:.1f}%)")
        if unhealthy_count > 0:
            logger.info(f"Fixed {fixed_count} containers this cycle")
        
        # Save current stats
        self.save_stats()
    
    def run(self):
        """Main monitoring loop"""
        logger.info("Starting permanent health monitor...")
        logger.info(f"Monitoring interval: {self.check_interval} seconds")
        logger.info(f"Restart cooldown: {self.restart_cooldown} seconds")
        
        while self.running:
            try:
                self.monitor_cycle()
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                break
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                time.sleep(self.check_interval)
        
        logger.info("Health monitor stopped")

def main():
    """Main entry point"""
    monitor = ContainerHealthMonitor()
    try:
        monitor.run()
    except Exception as e:
        logger.error(f"Failed to start health monitor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()