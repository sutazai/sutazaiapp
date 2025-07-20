#!/usr/bin/env python3
"""
Enterprise System Monitor for SutazAI
Real-time monitoring with automatic memory optimization and recovery
"""

import time
import psutil
import logging
import json
import subprocess
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import threading
import signal
import sys

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/system_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Docker client
try:
    import docker
    docker_client = docker.from_env()
except Exception as e:
    logger.warning(f"Docker client unavailable: {e}")
    docker_client = None

class EnterpriseSystemMonitor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'monitor_interval': int(os.getenv('MONITOR_INTERVAL', 30)),
            'memory_threshold': int(os.getenv('MEMORY_THRESHOLD', 80)),
            'restart_threshold': int(os.getenv('RESTART_THRESHOLD', 90)),
            'cleanup_threshold': int(os.getenv('CLEANUP_THRESHOLD', 85)),
            'max_restarts': 3,
            'restart_window': 300  # 5 minutes
        }
        
        self.monitoring = True
        self.restart_count = {}
        self.last_cleanup = datetime.now()
        self.shutdown_event = threading.Event()
        
        # Service endpoints
        self.services = {
            'ollama': 'http://ollama:11434/api/version',
            'backend': 'http://sutazai-backend:8001/health',
            'frontend': 'http://streamlit-frontend:8501/_stcore/health',
            'qdrant': 'http://qdrant:6333/health',
            'postgres': None,  # Will check via psutil
            'redis': None      # Will check via psutil
        }
        
        # Create logs directory
        os.makedirs('/app/logs', exist_ok=True)
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.monitoring = False
        self.shutdown_event.set()
        sys.exit(0)
    
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        swap = psutil.swap_memory()
        
        # Get container stats if available
        container_stats = self.get_container_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free,
                'buffers': getattr(memory, 'buffers', 0),
                'cached': getattr(memory, 'cached', 0)
            },
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count(),
                'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None,
                'per_cpu': psutil.cpu_percent(percpu=True)
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'swap': {
                'total': swap.total,
                'used': swap.used,
                'free': swap.free,
                'percent': swap.percent
            },
            'containers': container_stats,
            'processes': self.get_top_processes()
        }

    def get_container_stats(self) -> Dict:
        """Get Docker container resource usage"""
        if not docker_client:
            return {}
        
        stats = {}
        try:
            containers = docker_client.containers.list()
            for container in containers:
                try:
                    # Get container stats
                    container_stats = container.stats(stream=False)
                    
                    # Calculate memory usage percentage
                    memory_usage = container_stats['memory_stats'].get('usage', 0)
                    memory_limit = container_stats['memory_stats'].get('limit', 0)
                    memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
                    
                    # Calculate CPU usage
                    cpu_delta = container_stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               container_stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = container_stats['cpu_stats']['system_cpu_usage'] - \
                                  container_stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * len(container_stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0 if system_delta > 0 else 0
                    
                    stats[container.name] = {
                        'memory_usage': memory_usage,
                        'memory_limit': memory_limit,
                        'memory_percent': round(memory_percent, 2),
                        'cpu_percent': round(cpu_percent, 2),
                        'status': container.status
                    }
                except Exception as e:
                    logger.warning(f"Failed to get stats for container {container.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to get container stats: {e}")
            
        return stats
    
    def get_top_processes(self, limit: int = 10) -> List[Dict]:
        """Get top memory consuming processes"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by memory usage and return top N
        processes.sort(key=lambda x: x['memory_percent'] or 0, reverse=True)
        return processes[:limit]
    
    def check_service_health(self) -> Dict[str, bool]:
        """Check health of all services"""
        health_status = {}
        
        for service_name, endpoint in self.services.items():
            try:
                if endpoint:
                    response = requests.get(endpoint, timeout=10)
                    health_status[service_name] = response.status_code == 200
                else:
                    # Check via process name for services without HTTP endpoints
                    health_status[service_name] = any(
                        service_name.lower() in proc.name().lower() 
                        for proc in psutil.process_iter(['name'])
                    )
            except Exception as e:
                logger.warning(f"Health check failed for {service_name}: {e}")
                health_status[service_name] = False
        
        return health_status
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is above threshold"""
        memory = psutil.virtual_memory()
        
        if memory.percent > self.config['memory_threshold']:
            logger.warning(f"High memory usage detected: {memory.percent}%")
            
            # Log top memory consuming processes
            top_processes = self.get_top_processes(5)
            logger.info("Top memory consuming processes:")
            for proc in top_processes:
                logger.info(f"  {proc['name']} (PID: {proc['pid']}): {proc['memory_percent']:.1f}%")
            
            # Check container memory usage
            container_stats = self.get_container_stats()
            high_memory_containers = [
                name for name, stats in container_stats.items() 
                if stats['memory_percent'] > 80
            ]
            
            if high_memory_containers:
                logger.warning(f"High memory containers: {high_memory_containers}")
            
            return True
        return False

    def cleanup_memory(self) -> bool:
        """Perform aggressive memory cleanup"""
        try:
            logger.info("Starting aggressive memory cleanup...")
            
            # Clear system caches
            try:
                subprocess.run(['sync'], check=True, timeout=30)
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('3')
                logger.info("System caches cleared")
            except Exception as e:
                logger.warning(f"Failed to clear system caches: {e}")
            
            # Unload Ollama models if memory is critically low
            memory = psutil.virtual_memory()
            if memory.percent > self.config['cleanup_threshold']:
                self.unload_ollama_models()
            
            # Clean up Docker containers if available
            if docker_client:
                try:
                    docker_client.containers.prune()
                    docker_client.images.prune(filters={'dangling': True})
                    logger.info("Docker cleanup completed")
                except Exception as e:
                    logger.warning(f"Docker cleanup failed: {e}")
            
            # Clean up Python bytecode
            try:
                subprocess.run(
                    ['find', '/app', '-name', '__pycache__', '-type', 'd', '-exec', 'rm', '-rf', '{}', '+'], 
                    capture_output=True, timeout=60
                )
                subprocess.run(
                    ['find', '/app', '-name', '*.pyc', '-delete'], 
                    capture_output=True, timeout=60
                )
            except Exception as e:
                logger.warning(f"Python cleanup failed: {e}")
            
            # Clean up old log files
            self.cleanup_old_logs()
            
            self.last_cleanup = datetime.now()
            logger.info("Memory cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False

    def unload_ollama_models(self) -> bool:
        """Unload all Ollama models to free memory"""
        try:
            logger.info("Unloading Ollama models to free memory...")
            
            # Get list of loaded models
            response = requests.get('http://ollama:11434/api/ps', timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                
                for model in models:
                    model_name = model.get('name')
                    if model_name:
                        # Unload model by setting keep_alive to 0
                        unload_data = {
                            'model': model_name,
                            'keep_alive': 0
                        }
                        requests.post(
                            'http://ollama:11434/api/generate',
                            json=unload_data,
                            timeout=30
                        )
                        logger.info(f"Unloaded model: {model_name}")
                
                return True
            
        except Exception as e:
            logger.warning(f"Failed to unload Ollama models: {e}")
        
        return False
    
    def cleanup_old_logs(self, max_age_days: int = 7) -> None:
        """Clean up old log files"""
        log_dir = Path('/app/logs')
        if not log_dir.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for log_file in log_dir.glob('*.log*'):
            try:
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    log_file.unlink()
                    logger.info(f"Removed old log file: {log_file}")
            except Exception as e:
                logger.warning(f"Failed to remove log file {log_file}: {e}")
    
    def restart_container(self, container_name: str) -> bool:
        """Restart a specific Docker container"""
        if not docker_client:
            return False
        
        try:
            container = docker_client.containers.get(container_name)
            
            # Check restart limits
            restart_key = f"{container_name}_{datetime.now().strftime('%Y%m%d_%H')}"
            current_restarts = self.restart_count.get(restart_key, 0)
            
            if current_restarts >= self.config['max_restarts']:
                logger.error(f"Maximum restart limit reached for {container_name}")
                return False
            
            logger.info(f"Restarting container: {container_name}")
            container.restart(timeout=30)
            
            # Update restart count
            self.restart_count[restart_key] = current_restarts + 1
            
            # Wait for container to be healthy
            for _ in range(30):
                container.reload()
                if container.status == 'running':
                    logger.info(f"Container {container_name} restarted successfully")
                    return True
                time.sleep(2)
            
            logger.warning(f"Container {container_name} may not have started properly")
            return False
            
        except Exception as e:
            logger.error(f"Failed to restart container {container_name}: {e}")
            return False
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a service using docker-compose"""
        # Try container restart first
        if self.restart_container(service_name):
            return True
        
        # Fallback to docker-compose restart
        try:
            current_time = datetime.now()
            restart_key = f"{service_name}_{current_time.strftime('%H')}"
            
            if restart_key in self.restart_count:
                if self.restart_count[restart_key] >= self.config['max_restarts']:
                    logger.error(f"Maximum restart limit reached for {service_name}")
                    return False
                self.restart_count[restart_key] += 1
            else:
                self.restart_count[restart_key] = 1
            
            logger.info(f"Restarting service via docker-compose: {service_name}")
            
            # Service name mapping
            compose_services = {
                'ollama': 'ollama',
                'streamlit': 'streamlit-frontend',
                'frontend': 'streamlit-frontend',
                'backend': 'sutazai-backend',
                'qdrant': 'qdrant',
                'postgres': 'postgresql',
                'postgresql': 'postgresql',
                'redis': 'redis'
            }
            
            compose_service = compose_services.get(service_name, service_name)
            
            result = subprocess.run(
                ['docker-compose', '-f', 'docker-compose-stable.yml', 'restart', compose_service],
                cwd='/opt/sutazaiapp',
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info(f"Service {service_name} restarted successfully")
                return True
            else:
                logger.error(f"Failed to restart {service_name}: {result.stderr}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")
            return False

    def handle_critical_memory(self) -> bool:
        """Handle critical memory situations"""
        logger.critical("Critical memory situation detected!")
        
        # 1. Unload all Ollama models immediately
        self.unload_ollama_models()
        
        # 2. Restart high-memory containers
        container_stats = self.get_container_stats()
        for name, stats in container_stats.items():
            if stats['memory_percent'] > 80:
                logger.warning(f"Restarting high-memory container: {name}")
                self.restart_container(name)
        
        # 3. Emergency cleanup
        self.cleanup_memory()
        
        # 4. Check if we need to restart the frontend (often memory-hungry)
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logger.critical("Restarting frontend due to critical memory")
            self.restart_service('frontend')
        
        return memory.percent < 85
    
    def monitor_system(self) -> None:
        """Main monitoring loop"""
        logger.info("Starting enterprise system monitoring...")
        logger.info(f"Configuration: {self.config}")
        
        while self.monitoring and not self.shutdown_event.is_set():
            try:
                # Get comprehensive system metrics
                metrics = self.get_system_metrics()
                
                # Log metrics to structured file
                metrics_file = Path('/app/logs/metrics.jsonl')
                with open(metrics_file, 'a') as f:
                    json.dump(metrics, f)
                    f.write('\n')
                
                # Check service health
                health_status = self.check_service_health()
                unhealthy_services = [name for name, healthy in health_status.items() if not healthy]
                
                if unhealthy_services:
                    logger.warning(f"Unhealthy services detected: {unhealthy_services}")
                    for service in unhealthy_services:
                        if service in ['ollama', 'backend', 'frontend']:
                            self.restart_service(service)
                
                # Memory management
                memory_percent = metrics['memory']['percent']
                
                if memory_percent > self.config['restart_threshold']:
                    # Critical memory situation
                    if not self.handle_critical_memory():
                        logger.critical("Critical memory situation could not be resolved!")
                
                elif memory_percent > self.config['memory_threshold']:
                    # High memory usage - standard cleanup
                    logger.warning(f"High memory usage: {memory_percent}%")
                    self.cleanup_memory()
                
                # Periodic cleanup (every 2 hours)
                if (datetime.now() - self.last_cleanup).seconds > 7200:
                    logger.info("Performing scheduled cleanup")
                    self.cleanup_memory()
                
                # Clean up old restart counts
                current_hour = datetime.now().strftime('%Y%m%d_%H')
                old_keys = [
                    key for key in self.restart_count.keys() 
                    if not key.endswith(current_hour)
                ]
                for key in old_keys:
                    del self.restart_count[key]
                
                # Log current status
                logger.info(
                    f"System Status - Memory: {memory_percent}%, "
                    f"CPU: {metrics['cpu']['percent']}%, "
                    f"Healthy Services: {sum(health_status.values())}/{len(health_status)}"
                )
                
                # Sleep for check interval
                self.shutdown_event.wait(self.config['monitor_interval'])
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                self.shutdown_event.wait(self.config['monitor_interval'])

def main():
    """Main entry point"""
    try:
        monitor = EnterpriseSystemMonitor()
        monitor.monitor_system()
    except Exception as e:
        logger.error(f"Failed to start system monitor: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("System monitor shutdown complete")

if __name__ == "__main__":
    main()