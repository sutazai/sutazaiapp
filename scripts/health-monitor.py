#!/usr/bin/env python3
"""
SutazAI Health Monitor and Auto-Recovery System
Monitors system health and automatically recovers from failures
"""

import os
import sys
import time
import json
import psutil
import docker
import requests
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple
import smtplib
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/sutazai-health.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SutazAI-HealthMonitor')

class HealthMonitor:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.services = {
            'fastapi-backend': {
                'url': 'http://localhost:8000/health',
                'critical': True,
                'memory_limit_mb': 1024,
                'restart_on_failure': True
            },
            'streamlit-frontend': {
                'url': 'http://localhost:8501/healthz',
                'critical': True,
                'memory_limit_mb': 1024,
                'restart_on_failure': True
            },
            'ollama': {
                'url': 'http://localhost:11434/api/tags',
                'critical': True,
                'memory_limit_mb': 4096,
                'restart_on_failure': True,
                'custom_recovery': self.recover_ollama
            },
            'postgresql': {
                'port_check': 5432,
                'critical': True,
                'memory_limit_mb': 1024,
                'restart_on_failure': True
            },
            'redis': {
                'port_check': 6379,
                'critical': True,
                'memory_limit_mb': 512,
                'restart_on_failure': True
            },
            'chromadb': {
                'url': 'http://localhost:8001/api/v1/heartbeat',
                'critical': False,
                'memory_limit_mb': 1024,
                'restart_on_failure': True
            },
            'qdrant': {
                'url': 'http://localhost:6333/health',
                'critical': False,
                'memory_limit_mb': 1024,
                'restart_on_failure': True
            }
        }
        
        self.memory_thresholds = {
            'warning': 80,  # Percentage
            'critical': 90,
            'swap_warning': 50
        }
        
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
        self.alert_cooldown = {}
        self.metrics_history = []

    def check_system_resources(self) -> Dict:
        """Check overall system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory': {
                'total_mb': memory.total // (1024 * 1024),
                'used_mb': memory.used // (1024 * 1024),
                'available_mb': memory.available // (1024 * 1024),
                'percent': memory.percent
            },
            'swap': {
                'total_mb': swap.total // (1024 * 1024),
                'used_mb': swap.used // (1024 * 1024),
                'percent': swap.percent
            },
            'disk': {
                'total_gb': disk.total // (1024 ** 3),
                'used_gb': disk.used // (1024 ** 3),
                'free_gb': disk.free // (1024 ** 3),
                'percent': disk.percent
            }
        }

    def check_service_health(self, service_name: str, config: Dict) -> Tuple[bool, str]:
        """Check if a service is healthy"""
        try:
            # Check if container is running
            container = self.docker_client.containers.get(f"sutazaiapp_{service_name}_1")
            if container.status != 'running':
                return False, f"Container not running: {container.status}"
            
            # Check container memory usage
            stats = container.stats(stream=False)
            memory_usage_mb = stats['memory_stats']['usage'] // (1024 * 1024)
            memory_limit_mb = config.get('memory_limit_mb', 1024)
            
            if memory_usage_mb > memory_limit_mb * 0.9:
                return False, f"High memory usage: {memory_usage_mb}MB / {memory_limit_mb}MB"
            
            # Check service endpoint
            if 'url' in config:
                response = requests.get(config['url'], timeout=5)
                if response.status_code != 200:
                    return False, f"Health check failed: HTTP {response.status_code}"
            
            # Check port
            elif 'port_check' in config:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', config['port_check']))
                sock.close()
                if result != 0:
                    return False, f"Port {config['port_check']} not accessible"
            
            return True, "Healthy"
            
        except docker.errors.NotFound:
            return False, "Container not found"
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Check failed: {str(e)}"

    def recover_service(self, service_name: str, reason: str):
        """Attempt to recover a failed service"""
        logger.warning(f"Attempting to recover {service_name}: {reason}")
        
        # Track recovery attempts
        if service_name not in self.recovery_attempts:
            self.recovery_attempts[service_name] = 0
        
        self.recovery_attempts[service_name] += 1
        
        if self.recovery_attempts[service_name] > self.max_recovery_attempts:
            logger.error(f"Max recovery attempts reached for {service_name}")
            self.send_alert(f"Service {service_name} failed after {self.max_recovery_attempts} recovery attempts", 'critical')
            return
        
        try:
            # Get service configuration
            config = self.services[service_name]
            
            # Custom recovery if available
            if 'custom_recovery' in config:
                config['custom_recovery']()
                return
            
            # Standard recovery: restart container
            container = self.docker_client.containers.get(f"sutazaiapp_{service_name}_1")
            
            # Stop gracefully
            container.stop(timeout=30)
            time.sleep(5)
            
            # Start again
            container.start()
            
            # Wait for service to be ready
            time.sleep(10)
            
            # Verify recovery
            healthy, status = self.check_service_health(service_name, config)
            if healthy:
                logger.info(f"Successfully recovered {service_name}")
                self.recovery_attempts[service_name] = 0
            else:
                logger.error(f"Recovery failed for {service_name}: {status}")
                
        except Exception as e:
            logger.error(f"Recovery error for {service_name}: {str(e)}")

    def recover_ollama(self):
        """Custom recovery for Ollama service"""
        logger.info("Performing custom Ollama recovery")
        
        try:
            # First, try to unload all models to free memory
            try:
                response = requests.get('http://localhost:11434/api/ps', timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    for model in models:
                        logger.info(f"Unloading model: {model.get('name', 'unknown')}")
                        requests.post('http://localhost:11434/api/generate',
                                    json={'model': model['name'], 'keep_alive': 0})
            except:
                pass
            
            # Restart the container
            container = self.docker_client.containers.get("sutazaiapp_ollama_1")
            container.restart(timeout=30)
            
            # Wait for Ollama to be ready
            time.sleep(20)
            
            # Load minimal model
            subprocess.run([
                "docker", "exec", "sutazaiapp_ollama_1",
                "ollama", "pull", "qwen2.5:3b"
            ], check=False)
            
        except Exception as e:
            logger.error(f"Ollama recovery failed: {str(e)}")

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        resources = self.check_system_resources()
        memory_percent = resources['memory']['percent']
        
        if memory_percent > self.memory_thresholds['critical']:
            logger.critical(f"Critical memory usage: {memory_percent}%")
            self.handle_memory_pressure('critical')
            return True
        elif memory_percent > self.memory_thresholds['warning']:
            logger.warning(f"High memory usage: {memory_percent}%")
            self.handle_memory_pressure('warning')
            return True
        
        return False

    def handle_memory_pressure(self, level: str):
        """Handle memory pressure situations"""
        logger.info(f"Handling {level} memory pressure")
        
        # Get container memory usage
        containers = self.docker_client.containers.list()
        container_stats = []
        
        for container in containers:
            try:
                stats = container.stats(stream=False)
                memory_usage = stats['memory_stats']['usage'] // (1024 * 1024)
                container_stats.append({
                    'name': container.name,
                    'memory_mb': memory_usage,
                    'container': container
                })
            except:
                continue
        
        # Sort by memory usage
        container_stats.sort(key=lambda x: x['memory_mb'], reverse=True)
        
        if level == 'critical':
            # Kill non-critical services using most memory
            for stat in container_stats:
                service_name = stat['name'].replace('sutazaiapp_', '').replace('_1', '')
                if service_name in self.services and not self.services[service_name].get('critical', True):
                    logger.warning(f"Stopping non-critical service {service_name} to free memory")
                    try:
                        stat['container'].stop(timeout=10)
                    except:
                        pass
        
        # Force garbage collection on Python services
        for service in ['fastapi-backend', 'streamlit-frontend']:
            try:
                container = self.docker_client.containers.get(f"sutazaiapp_{service}_1")
                container.exec_run("python -c 'import gc; gc.collect()'")
            except:
                pass

    def cleanup_docker_resources(self):
        """Clean up unused Docker resources"""
        logger.info("Cleaning up Docker resources")
        
        try:
            # Remove stopped containers
            self.docker_client.containers.prune()
            
            # Remove unused images
            self.docker_client.images.prune(filters={'dangling': True})
            
            # Remove unused volumes
            self.docker_client.volumes.prune()
            
            # Remove unused networks
            self.docker_client.networks.prune()
            
        except Exception as e:
            logger.error(f"Docker cleanup failed: {str(e)}")

    def send_alert(self, message: str, severity: str = 'warning'):
        """Send alert notification"""
        alert_key = f"{severity}:{message[:50]}"
        current_time = time.time()
        
        # Check cooldown
        if alert_key in self.alert_cooldown:
            if current_time - self.alert_cooldown[alert_key] < 3600:  # 1 hour cooldown
                return
        
        self.alert_cooldown[alert_key] = current_time
        
        # Log alert
        logger.warning(f"ALERT [{severity}]: {message}")
        
        # Here you could add email/webhook notifications
        # Example: send to webhook
        webhook_url = os.environ.get('ALERT_WEBHOOK')
        if webhook_url:
            try:
                requests.post(webhook_url, json={
                    'text': f"SutazAI Alert [{severity}]: {message}",
                    'severity': severity,
                    'timestamp': datetime.now().isoformat()
                })
            except:
                pass

    def generate_report(self) -> Dict:
        """Generate health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_resources': self.check_system_resources(),
            'services': {},
            'overall_health': 'healthy'
        }
        
        # Check each service
        for service_name, config in self.services.items():
            healthy, status = self.check_service_health(service_name, config)
            report['services'][service_name] = {
                'healthy': healthy,
                'status': status,
                'critical': config.get('critical', False)
            }
            
            if not healthy and config.get('critical', False):
                report['overall_health'] = 'critical'
            elif not healthy and report['overall_health'] == 'healthy':
                report['overall_health'] = 'degraded'
        
        return report

    def run_health_check(self):
        """Main health check routine"""
        logger.info("Running health check")
        
        # Check memory pressure first
        if self.check_memory_pressure():
            self.send_alert("System under memory pressure", 'warning')
        
        # Check each service
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for service_name, config in self.services.items():
                future = executor.submit(self.check_service_health, service_name, config)
                futures.append((service_name, config, future))
            
            for service_name, config, future in futures:
                try:
                    healthy, status = future.result(timeout=10)
                    if not healthy:
                        logger.warning(f"{service_name} is unhealthy: {status}")
                        if config.get('restart_on_failure', True):
                            self.recover_service(service_name, status)
                        if config.get('critical', False):
                            self.send_alert(f"Critical service {service_name} is down: {status}", 'critical')
                except Exception as e:
                    logger.error(f"Health check failed for {service_name}: {str(e)}")
        
        # Store metrics
        report = self.generate_report()
        self.metrics_history.append(report)
        
        # Keep only last 1000 entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        # Save report
        with open('/var/log/sutazai-health-report.json', 'w') as f:
            json.dump(report, f, indent=2)

    def run(self):
        """Main monitoring loop"""
        logger.info("SutazAI Health Monitor started")
        
        # Schedule tasks
        schedule.every(60).seconds.do(self.run_health_check)
        schedule.every(30).minutes.do(self.cleanup_docker_resources)
        
        # Initial check
        self.run_health_check()
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Health monitor stopped")
                break
            except Exception as e:
                logger.error(f"Monitor error: {str(e)}")
                time.sleep(10)

if __name__ == "__main__":
    monitor = HealthMonitor()
    monitor.run()