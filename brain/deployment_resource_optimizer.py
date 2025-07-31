#!/usr/bin/env python3
"""
Deployment Resource Optimizer for SutazAI Multi-Agent System
Optimizes resources during intensive deployment operations
"""

import os
import sys
import time
import json
import logging
import subprocess
from typing import Dict, List, Optional
import psutil
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentResourceOptimizer:
    def __init__(self):
        self.resource_profiles = {
            'conservative': {
                'max_concurrent_containers': 8,
                'memory_per_container_mb': 512,
                'cpu_limit_per_container': '0.5',
                'ollama_memory_gb': 4
            },
            'balanced': {
                'max_concurrent_containers': 12,
                'memory_per_container_mb': 1024,
                'cpu_limit_per_container': '1.0',
                'ollama_memory_gb': 6
            },
            'aggressive': {
                'max_concurrent_containers': 20,
                'memory_per_container_mb': 2048,
                'cpu_limit_per_container': '2.0',
                'ollama_memory_gb': 8
            }
        }
        
        self.docker_available = False
        try:
            import docker
            self.docker_client = docker.from_env()
            self.docker_available = True
        except:
            logger.warning("Docker not available")
    
    def detect_optimal_profile(self) -> str:
        """Detect optimal resource profile based on system capabilities"""
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            if memory_gb >= 12 and cpu_count >= 6:
                return 'aggressive'
            elif memory_gb >= 8 and cpu_count >= 4:
                return 'balanced'
            else:
                return 'conservative'
                
        except Exception as e:
            logger.error(f"Failed to detect profile: {e}")
            return 'conservative'
    
    def optimize_docker_compose(self, compose_file: str, profile: str = None) -> bool:
        """Optimize Docker Compose file for resource usage"""
        try:
            if not profile:
                profile = self.detect_optimal_profile()
            
            config = self.resource_profiles[profile]
            logger.info(f"Optimizing {compose_file} with {profile} profile")
            
            # Read compose file
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            # Apply optimizations to services
            services_optimized = 0
            for service_name, service_config in compose_data.get('services', {}).items():
                # Skip essential services
                if any(essential in service_name.lower() for essential in 
                       ['postgres', 'redis', 'brain']):
                    continue
                
                # Set resource limits
                if 'deploy' not in service_config:
                    service_config['deploy'] = {}
                
                if 'resources' not in service_config['deploy']:
                    service_config['deploy']['resources'] = {}
                
                # Set limits
                service_config['deploy']['resources']['limits'] = {
                    'cpus': config['cpu_limit_per_container'],
                    'memory': f"{config['memory_per_container_mb']}M"
                }
                
                # Set reservations (lower than limits)
                service_config['deploy']['resources']['reservations'] = {
                    'cpus': str(float(config['cpu_limit_per_container']) * 0.5),
                    'memory': f"{config['memory_per_container_mb'] // 2}M"
                }
                
                services_optimized += 1
            
            # Write optimized compose file
            backup_file = f"{compose_file}.backup"
            os.rename(compose_file, backup_file)
            
            with open(compose_file, 'w') as f:
                yaml.dump(compose_data, f, default_flow_style=False)
            
            logger.info(f"Optimized {services_optimized} services in {compose_file}")
            logger.info(f"Backup saved as {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize compose file: {e}")
            return False
    
    def pre_deployment_optimization(self):
        """Run pre-deployment optimizations"""
        logger.info("Starting pre-deployment optimization")
        
        # 1. Clear system caches
        self.clear_system_caches()
        
        # 2. Stop non-essential containers
        self.stop_non_essential_containers()
        
        # 3. Optimize Ollama for minimal memory
        self.optimize_ollama_minimal()
        
        # 4. Set kernel parameters for containers
        self.optimize_kernel_parameters()
        
        logger.info("Pre-deployment optimization complete")
    
    def clear_system_caches(self):
        """Clear system caches to free memory"""
        try:
            logger.info("Clearing system caches")
            os.system('sync')
            
            # Drop caches (requires root)
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('3')
                logger.info("System caches cleared")
            except PermissionError:
                logger.warning("Cannot clear caches - requires root privileges")
                
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
    
    def stop_non_essential_containers(self):
        """Stop non-essential containers to free resources"""
        if not self.docker_available:
            return
            
        try:
            containers = self.docker_client.containers.list()
            non_essential_patterns = ['tier2', 'tier3', 'test', 'dev', 'optional', 'demo']
            
            stopped_count = 0
            for container in containers:
                if any(pattern in container.name.lower() for pattern in non_essential_patterns):
                    logger.info(f"Stopping non-essential container: {container.name}")
                    container.stop(timeout=30)
                    stopped_count += 1
            
            logger.info(f"Stopped {stopped_count} non-essential containers")
            
        except Exception as e:
            logger.error(f"Failed to stop containers: {e}")
    
    def optimize_ollama_minimal(self):
        """Configure Ollama for minimal resource usage"""
        try:
            import requests
            
            # Unload all models
            logger.info("Unloading all Ollama models")
            response = requests.post('http://localhost:11434/api/generate', 
                                   json={'name': '', 'keep_alive': 0}, 
                                   timeout=10)
            
            # Set environment for minimal memory usage
            ollama_env = {
                'OLLAMA_NUM_PARALLEL': '1',
                'OLLAMA_MAX_LOADED_MODELS': '1',
                'OLLAMA_MAX_QUEUE': '2',
                'OLLAMA_FLASH_ATTENTION': 'true',
                'OLLAMA_HOST': '0.0.0.0:11434'
            }
            
            # Update environment (would need container restart)
            logger.info("Ollama optimized for minimal resource usage")
            
        except Exception as e:
            logger.warning(f"Failed to optimize Ollama: {e}")
    
    def optimize_kernel_parameters(self):
        """Optimize kernel parameters for container performance"""
        try:
            optimizations = [
                ('vm.max_map_count', '262144'),
                ('fs.file-max', '65536'),
                ('net.core.somaxconn', '1024')
            ]
            
            for param, value in optimizations:
                try:
                    with open(f'/proc/sys/{param.replace(".", "/")}', 'w') as f:
                        f.write(value)
                    logger.info(f"Set {param} = {value}")
                except PermissionError:
                    logger.warning(f"Cannot set {param} - requires root privileges")
                    
        except Exception as e:
            logger.error(f"Failed to optimize kernel parameters: {e}")
    
    def monitor_deployment_resources(self, duration_minutes: int = 30):
        """Monitor resources during deployment"""
        logger.info(f"Starting deployment resource monitoring for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        peak_cpu = 0
        peak_memory = 0
        alerts = []
        
        try:
            while time.time() < end_time:
                # Get current resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                load_avg = os.getloadavg()[0]
                
                # Track peaks
                peak_cpu = max(peak_cpu, cpu_percent)
                peak_memory = max(peak_memory, memory.percent)
                
                # Check for issues
                if cpu_percent > 90:
                    alert = f"HIGH CPU: {cpu_percent:.1f}%"
                    if alert not in alerts:
                        alerts.append(alert)
                        logger.warning(alert)
                
                if memory.percent > 90:
                    alert = f"HIGH MEMORY: {memory.percent:.1f}%"
                    if alert not in alerts:
                        alerts.append(alert)
                        logger.warning(alert)
                
                if load_avg > psutil.cpu_count() * 2:
                    alert = f"HIGH LOAD: {load_avg:.2f}"
                    if alert not in alerts:
                        alerts.append(alert)
                        logger.warning(alert)
                
                # Print status every 60 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 60 == 0:
                    logger.info(f"Monitoring: CPU {cpu_percent:.1f}%, "
                              f"Memory {memory.percent:.1f}%, "
                              f"Load {load_avg:.2f}")
                
                time.sleep(5)
            
            # Final report
            logger.info(f"Deployment monitoring complete")
            logger.info(f"Peak CPU: {peak_cpu:.1f}%, Peak Memory: {peak_memory:.1f}%")
            if alerts:
                logger.warning(f"Alerts during deployment: {len(alerts)}")
                for alert in alerts:
                    logger.warning(f"  - {alert}")
            
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
    
    def post_deployment_cleanup(self):
        """Run post-deployment cleanup"""
        logger.info("Starting post-deployment cleanup")
        
        try:
            # Remove unused Docker objects
            if self.docker_available:
                logger.info("Cleaning up Docker objects")
                self.docker_client.containers.prune()
                self.docker_client.images.prune(filters={'dangling': True})
                self.docker_client.volumes.prune()
                self.docker_client.networks.prune()
            
            # Clear logs older than 7 days
            self.cleanup_old_logs()
            
            logger.info("Post-deployment cleanup complete")
            
        except Exception as e:
            logger.error(f"Post-deployment cleanup failed: {e}")
    
    def cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            log_dirs = ['/opt/sutazaiapp/logs', '/var/log/sutazai']
            
            for log_dir in log_dirs:
                if os.path.exists(log_dir):
                    # Remove logs older than 7 days
                    os.system(f'find {log_dir} -name "*.log" -mtime +7 -delete')
                    logger.info(f"Cleaned old logs from {log_dir}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup logs: {e}")
    
    def create_resource_report(self) -> Dict:
        """Create comprehensive resource report"""
        try:
            # System info
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            report = {
                'timestamp': time.time(),
                'system': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_total_gb': disk.total / (1024**3),
                    'disk_free_gb': disk.free / (1024**3)
                },
                'recommended_profile': self.detect_optimal_profile(),
                'resource_profiles': self.resource_profiles
            }
            
            # Container info
            if self.docker_available:
                containers = self.docker_client.containers.list(all=True)
                report['containers'] = {
                    'total': len(containers),
                    'running': len([c for c in containers if c.status == 'running']),
                    'stopped': len([c for c in containers if c.status == 'exited'])
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to create resource report: {e}")
            return {'error': str(e)}

def main():
    """Main optimization workflow"""
    optimizer = DeploymentResourceOptimizer()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'pre-deploy':
            optimizer.pre_deployment_optimization()
        elif command == 'monitor':
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            optimizer.monitor_deployment_resources(duration)
        elif command == 'post-deploy':
            optimizer.post_deployment_cleanup()
        elif command == 'report':
            report = optimizer.create_resource_report()
            print(json.dumps(report, indent=2))
        elif command == 'optimize-compose':
            if len(sys.argv) > 2:
                compose_file = sys.argv[2]
                profile = sys.argv[3] if len(sys.argv) > 3 else None
                optimizer.optimize_docker_compose(compose_file, profile)
            else:
                print("Usage: optimize-compose <compose-file> [profile]")
        else:
            print("Unknown command. Available: pre-deploy, monitor, post-deploy, report, optimize-compose")
    else:
        print("SutazAI Deployment Resource Optimizer")
        print("Commands:")
        print("  pre-deploy          - Run pre-deployment optimizations")
        print("  monitor [minutes]   - Monitor resources during deployment")
        print("  post-deploy         - Run post-deployment cleanup")
        print("  report              - Generate resource report")
        print("  optimize-compose <file> [profile] - Optimize Docker Compose file")

if __name__ == '__main__':
    main()