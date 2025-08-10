#!/usr/bin/env python3
"""
CONTAINER HEALTH MONITOR
Real-time Docker container health monitoring and auto-healing

Consolidated from:
- container-health-monitor.py
- permanent-health-monitor.py
- distributed-health-monitor.py
- comprehensive-agent-health-monitor.py

Purpose: Monitor Docker containers and apply automatic fixes
Author: ULTRA SCRIPT CONSOLIDATION MASTER
"""

import time
import json
import logging
import docker
import signal
import sys
from datetime import datetime, timedelta
from threading import Thread, Event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ContainerHealthMonitor')

class ContainerHealthMonitor:
    """Advanced container health monitoring with auto-healing"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.monitoring = False
        self.stop_event = Event()
        
        # Health check configuration
        self.config = {
            'check_interval': 30,  # seconds
            'restart_threshold': 3,  # consecutive failures before restart
            'memory_threshold': 90,  # percent
            'cpu_threshold': 95,   # percent
            'disk_threshold': 90,  # percent
            'auto_heal': True
        }
        
        # Container failure tracking
        self.failure_counts = {}
        self.restart_history = {}
        self.health_history = []
        
        # Define critical containers that must always be running
        self.critical_containers = {
            'sutazai-postgres',
            'sutazai-redis', 
            'sutazai-backend',
            'sutazai-frontend',
            'sutazai-ollama'
        }
        
    def get_container_stats(self, container) -> Dict[str, Any]:
        """Get comprehensive container statistics"""
        try:
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * \
                             len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
            
            # Network I/O
            networks = stats.get('networks', {})
            rx_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
            tx_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
            
            return {
                'cpu_percent': round(cpu_percent, 2),
                'memory_usage_mb': round(memory_usage / 1024 / 1024, 2),
                'memory_limit_mb': round(memory_limit / 1024 / 1024, 2),
                'memory_percent': round(memory_percent, 2),
                'network_rx_mb': round(rx_bytes / 1024 / 1024, 2),
                'network_tx_mb': round(tx_bytes / 1024 / 1024, 2),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for {container.name}: {e}")
            return {}
    
    def check_container_health(self, container) -> Dict[str, Any]:
        """Comprehensive container health check"""
        result = {
            'name': container.name,
            'id': container.short_id,
            'status': container.status,
            'health': 'unknown',
            'issues': [],
            'stats': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Basic status check
            container.reload()
            result['status'] = container.status
            
            if container.status != 'running':
                result['health'] = 'unhealthy'
                result['issues'].append(f"Container is {container.status}")
                return result
            
            # Get container stats
            stats = self.get_container_stats(container)
            result['stats'] = stats
            
            # Check resource thresholds
            if stats.get('memory_percent', 0) > self.config['memory_threshold']:
                result['issues'].append(f"High memory usage: {stats['memory_percent']:.1f}%")
                
            if stats.get('cpu_percent', 0) > self.config['cpu_threshold']:
                result['issues'].append(f"High CPU usage: {stats['cpu_percent']:.1f}%")
            
            # Check Docker health status if available
            health_status = container.attrs.get('State', {}).get('Health', {})
            if health_status:
                docker_health = health_status.get('Status', 'none')
                result['docker_health'] = docker_health
                
                if docker_health == 'unhealthy':
                    result['issues'].append("Docker health check failed")
                elif docker_health == 'starting':
                    result['issues'].append("Container still starting")
            
            # Check if container has been restarting
            restart_count = container.attrs.get('RestartCount', 0)
            if restart_count > 0:
                result['restart_count'] = restart_count
                if restart_count > 5:
                    result['issues'].append(f"High restart count: {restart_count}")
            
            # Determine overall health
            if result['issues']:
                result['health'] = 'unhealthy' if any('failed' in issue or 'unhealthy' in issue 
                                                     for issue in result['issues']) else 'degraded'
            else:
                result['health'] = 'healthy'
                
        except Exception as e:
            result['health'] = 'error'
            result['issues'].append(f"Health check error: {str(e)}")
            logger.error(f"Error checking health of {container.name}: {e}")
        
        return result
    
    def auto_heal_container(self, container_name: str, issues: List[str]) -> bool:
        """Apply automatic healing to unhealthy container"""
        if not self.config['auto_heal']:
            return False
            
        logger.info(f"Attempting auto-heal for {container_name}: {issues}")
        
        try:
            container = self.docker_client.containers.get(container_name)
            
            # Track restart attempts
            if container_name not in self.restart_history:
                self.restart_history[container_name] = []
            
            # Check if we've restarted too recently (within last 5 minutes)
            recent_restarts = [
                restart_time for restart_time in self.restart_history[container_name]
                if restart_time > datetime.now() - timedelta(minutes=5)
            ]
            
            if len(recent_restarts) >= 3:
                logger.warning(f"Too many recent restarts for {container_name}, skipping auto-heal")
                return False
            
            # Attempt restart
            logger.info(f"Restarting container {container_name}")
            container.restart(timeout=30)
            
            # Record restart
            self.restart_history[container_name].append(datetime.now())
            
            # Wait a moment and check if restart was successful
            time.sleep(10)
            container.reload()
            
            if container.status == 'running':
                logger.info(f"Successfully restarted {container_name}")
                # Reset failure count on successful restart
                self.failure_counts[container_name] = 0
                return True
            else:
                logger.error(f"Restart failed for {container_name}, status: {container.status}")
                return False
                
        except Exception as e:
            logger.error(f"Auto-heal failed for {container_name}: {e}")
            return False
    
    def monitor_containers(self):
        """Main monitoring loop"""
        logger.info("Starting container health monitoring")
        
        while not self.stop_event.is_set():
            try:
                # Get all containers
                containers = self.docker_client.containers.list(all=True)
                
                # Filter for SutazAI containers
                sutazai_containers = [
                    c for c in containers 
                    if c.name.startswith('sutazai-') or 'sutazai' in c.name.lower()
                ]
                
                if not sutazai_containers:
                    logger.warning("No SutazAI containers found")
                    time.sleep(self.config['check_interval'])
                    continue
                
                # Check health of each container
                health_results = []
                unhealthy_containers = []
                
                for container in sutazai_containers:
                    health_result = self.check_container_health(container)
                    health_results.append(health_result)
                    
                    container_name = health_result['name']
                    
                    if health_result['health'] in ['unhealthy', 'error']:
                        # Track consecutive failures
                        self.failure_counts[container_name] = self.failure_counts.get(container_name, 0) + 1
                        unhealthy_containers.append(health_result)
                        
                        logger.warning(f"Container {container_name} is {health_result['health']}: {health_result['issues']}")
                        
                        # Auto-heal if threshold reached
                        if (self.failure_counts[container_name] >= self.config['restart_threshold'] and
                            container_name in self.critical_containers):
                            
                            self.auto_heal_container(container_name, health_result['issues'])
                            
                    else:
                        # Reset failure count on healthy status
                        self.failure_counts[container_name] = 0
                        
                        if health_result['health'] == 'degraded':
                            logger.info(f"Container {container_name} is degraded: {health_result['issues']}")
                
                # Generate summary
                total_containers = len(health_results)
                healthy_containers = sum(1 for r in health_results if r['health'] == 'healthy')
                
                summary = {
                    'timestamp': datetime.now().isoformat(),
                    'total_containers': total_containers,
                    'healthy_containers': healthy_containers,
                    'unhealthy_containers': len(unhealthy_containers),
                    'health_percentage': (healthy_containers / total_containers * 100) if total_containers > 0 else 0,
                    'containers': health_results
                }
                
                # Store in history
                self.health_history.append(summary)
                
                # Keep only last 100 checks
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                
                # Log summary
                logger.info(f"Container Health: {healthy_containers}/{total_containers} healthy "
                           f"({summary['health_percentage']:.1f}%)")
                
                if unhealthy_containers:
                    logger.error(f"Unhealthy containers: {[c['name'] for c in unhealthy_containers]}")
                
                # Save detailed report periodically
                if len(self.health_history) % 10 == 0:  # Every 10 checks
                    self.save_health_report()
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            # Wait for next check
            self.stop_event.wait(self.config['check_interval'])
        
        logger.info("Container health monitoring stopped")
    
    def save_health_report(self):
        """Save detailed health report to file"""
        try:
            report_file = f"/opt/sutazaiapp/logs/container_health_{int(time.time())}.json"
            
            report = {
                'monitoring_config': self.config,
                'failure_counts': self.failure_counts,
                'restart_history': {
                    name: [t.isoformat() for t in times]
                    for name, times in self.restart_history.items()
                },
                'recent_history': self.health_history[-10:] if self.health_history else []
            }
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.debug(f"Health report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save health report: {e}")
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = Thread(target=self.monitor_containers, daemon=True)
        self.monitor_thread.start()
        logger.info("Container health monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring gracefully"""
        logger.info("Stopping container health monitoring...")
        self.monitoring = False
        self.stop_event.set()
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=10)
        self.save_health_report()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, 'monitor'):
        signal_handler.monitor.stop_monitoring()
    sys.exit(0)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SutazAI Container Health Monitor')
    parser.add_argument('--interval', '-i', type=int, default=30,
                       help='Check interval in seconds (default: 30)')
    parser.add_argument('--no-auto-heal', action='store_true',
                       help='Disable automatic healing')
    parser.add_argument('--memory-threshold', type=int, default=90,
                       help='Memory usage threshold percentage (default: 90)')
    parser.add_argument('--cpu-threshold', type=int, default=95,
                       help='CPU usage threshold percentage (default: 95)')
    parser.add_argument('--restart-threshold', type=int, default=3,
                       help='Consecutive failures before restart (default: 3)')
    
    args = parser.parse_args()
    
    # Create monitor instance
    monitor = ContainerHealthMonitor()
    
    # Apply configuration
    monitor.config.update({
        'check_interval': args.interval,
        'auto_heal': not args.no_auto_heal,
        'memory_threshold': args.memory_threshold,
        'cpu_threshold': args.cpu_threshold,
        'restart_threshold': args.restart_threshold
    })
    
    # Set up signal handlers
    signal_handler.monitor = monitor
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Starting SutazAI Container Health Monitor")
        logger.info(f"Configuration: {monitor.config}")
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Keep main thread alive
        while monitor.monitoring:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
    finally:
        monitor.stop_monitoring()
        logger.info("Container health monitoring stopped")


if __name__ == '__main__':
    main()