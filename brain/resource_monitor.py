#!/usr/bin/env python3
"""
Real-time Resource Monitor for SutazAI Multi-Agent System
Monitors system health during intensive AI operations
"""

import os
import sys
import time
import json
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Optional
import threading
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.alert_thresholds = {
            'cpu_critical': 95.0,
            'cpu_warning': 80.0,
            'memory_critical': 95.0,
            'memory_warning': 85.0,
            'disk_critical': 95.0,
            'disk_warning': 85.0,
            'load_critical': 16.0,  # 2x CPU cores
            'load_warning': 12.0    # 1.5x CPU cores
        }
        
        self.monitoring_interval = 5  # seconds
        self.history = []
        self.max_history = 100
        
        # Docker client setup
        try:
            import docker
            self.docker_client = docker.from_env()
            self.docker_available = True
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self.docker_client = None
            self.docker_available = False
    
    def get_system_snapshot(self) -> Dict:
        """Get comprehensive system resource snapshot"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            load_avg = os.getloadavg()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count_physical': cpu_count,
                    'count_logical': cpu_count_logical,
                    'load_1min': load_avg[0],
                    'load_5min': load_avg[1],
                    'load_15min': load_avg[2]
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent,
                    'cached_gb': memory.cached / (1024**3),
                    'buffers_gb': memory.buffers / (1024**3)
                },
                'swap': {
                    'total_gb': swap.total / (1024**3),
                    'used_gb': swap.used / (1024**3),
                    'percent': swap.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percent': disk.percent
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'processes': process_count
            }
            
            # Add Docker container info if available
            if self.docker_available:
                snapshot['containers'] = self.get_container_summary()
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to get system snapshot: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def get_container_summary(self) -> Dict:
        """Get Docker container resource summary"""
        try:
            containers = self.docker_client.containers.list(all=True)
            
            running_count = sum(1 for c in containers if c.status == 'running')
            stopped_count = sum(1 for c in containers if c.status == 'exited')
            
            # Get resource usage for running containers
            total_memory_mb = 0
            ai_containers = []
            
            for container in containers:
                if container.status == 'running':
                    try:
                        stats = container.stats(stream=False)
                        memory_usage = stats.get('memory_stats', {}).get('usage', 0) / (1024**2)
                        total_memory_mb += memory_usage
                        
                        # Track AI-related containers
                        if any(keyword in container.name.lower() for keyword in 
                               ['sutazai', 'ollama', 'brain', 'agent', 'ai']):
                            ai_containers.append({
                                'name': container.name,
                                'status': container.status,
                                'memory_mb': memory_usage
                            })
                    except:
                        pass
            
            return {
                'total_containers': len(containers),
                'running': running_count,
                'stopped': stopped_count,
                'total_memory_mb': total_memory_mb,
                'ai_containers': ai_containers
            }
            
        except Exception as e:
            logger.error(f"Failed to get container summary: {e}")
            return {'error': str(e)}
    
    def check_alerts(self, snapshot: Dict) -> List[str]:
        """Check for resource alerts"""
        alerts = []
        
        try:
            # CPU alerts
            cpu_percent = snapshot['cpu']['percent']
            if cpu_percent >= self.alert_thresholds['cpu_critical']:
                alerts.append(f"CRITICAL: CPU usage at {cpu_percent:.1f}%")
            elif cpu_percent >= self.alert_thresholds['cpu_warning']:
                alerts.append(f"WARNING: CPU usage at {cpu_percent:.1f}%")
            
            # Memory alerts
            memory_percent = snapshot['memory']['percent']
            if memory_percent >= self.alert_thresholds['memory_critical']:
                alerts.append(f"CRITICAL: Memory usage at {memory_percent:.1f}%")
            elif memory_percent >= self.alert_thresholds['memory_warning']:
                alerts.append(f"WARNING: Memory usage at {memory_percent:.1f}%")
            
            # Load average alerts
            load_1min = snapshot['cpu']['load_1min']
            if load_1min >= self.alert_thresholds['load_critical']:
                alerts.append(f"CRITICAL: Load average {load_1min:.2f}")
            elif load_1min >= self.alert_thresholds['load_warning']:
                alerts.append(f"WARNING: Load average {load_1min:.2f}")
            
            # Disk alerts
            disk_percent = snapshot['disk']['percent']
            if disk_percent >= self.alert_thresholds['disk_critical']:
                alerts.append(f"CRITICAL: Disk usage at {disk_percent:.1f}%")
            elif disk_percent >= self.alert_thresholds['disk_warning']:
                alerts.append(f"WARNING: Disk usage at {disk_percent:.1f}%")
            
            # Memory availability check
            available_gb = snapshot['memory']['available_gb']
            if available_gb < 1.0:
                alerts.append(f"CRITICAL: Only {available_gb:.1f}GB memory available")
            elif available_gb < 2.0:
                alerts.append(f"WARNING: Only {available_gb:.1f}GB memory available")
                
        except Exception as e:
            alerts.append(f"ERROR: Failed to check alerts: {e}")
        
        return alerts
    
    def suggest_optimizations(self, snapshot: Dict) -> List[str]:
        """Suggest resource optimizations"""
        suggestions = []
        
        try:
            cpu_percent = snapshot['cpu']['percent']
            memory_percent = snapshot['memory']['percent']
            load_1min = snapshot['cpu']['load_1min']
            
            # High CPU suggestions
            if cpu_percent > 80:
                suggestions.append("Consider reducing concurrent operations")
                suggestions.append("Stop non-essential containers")
                
            # High memory suggestions
            if memory_percent > 85:
                suggestions.append("Unload large AI models from Ollama")
                suggestions.append("Clear system caches: sync && echo 3 > /proc/sys/vm/drop_caches")
                suggestions.append("Scale down memory-intensive containers")
            
            # High load suggestions
            if load_1min > 12:
                suggestions.append("Reduce parallel processing tasks")
                suggestions.append("Implement task queuing for intensive operations")
            
            # Container-specific suggestions
            if 'containers' in snapshot:
                total_containers = snapshot['containers'].get('running', 0)
                if total_containers > 15:
                    suggestions.append(f"Consider reducing running containers ({total_containers} active)")
                
                container_memory = snapshot['containers'].get('total_memory_mb', 0)
                if container_memory > 8000:  # 8GB
                    suggestions.append(f"Container memory usage high: {container_memory:.0f}MB")
                    suggestions.append("Review container memory limits")
                    
        except Exception as e:
            suggestions.append(f"ERROR: Failed to generate suggestions: {e}")
        
        return suggestions
    
    def emergency_cleanup(self):
        """Emergency resource cleanup procedures"""
        logger.warning("Initiating emergency resource cleanup")
        
        try:
            # Stop non-essential containers
            if self.docker_available:
                containers = self.docker_client.containers.list()
                non_essential = ['tier2', 'tier3', 'test', 'dev', 'optional']
                
                for container in containers:
                    if any(pattern in container.name.lower() for pattern in non_essential):
                        logger.warning(f"Emergency stopping container: {container.name}")
                        container.stop(timeout=10)
            
            # Clear system caches (requires root)
            try:
                os.system('sync')
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('3')
                logger.info("System caches cleared")
            except:
                logger.warning("Could not clear system caches (requires root)")
            
            # Unload Ollama models
            try:
                import requests
                response = requests.post('http://localhost:11434/api/generate', 
                                       json={'name': '', 'keep_alive': 0}, 
                                       timeout=5)
                logger.info("Ollama models unloaded")
            except:
                logger.warning("Could not unload Ollama models")
                
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def print_summary(self, snapshot: Dict, alerts: List[str], suggestions: List[str]):
        """Print formatted resource summary"""
        print("\n" + "="*80)
        print(f"🔧 RESOURCE MONITOR - {snapshot['timestamp']}")
        print("="*80)
        
        # System resources
        cpu = snapshot['cpu']
        memory = snapshot['memory']
        disk = snapshot['disk']
        
        print(f"💻 CPU: {cpu['percent']:.1f}% | Load: {cpu['load_1min']:.2f} | Cores: {cpu['count_physical']}")
        print(f"💾 Memory: {memory['percent']:.1f}% | Used: {memory['used_gb']:.1f}GB | Available: {memory['available_gb']:.1f}GB")
        print(f"💿 Disk: {disk['percent']:.1f}% | Used: {disk['used_gb']:.0f}GB | Free: {disk['free_gb']:.0f}GB")
        
        # Container info
        if 'containers' in snapshot:
            containers = snapshot['containers']
            print(f"🐳 Containers: {containers['running']} running | {containers['total_memory_mb']:.0f}MB total")
            
            if containers.get('ai_containers'):
                print("🤖 AI Containers:")
                for container in containers['ai_containers'][:5]:  # Show top 5
                    print(f"   • {container['name']}: {container['memory_mb']:.0f}MB")
        
        # Alerts
        if alerts:
            print("\n🚨 ALERTS:")
            for alert in alerts:
                print(f"   {alert}")
        
        # Suggestions
        if suggestions:
            print("\n💡 OPTIMIZATION SUGGESTIONS:")
            for suggestion in suggestions:
                print(f"   • {suggestion}")
        
        print("="*80)
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring = True
        logger.info("Starting resource monitoring...")
        
        try:
            while self.monitoring:
                snapshot = self.get_system_snapshot()
                alerts = self.check_alerts(snapshot)
                suggestions = self.suggest_optimizations(snapshot)
                
                # Add to history
                self.history.append(snapshot)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                
                # Print summary
                self.print_summary(snapshot, alerts, suggestions)
                
                # Emergency cleanup if system is critical
                critical_alerts = [a for a in alerts if 'CRITICAL' in a]
                if len(critical_alerts) >= 2:
                    logger.error("Multiple critical alerts - initiating emergency cleanup")
                    self.emergency_cleanup()
                
                time.sleep(self.monitoring_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
        finally:
            self.monitoring = False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("Stopping resource monitoring...")

# Global monitor instance
monitor = SystemResourceMonitor()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nShutting down resource monitor...")
    monitor.stop_monitoring()
    sys.exit(0)

if __name__ == '__main__':
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🔧 SutazAI Hardware Resource Monitor")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 50)
    
    # Start monitoring
    monitor.start_monitoring()