#!/usr/bin/env python3
"""
Automated Resource Monitor for Sutazaiapp
Created: 2025-08-20
Purpose: Continuous monitoring and automatic remediation of resource issues
"""

import os
import sys
import time
import json
import psutil
import docker
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Configuration
@dataclass
class MonitorConfig:
    """Monitoring configuration with thresholds"""
    memory_threshold_percent: float = 80.0
    cpu_threshold_percent: float = 70.0
    zombie_threshold: int = 10
    container_memory_limit_mb: Dict[str, int] = None
    check_interval_seconds: int = 60
    alert_cooldown_minutes: int = 15
    log_file: str = "/var/log/sutazaiapp_monitor.log"
    metrics_file: str = "/opt/sutazaiapp/metrics/resource_usage.json"
    
    def __post_init__(self):
        if self.container_memory_limit_mb is None:
            self.container_memory_limit_mb = {
                "sutazai-mcp-orchestrator": 256,
                "sutazai-neo4j": 512,
                "sutazai-ollama": 512,
                "sutazai-grafana": 128,
                "sutazai-prometheus": 256,
                "sutazai-backend": 256,
                "sutazai-frontend": 128,
            }

class ResourceMonitor:
    """Main resource monitoring and remediation class"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.docker_client = docker.from_env()
        self.last_alert_time = {}
        self.setup_logging()
        self.metrics_history = []
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_system_metrics(self) -> Dict:
        """Collect current system metrics"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "load_avg": os.getloadavg(),
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "used": psutil.virtual_memory().used,
                "percent": psutil.virtual_memory().percent,
                "available": psutil.virtual_memory().available,
            },
            "swap": {
                "total": psutil.swap_memory().total,
                "used": psutil.swap_memory().used,
                "percent": psutil.swap_memory().percent,
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "percent": psutil.disk_usage('/').percent,
            },
        }
        return metrics
    
    def count_zombie_processes(self) -> Tuple[int, List[Dict]]:
        """Count and identify zombie processes"""
        zombies = []
        for proc in psutil.process_iter(['pid', 'ppid', 'name', 'status']):
            try:
                if proc.info['status'] == psutil.STATUS_ZOMBIE:
                    zombies.append({
                        'pid': proc.info['pid'],
                        'ppid': proc.info['ppid'],
                        'name': proc.info['name']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return len(zombies), zombies
    
    def get_container_stats(self) -> List[Dict]:
        """Get Docker container resource statistics"""
        stats = []
        try:
            for container in self.docker_client.containers.list():
                try:
                    container_stats = container.stats(stream=False)
                    memory_usage = container_stats['memory_stats'].get('usage', 0)
                    memory_limit = container_stats['memory_stats'].get('limit', 1)
                    
                    cpu_delta = container_stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               container_stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = container_stats['cpu_stats']['system_cpu_usage'] - \
                                  container_stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0
                    
                    stats.append({
                        'name': container.name,
                        'id': container.short_id,
                        'status': container.status,
                        'memory_usage_mb': memory_usage / (1024 * 1024),
                        'memory_percent': (memory_usage / memory_limit) * 100,
                        'cpu_percent': cpu_percent,
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to get stats for container {container.name}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to get container stats: {e}")
        return stats
    
    def clean_zombie_processes(self, zombies: List[Dict]) -> int:
        """Clean up zombie processes"""
        cleaned = 0
        parent_pids = set(z['ppid'] for z in zombies)
        
        for ppid in parent_pids:
            if ppid > 1:  # Don't touch init process
                try:
                    os.kill(ppid, 18)  # SIGCHLD
                    cleaned += 1
                    self.logger.info(f"Sent SIGCHLD to parent process {ppid}")
                except ProcessLookupError:
                    pass
                except PermissionError:
                    self.logger.warning(f"Permission denied to signal process {ppid}")
        
        time.sleep(2)  # Give time for cleanup
        return cleaned
    
    def optimize_container_memory(self, container_stats: List[Dict]):
        """Apply memory limits to containers exceeding thresholds"""
        for stats in container_stats:
            container_name = stats['name']
            memory_usage_mb = stats['memory_usage_mb']
            
            if container_name in self.config.container_memory_limit_mb:
                limit_mb = self.config.container_memory_limit_mb[container_name]
                
                if memory_usage_mb > limit_mb * 1.2:  # 20% over limit
                    try:
                        container = self.docker_client.containers.get(container_name)
                        container.update(mem_limit=f"{limit_mb}m")
                        self.logger.info(f"Updated memory limit for {container_name} to {limit_mb}MB")
                    except Exception as e:
                        self.logger.error(f"Failed to update memory limit for {container_name}: {e}")
    
    def kill_duplicate_mcp_processes(self):
        """Identify and kill duplicate MCP processes"""
        mcp_processes = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'mcp' in cmdline.lower() and 'grep' not in cmdline:
                    # Create a key based on the command pattern
                    key = cmdline.split()[0] if cmdline else proc.info['name']
                    
                    if key not in mcp_processes:
                        mcp_processes[key] = []
                    mcp_processes[key].append({
                        'pid': proc.info['pid'],
                        'create_time': proc.info['create_time'],
                        'cmdline': cmdline
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Kill duplicates (keep oldest)
        killed = 0
        for key, processes in mcp_processes.items():
            if len(processes) > 1:
                processes.sort(key=lambda x: x['create_time'])
                for proc in processes[1:]:  # Keep first (oldest), kill rest
                    try:
                        os.kill(proc['pid'], 15)  # SIGTERM
                        killed += 1
                        self.logger.info(f"Killed duplicate MCP process: PID {proc['pid']}")
                    except:
                        pass
        
        return killed
    
    def should_alert(self, alert_type: str) -> bool:
        """Check if we should send an alert (respecting cooldown)"""
        now = datetime.utcnow()
        last_alert = self.last_alert_time.get(alert_type)
        
        if last_alert is None:
            return True
        
        cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
        return now - last_alert > cooldown
    
    def record_alert(self, alert_type: str):
        """Record that an alert was sent"""
        self.last_alert_time[alert_type] = datetime.utcnow()
    
    def save_metrics(self, metrics: Dict):
        """Save metrics to file for historical analysis"""
        self.metrics_history.append(metrics)
        
        # Keep only last 24 hours of metrics
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        # Save to file
        try:
            Path(self.config.metrics_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def run_monitoring_cycle(self):
        """Run a single monitoring cycle"""
        self.logger.info("Starting monitoring cycle")
        
        # Collect metrics
        system_metrics = self.get_system_metrics()
        zombie_count, zombies = self.count_zombie_processes()
        container_stats = self.get_container_stats()
        
        # Combine metrics
        metrics = {
            **system_metrics,
            "zombies": {
                "count": zombie_count,
                "processes": zombies
            },
            "containers": container_stats
        }
        
        # Save metrics
        self.save_metrics(metrics)
        
        # Check thresholds and take action
        actions_taken = []
        
        # Memory threshold
        if system_metrics['memory']['percent'] > self.config.memory_threshold_percent:
            if self.should_alert('high_memory'):
                self.logger.warning(f"High memory usage: {system_metrics['memory']['percent']:.1f}%")
                self.record_alert('high_memory')
                
                # Take corrective action
                self.optimize_container_memory(container_stats)
                killed_mcp = self.kill_duplicate_mcp_processes()
                
                actions_taken.append(f"Optimized containers, killed {killed_mcp} duplicate MCP processes")
        
        # CPU threshold
        if system_metrics['cpu']['percent'] > self.config.cpu_threshold_percent:
            if self.should_alert('high_cpu'):
                self.logger.warning(f"High CPU usage: {system_metrics['cpu']['percent']:.1f}%")
                self.record_alert('high_cpu')
                actions_taken.append("High CPU alert sent")
        
        # Zombie threshold
        if zombie_count > self.config.zombie_threshold:
            if self.should_alert('zombies'):
                self.logger.warning(f"High zombie count: {zombie_count}")
                self.record_alert('zombies')
                
                # Clean zombies
                cleaned = self.clean_zombie_processes(zombies)
                actions_taken.append(f"Cleaned {cleaned} zombie parent processes")
        
        # Log summary
        self.logger.info(f"Monitoring cycle complete. Actions: {actions_taken if actions_taken else 'None'}")
        
        return metrics, actions_taken
    
    def run(self):
        """Main monitoring loop"""
        self.logger.info("Resource monitor started")
        
        while True:
            try:
                metrics, actions = self.run_monitoring_cycle()
                
                # Generate summary report every hour
                if datetime.utcnow().minute == 0:
                    self.generate_summary_report()
                
            except KeyboardInterrupt:
                self.logger.info("Monitor stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}", exc_info=True)
            
            # Always sleep to prevent CPU spinning
            time.sleep(self.config.check_interval_seconds)
    
    def generate_summary_report(self):
        """Generate hourly summary report"""
        if not self.metrics_history:
            return
        
        # Calculate averages
        avg_cpu = sum(m['cpu']['percent'] for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m['memory']['percent'] for m in self.metrics_history) / len(self.metrics_history)
        max_zombies = max(m['zombies']['count'] for m in self.metrics_history)
        
        report = f"""
HOURLY RESOURCE SUMMARY
=======================
Time: {datetime.utcnow().isoformat()}
Samples: {len(self.metrics_history)}

CPU:
  Average: {avg_cpu:.1f}%
  Current: {self.metrics_history[-1]['cpu']['percent']:.1f}%

Memory:
  Average: {avg_memory:.1f}%
  Current: {self.metrics_history[-1]['memory']['percent']:.1f}%

Zombies:
  Maximum: {max_zombies}
  Current: {self.metrics_history[-1]['zombies']['count']}

Top Memory Containers:
"""
        
        # Add container stats
        if self.metrics_history[-1].get('containers'):
            containers = sorted(
                self.metrics_history[-1]['containers'],
                key=lambda x: x['memory_usage_mb'],
                reverse=True
            )[:5]
            
            for c in containers:
                report += f"  {c['name']}: {c['memory_usage_mb']:.1f}MB ({c['memory_percent']:.1f}%)\n"
        
        self.logger.info(report)


def main():
    """Main entry point"""
    # Load configuration
    config = MonitorConfig()
    
    # Override from environment if available
    if os.getenv('MEMORY_THRESHOLD'):
        config.memory_threshold_percent = float(os.getenv('MEMORY_THRESHOLD'))
    if os.getenv('CPU_THRESHOLD'):
        config.cpu_threshold_percent = float(os.getenv('CPU_THRESHOLD'))
    if os.getenv('CHECK_INTERVAL'):
        config.check_interval_seconds = int(os.getenv('CHECK_INTERVAL'))
    
    # Create and run monitor
    monitor = ResourceMonitor(config)
    
    # Run in daemon mode or single cycle
    if '--daemon' in sys.argv:
        monitor.run()
    else:
        metrics, actions = monitor.run_monitoring_cycle()
        print(json.dumps(metrics, indent=2, default=str))
        if actions:
            print(f"\nActions taken: {actions}")


if __name__ == "__main__":
    main()