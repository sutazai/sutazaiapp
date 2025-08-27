#!/usr/bin/env python3
"""
Hardware Performance Monitoring System
Continuous monitoring of hardware optimization results
"""

import time
import json
import psutil
import subprocess
import sys
import os
from datetime import datetime
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/hardware_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HardwarePerformanceMonitor:
    """Comprehensive hardware performance monitoring"""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'load_average_1min': 4.0,
            'dockerd_cpu': 25.0,
            'process_count': 600
        }
        
    def get_dockerd_cpu(self) -> float:
        """Get dockerd CPU usage"""
        try:
            result = subprocess.run(
                "ps aux | grep dockerd | grep -v grep | awk '{print $3}'",
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            return 0.0
        except:
            return 0.0
    
    def get_mcp_containers_count(self) -> int:
        """Get count of MCP containers"""
        try:
            result = subprocess.run(
                "docker ps --format '{{.Names}}' | grep -E '(mcp|duckduckgo|fetch|sequentialthinking)' | wc -l",
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
            return 0
        except:
            return 0
    
    def collect_metrics(self) -> Dict[str, any]:
        """Collect comprehensive system metrics"""
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            load_avg = os.getloadavg()
            process_count = len(psutil.pids())
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            # Docker-specific metrics
            dockerd_cpu = self.get_dockerd_cpu()
            mcp_containers = self.get_mcp_containers_count()
            
            # Container resource usage
            container_stats = self.get_container_stats()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'memory_used_gb': round(memory.used / (1024**3), 2),
                    'memory_total_gb': round(memory.total / (1024**3), 2),
                    'load_average_1min': load_avg[0],
                    'load_average_5min': load_avg[1],
                    'load_average_15min': load_avg[2],
                    'process_count': process_count
                },
                'docker': {
                    'dockerd_cpu': dockerd_cpu,
                    'mcp_containers_count': mcp_containers,
                    'total_containers': len(container_stats),
                    'container_stats': container_stats
                },
                'io': {
                    'disk_read_mb': round(disk_io.read_bytes / (1024**2), 2) if disk_io else 0,
                    'disk_write_mb': round(disk_io.write_bytes / (1024**2), 2) if disk_io else 0,
                    'network_sent_mb': round(network_io.bytes_sent / (1024**2), 2) if network_io else 0,
                    'network_recv_mb': round(network_io.bytes_recv / (1024**2), 2) if network_io else 0
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {}
    
    def get_container_stats(self) -> List[Dict]:
        """Get Docker container resource statistics"""
        try:
            result = subprocess.run(
                "docker stats --no-stream --format 'json'",
                shell=True, capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            stats = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        container_data = json.loads(line)
                        # Parse CPU percentage
                        cpu_str = container_data.get('CPUPerc', '0.00%').replace('%', '')
                        cpu_percent = float(cpu_str) if cpu_str != '--' else 0.0
                        
                        # Parse memory usage
                        mem_usage = container_data.get('MemUsage', '0B / 0B')
                        mem_parts = mem_usage.split(' / ')
                        if len(mem_parts) == 2:
                            used_mem = self.parse_memory_string(mem_parts[0])
                            total_mem = self.parse_memory_string(mem_parts[1])
                        else:
                            used_mem = total_mem = 0
                        
                        stats.append({
                            'name': container_data.get('Name', 'unknown'),
                            'cpu_percent': cpu_percent,
                            'memory_used_mb': round(used_mem / (1024**2), 2),
                            'memory_limit_mb': round(total_mem / (1024**2), 2)
                        })
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse container stats line: {e}")
                        continue
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get container stats: {e}")
            return []
    
    def parse_memory_string(self, mem_str: str) -> float:
        """Parse memory string like '123.4MiB' to bytes"""
        try:
            mem_str = mem_str.strip()
            if mem_str.endswith('B'):
                mem_str = mem_str[:-1]
            
            if mem_str.endswith('Ki'):
                return float(mem_str[:-2]) * 1024
            elif mem_str.endswith('Mi'):
                return float(mem_str[:-2]) * 1024 * 1024
            elif mem_str.endswith('Gi'):
                return float(mem_str[:-2]) * 1024 * 1024 * 1024
            elif mem_str.endswith('Ti'):
                return float(mem_str[:-2]) * 1024 * 1024 * 1024 * 1024
            else:
                return float(mem_str)
        except:
            return 0.0
    
    def check_alerts(self, metrics: Dict[str, any]) -> List[str]:
        """Check for performance alerts"""
        alerts = []
        
        system = metrics.get('system', {})
        docker = metrics.get('docker', {})
        
        # CPU usage alert
        if system.get('cpu_usage', 0) > self.alert_thresholds['cpu_usage']:
            alerts.append(f"HIGH CPU USAGE: {system['cpu_usage']:.1f}%")
        
        # Memory usage alert
        if system.get('memory_usage', 0) > self.alert_thresholds['memory_usage']:
            alerts.append(f"HIGH MEMORY USAGE: {system['memory_usage']:.1f}%")
        
        # Load average alert
        if system.get('load_average_1min', 0) > self.alert_thresholds['load_average_1min']:
            alerts.append(f"HIGH LOAD AVERAGE: {system['load_average_1min']:.2f}")
        
        # Docker daemon CPU alert
        if docker.get('dockerd_cpu', 0) > self.alert_thresholds['dockerd_cpu']:
            alerts.append(f"HIGH DOCKERD CPU: {docker['dockerd_cpu']:.1f}%")
        
        # Process count alert
        if system.get('process_count', 0) > self.alert_thresholds['process_count']:
            alerts.append(f"HIGH PROCESS COUNT: {system['process_count']}")
        
        return alerts
    
    def save_metrics(self, metrics: Dict[str, any], alerts: List[str]) -> None:
        """Save metrics to file"""
        try:
            # Ensure logs directory exists
            os.makedirs('/opt/sutazaiapp/logs', exist_ok=True)
            
            # Save to timestamped file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = f'/opt/sutazaiapp/logs/hardware_metrics_{timestamp}.json'
            
            data = {
                'metrics': metrics,
                'alerts': alerts,
                'collected_at': datetime.now().isoformat()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Also save to latest file
            with open('/opt/sutazaiapp/logs/hardware_metrics_latest.json', 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def generate_performance_report(self, metrics: Dict[str, any]) -> str:
        """Generate human-readable performance report"""
        system = metrics.get('system', {})
        docker = metrics.get('docker', {})
        io = metrics.get('io', {})
        
        report = f"""
🔍 HARDWARE PERFORMANCE REPORT - {metrics['timestamp']}

📊 SYSTEM METRICS:
  CPU Usage: {system.get('cpu_usage', 0):.1f}%
  Memory Usage: {system.get('memory_usage', 0):.1f}% ({system.get('memory_used_gb', 0):.1f}GB / {system.get('memory_total_gb', 0):.1f}GB)
  Load Average: {system.get('load_average_1min', 0):.2f} (1m) | {system.get('load_average_5min', 0):.2f} (5m) | {system.get('load_average_15min', 0):.2f} (15m)
  Process Count: {system.get('process_count', 0)}

🐳 DOCKER METRICS:
  Docker Daemon CPU: {docker.get('dockerd_cpu', 0):.1f}%
  MCP Containers: {docker.get('mcp_containers_count', 0)}
  Total Containers: {docker.get('total_containers', 0)}

💾 I/O METRICS:
  Disk Read: {io.get('disk_read_mb', 0):.1f}MB
  Disk Write: {io.get('disk_write_mb', 0):.1f}MB
  Network Sent: {io.get('network_sent_mb', 0):.1f}MB
  Network Received: {io.get('network_recv_mb', 0):.1f}MB
        """
        
        # Add container details if available
        container_stats = docker.get('container_stats', [])
        if container_stats:
            report += "\n📦 TOP RESOURCE CONSUMING CONTAINERS:\n"
            # Sort by CPU usage
            top_containers = sorted(container_stats, key=lambda x: x['cpu_percent'], reverse=True)[:10]
            for container in top_containers:
                report += f"  {container['name']}: {container['cpu_percent']:.1f}% CPU, {container['memory_used_mb']:.1f}MB RAM\n"
        
        return report
    
    def monitor_continuous(self, interval: int = 60, duration: int = 3600) -> None:
        """Run continuous monitoring"""
        logger.info(f"🚀 Starting continuous hardware monitoring (interval: {interval}s, duration: {duration}s)")
        
        start_time = time.time()
        cycle_count = 0
        
        try:
            while time.time() - start_time < duration:
                cycle_count += 1
                logger.info(f"📊 Monitoring cycle #{cycle_count}")
                
                # Collect metrics
                metrics = self.collect_metrics()
                if not metrics:
                    logger.warning("Failed to collect metrics, skipping cycle")
                    time.sleep(interval)
                    continue
                
                # Check for alerts
                alerts = self.check_alerts(metrics)
                
                # Log alerts if any
                if alerts:
                    logger.warning(f"⚠️  ALERTS DETECTED: {', '.join(alerts)}")
                else:
                    logger.info("✅ All metrics within normal ranges")
                
                # Save metrics
                self.save_metrics(metrics, alerts)
                
                # Generate and log report
                report = self.generate_performance_report(metrics)
                logger.info(report)
                
                # Store in history (keep last 100 entries)
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring failed: {e}")
        
        logger.info(f"✅ Monitoring completed after {cycle_count} cycles")
    
    def single_snapshot(self) -> Dict[str, any]:
        """Take a single performance snapshot"""
        logger.info("📊 Taking hardware performance snapshot")
        
        metrics = self.collect_metrics()
        if not metrics:
            logger.error("Failed to collect metrics")
            return {}
        
        alerts = self.check_alerts(metrics)
        self.save_metrics(metrics, alerts)
        
        report = self.generate_performance_report(metrics)
        logger.info(report)
        
        if alerts:
            logger.warning(f"⚠️  ALERTS: {', '.join(alerts)}")
        else:
            logger.info("✅ All metrics normal")
        
        return {
            'metrics': metrics,
            'alerts': alerts,
            'report': report
        }

def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        logger.info("Usage:")
        logger.info("  python3 hardware_performance_monitor.py snapshot")
        logger.info("  python3 hardware_performance_monitor.py monitor [interval_seconds] [duration_seconds]")
        sys.exit(1)
    
    # Ensure logs directory exists
    os.makedirs('/opt/sutazaiapp/logs', exist_ok=True)
    
    monitor = HardwarePerformanceMonitor()
    
    if sys.argv[1] == 'snapshot':
        result = monitor.single_snapshot()
        logger.info(json.dumps(result, indent=2))
    
    elif sys.argv[1] == 'monitor':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
        monitor.monitor_continuous(interval, duration)
    
    else:
        logger.info("Invalid command. Use 'snapshot' or 'monitor'")
        sys.exit(1)

if __name__ == "__main__":
    main()