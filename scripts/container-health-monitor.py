#!/usr/bin/env python3
"""
Purpose: Real-time container health monitoring during cleanup operations
Usage: python container-health-monitor.py [--watch-cleanup] [--alert-threshold 90]
Requirements: Python 3.8+, docker, psutil

Monitors container health during infrastructure changes with automated rollback.
"""

import os
import sys
import json
import time
import subprocess
import threading
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/health-monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContainerHealthMonitor:
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.monitoring = False
        self.health_data = {}
        self.alert_threshold = 90  # CPU/Memory percentage
        self.critical_services = {
            'backend', 'frontend', 'ollama', 'postgres', 'redis', 
            'nginx', 'monitoring', 'health-monitor'
        }
        self.rollback_triggered = False
        
    def get_docker_stats(self) -> Dict:
        """Get Docker container statistics"""
        try:
            # Get running containers
            result = subprocess.run([
                'docker', 'ps', '--format', 
                'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return {'error': 'Docker not available'}
                
            containers = {}
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        name = parts[0]
                        status = parts[1]
                        ports = parts[2] if len(parts) > 2 else ''
                        
                        containers[name] = {
                            'status': status,
                            'ports': ports,
                            'is_critical': any(crit in name for crit in self.critical_services)
                        }
            
            # Get detailed stats for running containers
            for name in containers.keys():
                try:
                    stats_result = subprocess.run([
                        'docker', 'stats', name, '--no-stream', 
                        '--format', 'table {{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}'
                    ], capture_output=True, text=True, timeout=10)
                    
                    if stats_result.returncode == 0:
                        stats_lines = stats_result.stdout.strip().split('\n')
                        if len(stats_lines) > 1:
                            stats_data = stats_lines[1].split('\t')
                            if len(stats_data) >= 3:
                                containers[name].update({
                                    'cpu_percent': stats_data[0],
                                    'memory_usage': stats_data[1],
                                    'network_io': stats_data[2]
                                })
                                
                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout getting stats for {name}")
                except Exception as e:
                    logger.warning(f"Error getting stats for {name}: {e}")
            
            return containers
            
        except Exception as e:
            logger.error(f"Error getting Docker stats: {e}")
            return {'error': str(e)}
    
    def get_system_resources(self) -> Dict:
        """Get system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {'error': str(e)}
    
    def check_service_health(self) -> Dict:
        """Check health of critical services"""
        health_status = {}
        
        # Check HTTP endpoints
        endpoints = {
            'backend': 'http://localhost:8000/health',
            'frontend': 'http://localhost:3000',
            'grafana': 'http://localhost:3001',
            'prometheus': 'http://localhost:9090/-/healthy'
        }
        
        for service, url in endpoints.items():
            try:
                result = subprocess.run([
                    'curl', '-s', '-f', '--max-time', '5', url
                ], capture_output=True, text=True, timeout=10)
                
                health_status[service] = {
                    'status': 'healthy' if result.returncode == 0 else 'unhealthy',
                    'response_code': result.returncode,
                    'response_time': 'unknown'  # Could add timing
                }
                
            except Exception as e:
                health_status[service] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Check database connections
        db_services = {
            'postgres': 'SELECT 1;',
            'redis': 'redis-cli ping'
        }
        
        for service, command in db_services.items():
            try:
                if service == 'postgres':
                    result = subprocess.run([
                        'docker', 'exec', 'postgres', 'psql', 
                        '-U', 'postgres', '-c', command
                    ], capture_output=True, text=True, timeout=5)
                elif service == 'redis':
                    result = subprocess.run([
                        'docker', 'exec', 'redis', 'redis-cli', 'ping'
                    ], capture_output=True, text=True, timeout=5)
                
                health_status[service] = {
                    'status': 'healthy' if result.returncode == 0 else 'unhealthy',
                    'response': result.stdout.strip()
                }
                
            except Exception as e:
                health_status[service] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return health_status
    
    def analyze_health_trends(self) -> Dict:
        """Analyze health trends and predict issues"""
        if len(self.health_data) < 3:
            return {'status': 'insufficient_data'}
        
        # Get last 3 measurements
        recent_data = list(self.health_data.values())[-3:]
        
        trends = {
            'cpu_trend': 'stable',
            'memory_trend': 'stable',
            'container_failures': 0,
            'recommendations': []
        }
        
        # Analyze CPU trend
        cpu_values = [data.get('system', {}).get('cpu_percent', 0) for data in recent_data]
        if len(cpu_values) >= 3:
            if cpu_values[-1] > cpu_values[-2] > cpu_values[-3]:
                trends['cpu_trend'] = 'increasing'
                if cpu_values[-1] > self.alert_threshold:
                    trends['recommendations'].append('High CPU usage detected - consider scaling')
            elif cpu_values[-1] < cpu_values[-2] < cpu_values[-3]:
                trends['cpu_trend'] = 'decreasing'
        
        # Analyze memory trend
        mem_values = [data.get('system', {}).get('memory_percent', 0) for data in recent_data]
        if len(mem_values) >= 3:
            if mem_values[-1] > mem_values[-2] > mem_values[-3]:
                trends['memory_trend'] = 'increasing'
                if mem_values[-1] > self.alert_threshold:
                    trends['recommendations'].append('High memory usage - check for memory leaks')
        
        # Count container failures
        latest_containers = recent_data[-1].get('containers', {})
        for name, info in latest_containers.items():
            if 'unhealthy' in info.get('status', '').lower():
                trends['container_failures'] += 1
        
        return trends
    
    def trigger_rollback(self, reason: str):
        """Trigger automated rollback"""
        if self.rollback_triggered:
            return  # Already triggered
            
        self.rollback_triggered = True
        logger.critical(f"🚨 TRIGGERING ROLLBACK: {reason}")
        
        try:
            # Stop monitoring
            self.monitoring = False
            
            # Execute rollback based on context
            rollback_scripts = [
                '/opt/sutazaiapp/archive/requirements_cleanup_*/rollback.sh',
                '/opt/sutazaiapp/archive/container_validation_*/rollback.sh'
            ]
            
            for script_pattern in rollback_scripts:
                import glob
                scripts = glob.glob(script_pattern)
                for script in scripts:
                    if os.path.exists(script):
                        logger.info(f"Executing rollback script: {script}")
                        subprocess.run(['bash', script], timeout=300)
                        break
            
            logger.info("✅ Rollback completed")
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
    
    def collect_health_data(self) -> Dict:
        """Collect comprehensive health data"""
        timestamp = datetime.datetime.now().isoformat()
        
        data = {
            'timestamp': timestamp,
            'containers': self.get_docker_stats(),
            'system': self.get_system_resources(),
            'services': self.check_service_health(),
            'trends': self.analyze_health_trends()
        }
        
        # Store for trend analysis
        self.health_data[timestamp] = data
        
        # Keep only last 20 measurements
        if len(self.health_data) > 20:
            oldest_key = min(self.health_data.keys())
            del self.health_data[oldest_key]
        
        return data
    
    def check_alert_conditions(self, data: Dict) -> List[str]:
        """Check for alert conditions"""
        alerts = []
        
        # System resource alerts
        system = data.get('system', {})
        if system.get('cpu_percent', 0) > self.alert_threshold:
            alerts.append(f"High CPU usage: {system['cpu_percent']:.1f}%")
        
        if system.get('memory_percent', 0) > self.alert_threshold:
            alerts.append(f"High memory usage: {system['memory_percent']:.1f}%")
        
        if system.get('disk_percent', 0) > 90:
            alerts.append(f"Low disk space: {system['disk_percent']:.1f}% used")
        
        # Container alerts
        containers = data.get('containers', {})
        unhealthy_critical = []
        
        for name, info in containers.items():
            if info.get('is_critical', False):
                if 'unhealthy' in info.get('status', '').lower():
                    unhealthy_critical.append(name)
                    
        if unhealthy_critical:
            alerts.append(f"Critical containers unhealthy: {', '.join(unhealthy_critical)}")
        
        # Service alerts
        services = data.get('services', {})
        failed_services = [name for name, info in services.items() 
                          if info.get('status') != 'healthy']
        
        if failed_services:
            alerts.append(f"Services failing health checks: {', '.join(failed_services)}")
        
        return alerts
    
    def monitor_loop(self, watch_cleanup: bool = False):
        """Main monitoring loop"""
        logger.info("🚀 Starting container health monitoring...")
        self.monitoring = True
        
        consecutive_alerts = 0
        max_consecutive_alerts = 3
        
        while self.monitoring:
            try:
                # Collect health data
                data = self.collect_health_data()
                
                # Check for alerts
                alerts = self.check_alert_conditions(data)
                
                if alerts:
                    consecutive_alerts += 1
                    logger.warning(f"⚠️  Health alerts ({consecutive_alerts}/{max_consecutive_alerts}):")
                    for alert in alerts:
                        logger.warning(f"  - {alert}")
                    
                    # Trigger rollback if too many consecutive alerts
                    if consecutive_alerts >= max_consecutive_alerts and watch_cleanup:
                        self.trigger_rollback(f"Too many consecutive alerts: {alerts}")
                        break
                        
                else:
                    consecutive_alerts = 0
                    logger.info("✅ All systems healthy")
                
                # Log summary
                system = data.get('system', {})
                container_count = len(data.get('containers', {}))
                
                logger.info(f"📊 System: CPU {system.get('cpu_percent', 0):.1f}%, "
                           f"Memory {system.get('memory_percent', 0):.1f}%, "
                           f"Containers: {container_count}")
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("👋 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(10)  # Wait before retry
        
        self.monitoring = False
        logger.info("🛑 Health monitoring stopped")
    
    def generate_health_report(self) -> str:
        """Generate current health report"""
        data = self.collect_health_data()
        
        report = f"""# Container Health Report
Generated: {data['timestamp']}

## System Resources
- CPU Usage: {data['system'].get('cpu_percent', 0):.1f}%
- Memory Usage: {data['system'].get('memory_percent', 0):.1f}%
- Disk Usage: {data['system'].get('disk_percent', 0):.1f}%
- Available Memory: {data['system'].get('memory_available_gb', 0):.1f}GB
- Free Disk: {data['system'].get('disk_free_gb', 0):.1f}GB

## Container Status
"""
        
        containers = data.get('containers', {})
        for name, info in containers.items():
            status_icon = "🟢" if "up" in info.get('status', '').lower() else "🔴"
            critical_icon = "⭐" if info.get('is_critical', False) else ""
            
            report += f"- {status_icon} {critical_icon} **{name}**: {info.get('status', 'unknown')}\n"
            if 'cpu_percent' in info:
                report += f"  - CPU: {info['cpu_percent']}, Memory: {info['memory_usage']}\n"
        
        report += f"\n## Service Health Checks\n"
        services = data.get('services', {})
        for name, info in services.items():
            status_icon = "🟢" if info.get('status') == 'healthy' else "🔴"
            report += f"- {status_icon} **{name}**: {info.get('status', 'unknown')}\n"
        
        # Add alerts if any
        alerts = self.check_alert_conditions(data)
        if alerts:
            report += f"\n## 🚨 Active Alerts\n"
            for alert in alerts:
                report += f"- {alert}\n"
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Container health monitoring")
    parser.add_argument("--watch-cleanup", action="store_true",
                       help="Monitor during cleanup with auto-rollback")
    parser.add_argument("--alert-threshold", type=int, default=90,
                       help="CPU/Memory alert threshold percentage")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate health report and exit")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs("/opt/sutazaiapp/logs", exist_ok=True)
    
    try:
        monitor = ContainerHealthMonitor()
        monitor.alert_threshold = args.alert_threshold
        
        if args.report_only:
            # Generate and save report
            report = monitor.generate_health_report()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"/opt/sutazaiapp/reports/health_report_{timestamp}.md"
            
            with open(report_path, 'w') as f:
                f.write(report)
                
            print(report)
            print(f"\n📄 Report saved: {report_path}")
        else:
            # Start monitoring
            monitor.monitor_loop(watch_cleanup=args.watch_cleanup)
            
    except Exception as e:
        logger.error(f"Health monitor failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()