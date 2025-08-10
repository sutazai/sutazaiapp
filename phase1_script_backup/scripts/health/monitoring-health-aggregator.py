#!/usr/bin/env python3
"""
MONITORING HEALTH AGGREGATOR
Comprehensive health monitoring with metrics collection

Consolidated from:
- system-health-validator.py
- validate-production-health.py
- database_health_check.py
- fix-agent-health-checks.py

Purpose: Advanced health monitoring with metrics and alerting
Author: ULTRA SCRIPT CONSOLIDATION MASTER
"""

import time
import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MonitoringHealthAggregator')

class MonitoringHealthAggregator:
    """Advanced health monitoring with metrics collection and alerting"""
    
    def __init__(self):
        self.project_root = Path('/opt/sutazaiapp')
        self.metrics_history = []
        self.alert_thresholds = {
            'critical_service_down': 1,      # Number of critical services down
            'overall_health_percent': 80,    # Minimum overall health percentage
            'response_time_threshold': 5.0,  # Maximum acceptable response time
            'memory_threshold': 85,          # Memory usage percentage
            'cpu_threshold': 90,             # CPU usage percentage
            'error_rate_threshold': 5        # Error rate percentage
        }
        
        # Metrics collection configuration
        self.metrics_config = {
            'collect_system_metrics': True,
            'collect_application_metrics': True,
            'collect_database_metrics': True,
            'collect_docker_metrics': True,
            'retention_hours': 24
        }
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        import psutil
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        return {
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'load_1min': load_avg[0],
                'load_5min': load_avg[1],
                'load_15min': load_avg[2]
            },
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'percent_used': memory.percent,
                'swap_percent': swap.percent
            },
            'disk': {
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'percent_used': round((disk_usage.used / disk_usage.total) * 100, 2),
                'read_mb': round(disk_io.read_bytes / (1024**2), 2) if disk_io else 0,
                'write_mb': round(disk_io.write_bytes / (1024**2), 2) if disk_io else 0
            },
            'network': {
                'bytes_sent_mb': round(network_io.bytes_sent / (1024**2), 2),
                'bytes_recv_mb': round(network_io.bytes_recv / (1024**2), 2),
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics"""
        app_metrics = {}
        
        # Backend API metrics
        try:
            response = requests.get('http://localhost:10010/metrics', timeout=5)
            if response.status_code == 200:
                # Parse Prometheus metrics (simplified)
                app_metrics['backend'] = {
                    'status': 'healthy',
                    'response_time': response.elapsed.total_seconds(),
                    'metrics_available': True
                }
            else:
                app_metrics['backend'] = {
                    'status': 'degraded',
                    'error': f"HTTP {response.status_code}"
                }
        except Exception as e:
            app_metrics['backend'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Ollama metrics
        try:
            response = requests.get('http://localhost:10104/api/tags', timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                app_metrics['ollama'] = {
                    'status': 'healthy',
                    'model_count': len(models),
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                app_metrics['ollama'] = {
                    'status': 'degraded',
                    'error': f"HTTP {response.status_code}"
                }
        except Exception as e:
            app_metrics['ollama'] = {
                'status': 'error',
                'error': str(e)
            }
        
        return app_metrics
    
    def collect_database_metrics(self) -> Dict[str, Any]:
        """Collect database performance metrics"""
        db_metrics = {}
        
        # PostgreSQL metrics
        try:
            import psycopg2
            conn = psycopg2.connect(
                host='localhost', port=10000, database='sutazai',
                user='sutazai', password='sutazai', connect_timeout=5
            )
            
            cursor = conn.cursor()
            
            # Basic connectivity and table count
            cursor.execute('SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s;', ('public',))
            table_count = cursor.fetchone()[0]
            
            # Database size
            cursor.execute("SELECT pg_size_pretty(pg_database_size('sutazai'));")
            db_size = cursor.fetchone()[0]
            
            # Active connections
            cursor.execute('SELECT COUNT(*) FROM pg_stat_activity;')
            connection_count = cursor.fetchone()[0]
            
            db_metrics['postgresql'] = {
                'status': 'healthy',
                'table_count': table_count,
                'database_size': db_size,
                'active_connections': connection_count
            }
            
            conn.close()
            
        except Exception as e:
            db_metrics['postgresql'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Redis metrics
        try:
            import redis
            r = redis.Redis(host='localhost', port=10001, decode_responses=True, socket_timeout=5)
            info = r.info()
            
            db_metrics['redis'] = {
                'status': 'healthy',
                'used_memory': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
            
        except Exception as e:
            db_metrics['redis'] = {
                'status': 'error',
                'error': str(e)
            }
        
        return db_metrics
    
    def collect_docker_metrics(self) -> Dict[str, Any]:
        """Collect Docker container metrics"""
        try:
            import docker
            client = docker.from_env()
            
            containers = client.containers.list(all=True)
            sutazai_containers = [c for c in containers if 'sutazai' in c.name]
            
            running_count = sum(1 for c in sutazai_containers if c.status == 'running')
            stopped_count = sum(1 for c in sutazai_containers if c.status == 'exited')
            
            # Get resource usage for running containers
            resource_usage = []
            for container in sutazai_containers:
                if container.status == 'running':
                    try:
                        stats = container.stats(stream=False)
                        
                        # Simplified CPU calculation
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                   stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                      stats['precpu_stats']['system_cpu_usage']
                        
                        cpu_percent = 0.0
                        if system_delta > 0:
                            cpu_percent = (cpu_delta / system_delta) * \
                                         len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
                        
                        memory_usage = stats['memory_stats']['usage']
                        memory_limit = stats['memory_stats']['limit']
                        memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
                        
                        resource_usage.append({
                            'name': container.name,
                            'cpu_percent': round(cpu_percent, 2),
                            'memory_mb': round(memory_usage / 1024 / 1024, 2),
                            'memory_percent': round(memory_percent, 2)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not get stats for {container.name}: {e}")
            
            return {
                'total_containers': len(sutazai_containers),
                'running_containers': running_count,
                'stopped_containers': stopped_count,
                'resource_usage': resource_usage
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def aggregate_health_metrics(self) -> Dict[str, Any]:
        """Collect and aggregate all health metrics"""
        logger.info("Collecting comprehensive health metrics...")
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'collection_time': time.time()
        }
        
        # Collect system metrics
        if self.metrics_config['collect_system_metrics']:
            try:
                metrics['system'] = self.collect_system_metrics()
            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")
                metrics['system'] = {'error': str(e)}
        
        # Collect application metrics
        if self.metrics_config['collect_application_metrics']:
            try:
                metrics['applications'] = self.collect_application_metrics()
            except Exception as e:
                logger.error(f"Failed to collect application metrics: {e}")
                metrics['applications'] = {'error': str(e)}
        
        # Collect database metrics
        if self.metrics_config['collect_database_metrics']:
            try:
                metrics['databases'] = self.collect_database_metrics()
            except Exception as e:
                logger.error(f"Failed to collect database metrics: {e}")
                metrics['databases'] = {'error': str(e)}
        
        # Collect Docker metrics
        if self.metrics_config['collect_docker_metrics']:
            try:
                metrics['docker'] = self.collect_docker_metrics()
            except Exception as e:
                logger.error(f"Failed to collect Docker metrics: {e}")
                metrics['docker'] = {'error': str(e)}
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Cleanup old metrics
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_config['retention_hours'])
        self.metrics_history = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        return metrics
    
    def check_alert_conditions(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions in collected metrics"""
        alerts = []
        
        # Check system resource alerts
        if 'system' in metrics and 'error' not in metrics['system']:
            system = metrics['system']
            
            if system['memory']['percent_used'] > self.alert_thresholds['memory_threshold']:
                alerts.append({
                    'level': 'warning',
                    'type': 'system_resource',
                    'message': f"High memory usage: {system['memory']['percent_used']:.1f}%",
                    'value': system['memory']['percent_used'],
                    'threshold': self.alert_thresholds['memory_threshold']
                })
            
            if system['cpu']['percent'] > self.alert_thresholds['cpu_threshold']:
                alerts.append({
                    'level': 'warning',
                    'type': 'system_resource',
                    'message': f"High CPU usage: {system['cpu']['percent']:.1f}%",
                    'value': system['cpu']['percent'],
                    'threshold': self.alert_thresholds['cpu_threshold']
                })
        
        # Check application alerts
        if 'applications' in metrics:
            apps = metrics['applications']
            
            critical_apps_down = 0
            for app_name, app_data in apps.items():
                if app_data.get('status') in ['error', 'down']:
                    if app_name in ['backend', 'ollama']:  # Critical applications
                        critical_apps_down += 1
                        alerts.append({
                            'level': 'critical',
                            'type': 'service_down',
                            'message': f"Critical application {app_name} is down",
                            'service': app_name,
                            'error': app_data.get('error', 'Unknown error')
                        })
                
                # Check response time
                if 'response_time' in app_data:
                    if app_data['response_time'] > self.alert_thresholds['response_time_threshold']:
                        alerts.append({
                            'level': 'warning',
                            'type': 'performance',
                            'message': f"{app_name} slow response time: {app_data['response_time']:.2f}s",
                            'value': app_data['response_time'],
                            'threshold': self.alert_thresholds['response_time_threshold']
                        })
        
        # Check database alerts
        if 'databases' in metrics:
            dbs = metrics['databases']
            
            for db_name, db_data in dbs.items():
                if db_data.get('status') == 'error':
                    alerts.append({
                        'level': 'critical',
                        'type': 'database_error',
                        'message': f"Database {db_name} error: {db_data.get('error', 'Unknown')}",
                        'database': db_name
                    })
        
        return alerts
    
    def generate_health_report(self, metrics: Dict[str, Any], alerts: List[Dict[str, Any]]) -> str:
        """Generate comprehensive health report"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("SUTAZAI MONITORING HEALTH REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Alert summary
        if alerts:
            critical_alerts = [a for a in alerts if a['level'] == 'critical']
            warning_alerts = [a for a in alerts if a['level'] == 'warning']
            
            report_lines.append(f"\n🚨 ALERTS: {len(critical_alerts)} critical, {len(warning_alerts)} warnings")
            
            for alert in critical_alerts:
                report_lines.append(f"  🚨 CRITICAL: {alert['message']}")
            
            for alert in warning_alerts:
                report_lines.append(f"  ⚠️  WARNING: {alert['message']}")
        else:
            report_lines.append("\n✅ NO ALERTS")
        
        # System metrics
        if 'system' in metrics and 'error' not in metrics['system']:
            system = metrics['system']
            report_lines.append(f"\n📊 SYSTEM METRICS:")
            report_lines.append(f"  CPU: {system['cpu']['percent']:.1f}% (Load: {system['cpu']['load_1min']:.2f})")
            report_lines.append(f"  Memory: {system['memory']['percent_used']:.1f}% used ({system['memory']['available_gb']:.1f}GB available)")
            report_lines.append(f"  Disk: {system['disk']['percent_used']:.1f}% used ({system['disk']['free_gb']:.1f}GB free)")
        
        # Application status
        if 'applications' in metrics:
            report_lines.append(f"\n🚀 APPLICATIONS:")
            for app_name, app_data in metrics['applications'].items():
                status_icon = {'healthy': '✅', 'degraded': '⚠️', 'error': '❌'}.get(app_data['status'], '❓')
                report_lines.append(f"  {status_icon} {app_name.title()}: {app_data['status']}")
                
                if 'response_time' in app_data:
                    report_lines.append(f"      Response time: {app_data['response_time']:.2f}s")
        
        # Database status
        if 'databases' in metrics:
            report_lines.append(f"\n💾 DATABASES:")
            for db_name, db_data in metrics['databases'].items():
                status_icon = {'healthy': '✅', 'degraded': '⚠️', 'error': '❌'}.get(db_data['status'], '❓')
                report_lines.append(f"  {status_icon} {db_name.title()}: {db_data['status']}")
        
        # Docker containers
        if 'docker' in metrics and 'error' not in metrics['docker']:
            docker = metrics['docker']
            report_lines.append(f"\n🐳 DOCKER CONTAINERS:")
            report_lines.append(f"  Running: {docker['running_containers']}/{docker['total_containers']}")
            
            if docker['resource_usage']:
                high_cpu_containers = [c for c in docker['resource_usage'] if c['cpu_percent'] > 50]
                high_mem_containers = [c for c in docker['resource_usage'] if c['memory_percent'] > 80]
                
                if high_cpu_containers:
                    report_lines.append(f"  High CPU: {[c['name'] for c in high_cpu_containers]}")
                if high_mem_containers:
                    report_lines.append(f"  High Memory: {[c['name'] for c in high_mem_containers]}")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run a complete monitoring cycle"""
        # Collect metrics
        metrics = self.aggregate_health_metrics()
        
        # Check for alerts
        alerts = self.check_alert_conditions(metrics)
        
        # Generate report
        report = self.generate_health_report(metrics, alerts)
        
        # Save detailed data
        self.save_monitoring_data(metrics, alerts)
        
        return {
            'metrics': metrics,
            'alerts': alerts,
            'report': report,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_monitoring_data(self, metrics: Dict[str, Any], alerts: List[Dict[str, Any]]):
        """Save monitoring data to files"""
        try:
            timestamp = int(time.time())
            
            # Save metrics
            metrics_file = f"/opt/sutazaiapp/logs/monitoring_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save alerts if any
            if alerts:
                alerts_file = f"/opt/sutazaiapp/logs/monitoring_alerts_{timestamp}.json"
                with open(alerts_file, 'w') as f:
                    json.dump(alerts, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to save monitoring data: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SutazAI Monitoring Health Aggregator')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Run continuous monitoring')
    parser.add_argument('--interval', '-i', type=int, default=60,
                       help='Monitoring interval in seconds (default: 60)')
    parser.add_argument('--output', '-o', 
                       help='Save report to file')
    parser.add_argument('--json', action='store_true',
                       help='Output JSON format')
    
    args = parser.parse_args()
    
    # Create aggregator
    aggregator = MonitoringHealthAggregator()
    
    if args.continuous:
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        
        try:
            while True:
                result = aggregator.run_monitoring_cycle()
                
                # Print summary
                alerts = result['alerts']
                critical_count = sum(1 for a in alerts if a['level'] == 'critical')
                warning_count = sum(1 for a in alerts if a['level'] == 'warning')
                
                timestamp = datetime.now().strftime('%H:%M:%S')
                if critical_count > 0:
                    print(f"[{timestamp}] 🚨 {critical_count} critical alerts, {warning_count} warnings")
                elif warning_count > 0:
                    print(f"[{timestamp}] ⚠️ {warning_count} warnings")
                else:
                    print(f"[{timestamp}] ✅ System healthy")
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    else:
        # Single monitoring cycle
        result = aggregator.run_monitoring_cycle()
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(result['report'])
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result['report'])
                f.write("\n\nRaw Data:\n")
                f.write(json.dumps(result, indent=2))
            print(f"\nDetailed report saved to {args.output}")


if __name__ == '__main__':
    main()