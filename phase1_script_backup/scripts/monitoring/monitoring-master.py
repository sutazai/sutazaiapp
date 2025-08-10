#!/usr/bin/env python3
"""
SutazAI Master Monitoring System
Consolidates 38+ monitoring script variations into one unified system
Author: DevOps Manager - Deduplication Operation
Date: August 10, 2025
"""

import asyncio
import json
import time
import logging
import argparse
import requests
import docker
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports" 
CONFIG_FILE = PROJECT_ROOT / "config" / "monitoring-master.json"

# Service definitions
SERVICES = {
    "core": {
        "postgres": {"port": 10000, "endpoint": "/", "type": "database"},
        "redis": {"port": 10001, "endpoint": "/", "type": "cache"},
        "neo4j": {"port": 10002, "endpoint": "/", "type": "graph_db"},
        "ollama": {"port": 10104, "endpoint": "/api/tags", "type": "ai_model"}
    },
    "agents": {
        "hardware-optimizer": {"port": 11110, "endpoint": "/health", "type": "agent"},
        "jarvis-automation": {"port": 11102, "endpoint": "/health", "type": "agent"},
        "ai-orchestrator": {"port": 8589, "endpoint": "/health", "type": "agent"},
        "ollama-integration": {"port": 8090, "endpoint": "/health", "type": "agent"}
    },
    "monitoring": {
        "prometheus": {"port": 10200, "endpoint": "/-/ready", "type": "metrics"},
        "grafana": {"port": 10201, "endpoint": "/api/health", "type": "dashboard"},
        "loki": {"port": 10202, "endpoint": "/ready", "type": "logs"}
    },
    "application": {
        "backend": {"port": 10010, "endpoint": "/health", "type": "api"},
        "frontend": {"port": 10011, "endpoint": "/", "type": "ui"}
    }
}

@dataclass
class ServiceHealth:
    name: str
    status: str  # healthy, unhealthy, unknown
    response_time: float
    error_message: Optional[str] = None
    last_check: str = None
    metadata: Dict[str, Any] = None

@dataclass 
class SystemMetrics:
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    container_count: int
    healthy_services: int
    unhealthy_services: int
    
class MasterMonitor:
    """Unified monitoring system for all SutazAI services."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_FILE
        self.docker_client = docker.from_env()
        self.session = requests.Session()
        self.session.timeout = (5, 10)  # Connect, read timeout
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.load_config()
        
    def setup_logging(self):
        """Configure logging for the monitoring system."""
        LOG_DIR.mkdir(exist_ok=True)
        
        log_file = LOG_DIR / f"monitoring_master_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def load_config(self):
        """Load monitoring configuration."""
        default_config = {
            "check_interval": 30,
            "timeout": 10,
            "retry_count": 3,
            "alert_thresholds": {
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90,
                "response_time": 5.0
            },
            "enable_notifications": False,
            "webhook_url": None
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Error loading config: {e}, using defaults")
        
        self.config = default_config
        
    async def check_service_health(self, service_name: str, service_config: Dict) -> ServiceHealth:
        """Check health of a single service."""
        start_time = time.time()
        port = service_config["port"]
        endpoint = service_config.get("endpoint", "/health")
        service_type = service_config.get("type", "unknown")
        
        try:
            url = f"http://localhost:{port}{endpoint}"
            
            # Special handling for different service types
            if service_type == "database":
                # For databases, just check if port is open
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config["timeout"])
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    status = "healthy"
                    error_message = None
                else:
                    status = "unhealthy"
                    error_message = f"Port {port} not accessible"
                    
            else:
                # HTTP health check
                response = self.session.get(url, timeout=self.config["timeout"])
                
                if response.status_code == 200:
                    status = "healthy"
                    error_message = None
                else:
                    status = "unhealthy" 
                    error_message = f"HTTP {response.status_code}"
                    
        except requests.exceptions.RequestException as e:
            status = "unhealthy"
            error_message = str(e)
        except Exception as e:
            status = "unknown"
            error_message = str(e)
            
        response_time = time.time() - start_time
        
        return ServiceHealth(
            name=service_name,
            status=status,
            response_time=response_time,
            error_message=error_message,
            last_check=datetime.now().isoformat(),
            metadata={"type": service_type, "port": port}
        )
        
    def get_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Container metrics
            containers = self.docker_client.containers.list(all=True)
            container_count = len(containers)
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                container_count=container_count,
                healthy_services=0,  # Will be updated by health check results
                unhealthy_services=0
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0, 
                disk_percent=0.0,
                container_count=0,
                healthy_services=0,
                unhealthy_services=0
            )
            
    async def run_health_checks(self, service_groups: List[str] = None) -> Dict[str, List[ServiceHealth]]:
        """Run health checks for specified service groups."""
        if service_groups is None:
            service_groups = list(SERVICES.keys())
            
        results = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            
            for group in service_groups:
                if group not in SERVICES:
                    self.logger.warning(f"Unknown service group: {group}")
                    continue
                    
                results[group] = []
                
                for service_name, service_config in SERVICES[group].items():
                    future = executor.submit(
                        asyncio.run,
                        self.check_service_health(service_name, service_config)
                    )
                    futures[future] = (group, service_name)
                    
            for future in as_completed(futures):
                group, service_name = futures[future]
                try:
                    health = future.result()
                    results[group].append(health)
                except Exception as e:
                    self.logger.error(f"Error checking {group}.{service_name}: {e}")
                    
        return results
        
    def generate_report(self, health_results: Dict, system_metrics: SystemMetrics) -> Dict:
        """Generate comprehensive monitoring report."""
        # Calculate summary statistics
        total_services = sum(len(services) for services in health_results.values())
        healthy_count = 0
        unhealthy_count = 0
        
        service_details = []
        
        for group, services in health_results.items():
            for service in services:
                if service.status == "healthy":
                    healthy_count += 1
                else:
                    unhealthy_count += 1
                    
                service_details.append({
                    "group": group,
                    "name": service.name,
                    "status": service.status,
                    "response_time": service.response_time,
                    "error": service.error_message,
                    "last_check": service.last_check,
                    "metadata": service.metadata
                })
                
        # Update system metrics with service counts
        system_metrics.healthy_services = healthy_count
        system_metrics.unhealthy_services = unhealthy_count
        
        # Generate alerts
        alerts = []
        if system_metrics.cpu_percent > self.config["alert_thresholds"]["cpu_percent"]:
            alerts.append(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
        if system_metrics.memory_percent > self.config["alert_thresholds"]["memory_percent"]:
            alerts.append(f"High memory usage: {system_metrics.memory_percent:.1f}%")
        if system_metrics.disk_percent > self.config["alert_thresholds"]["disk_percent"]:
            alerts.append(f"High disk usage: {system_metrics.disk_percent:.1f}%")
            
        # Check for slow services
        for service in service_details:
            if service["response_time"] > self.config["alert_thresholds"]["response_time"]:
                alerts.append(f"Slow response: {service['name']} ({service['response_time']:.2f}s)")
                
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_services": total_services,
                "healthy_services": healthy_count,
                "unhealthy_services": unhealthy_count,
                "health_percentage": (healthy_count / total_services * 100) if total_services > 0 else 0
            },
            "system_metrics": asdict(system_metrics),
            "service_details": service_details,
            "alerts": alerts,
            "configuration": self.config
        }
        
        return report
        
    def save_report(self, report: Dict, format: str = "json"):
        """Save monitoring report to file."""
        REPORTS_DIR.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"monitoring_report_{timestamp}.json"
            filepath = REPORTS_DIR / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
                
        elif format == "summary":
            filename = f"monitoring_summary_{timestamp}.txt"
            filepath = REPORTS_DIR / filename
            
            with open(filepath, 'w') as f:
                f.write(f"SutazAI System Health Report - {report['timestamp']}\n")
                f.write("=" * 60 + "\n\n")
                
                summary = report['summary']
                f.write(f"Overall Health: {summary['health_percentage']:.1f}%\n")
                f.write(f"Services: {summary['healthy_services']}/{summary['total_services']} healthy\n\n")
                
                if report['alerts']:
                    f.write("ALERTS:\n")
                    for alert in report['alerts']:
                        f.write(f"  - {alert}\n")
                    f.write("\n")
                    
                # System metrics
                metrics = report['system_metrics']
                f.write(f"System Metrics:\n")
                f.write(f"  CPU: {metrics['cpu_percent']:.1f}%\n")
                f.write(f"  Memory: {metrics['memory_percent']:.1f}%\n") 
                f.write(f"  Disk: {metrics['disk_percent']:.1f}%\n")
                f.write(f"  Containers: {metrics['container_count']}\n\n")
                
                # Service details
                f.write("Service Details:\n")
                for service in report['service_details']:
                    status_icon = "✅" if service['status'] == 'healthy' else "❌"
                    f.write(f"  {status_icon} {service['group']}.{service['name']}")
                    if service['error']:
                        f.write(f" - {service['error']}")
                    f.write("\n")
                    
        self.logger.info(f"Report saved: {filepath}")
        return filepath
        
    async def continuous_monitoring(self, interval: int = None):
        """Run continuous monitoring loop."""
        interval = interval or self.config["check_interval"]
        
        self.logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                # Run health checks
                health_results = await self.run_health_checks()
                
                # Collect system metrics
                system_metrics = self.get_system_metrics()
                
                # Generate report
                report = self.generate_report(health_results, system_metrics)
                
                # Log summary
                summary = report['summary']
                self.logger.info(
                    f"Health check: {summary['healthy_services']}/{summary['total_services']} "
                    f"services healthy ({summary['health_percentage']:.1f}%)"
                )
                
                # Log any alerts
                for alert in report['alerts']:
                    self.logger.warning(alert)
                    
                # Save report
                self.save_report(report, "json")
                
                # Wait for next check
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            raise
            
    async def one_time_check(self, service_groups: List[str] = None, output_format: str = "summary"):
        """Run one-time health check and report."""
        self.logger.info("Running one-time health check")
        
        # Run health checks
        health_results = await self.run_health_checks(service_groups)
        
        # Collect system metrics  
        system_metrics = self.get_system_metrics()
        
        # Generate report
        report = self.generate_report(health_results, system_metrics)
        
        # Save and display report
        report_file = self.save_report(report, output_format)
        
        # Print summary to console
        summary = report['summary']
        print(f"\nSutazAI Health Check Results:")
        print(f"Overall Health: {summary['health_percentage']:.1f}%")
        print(f"Services: {summary['healthy_services']}/{summary['total_services']} healthy")
        
        if report['alerts']:
            print(f"\nAlerts ({len(report['alerts'])}):")
            for alert in report['alerts']:
                print(f"  ⚠️  {alert}")
                
        print(f"\nDetailed report: {report_file}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description="SutazAI Master Monitoring System")
    parser.add_argument("--mode", choices=["check", "monitor"], default="check",
                       help="Run one-time check or continuous monitoring")
    parser.add_argument("--groups", nargs="+", choices=list(SERVICES.keys()),
                       help="Service groups to monitor")
    parser.add_argument("--interval", type=int, default=30,
                       help="Monitoring interval in seconds")
    parser.add_argument("--format", choices=["json", "summary"], default="summary",
                       help="Output format")
    parser.add_argument("--config", type=Path,
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = MasterMonitor(config_path=args.config)
    
    try:
        if args.mode == "check":
            # One-time check
            asyncio.run(monitor.one_time_check(args.groups, args.format))
        else:
            # Continuous monitoring
            asyncio.run(monitor.continuous_monitoring(args.interval))
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())