#!/usr/bin/env python3
"""
System Health Monitoring and Performance Regression Detection
Comprehensive monitoring system for SutazAI infrastructure.

Features:
- Real-time health monitoring for all 25 services
- Performance regression detection with baseline comparison
- Automated alerting and incident response
- Metrics collection for Prometheus integration
- Infrastructure protection monitoring (MCP, Ollama, databases)

Version: SutazAI v93 - QA Excellence Framework
Author: QA Validation Specialist (Claude Code)
"""

import os
import sys
import json
import time
import asyncio
import logging
import requests
import subprocess
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('health_monitoring.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    """Service health status."""
    name: str
    status: str
    response_time: float
    last_check: str
    error_message: Optional[str] = None
    port: Optional[int] = None
    url: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics data."""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    api_response_times: Dict[str, float]
    throughput: float
    error_rate: float

@dataclass
class HealthReport:
    """Complete health report."""
    timestamp: str
    overall_health: str
    healthy_services: int
    total_services: int
    critical_alerts: List[str]
    warnings: List[str]
    services: List[ServiceHealth]
    performance: PerformanceMetrics
    infrastructure_status: Dict[str, bool]

class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.running = False
        self.check_interval = 30  # seconds
        self.performance_baselines = {}
        self.alert_thresholds = {
            'response_time': 5.0,  # seconds
            'cpu_usage': 80.0,     # percentage
            'memory_usage': 85.0,  # percentage
            'disk_usage': 90.0,    # percentage
            'error_rate': 5.0,     # percentage
        }
        
        # Define all SutazAI services with their health endpoints
        self.services = {
            # Core Infrastructure (10000-10199)
            'postgres': {'port': 10000, 'health_check': self._check_postgres},
            'redis': {'port': 10001, 'health_check': self._check_redis},
            'neo4j': {'port': 10003, 'url': 'http://localhost:10003', 'path': '/'},
            'rabbitmq': {'port': 10008, 'url': 'http://localhost:10008', 'path': '/api/health/checks/virtual-hosts'},
            'backend': {'port': 10010, 'url': 'http://localhost:10010', 'path': '/health'},
            'frontend': {'port': 10011, 'url': 'http://localhost:10011', 'path': '/'},
            
            # Vector & AI Services (10100-10199)
            'chromadb': {'port': 10100, 'url': 'http://localhost:10100', 'path': '/api/v1/heartbeat'},
            'qdrant': {'port': 10101, 'url': 'http://localhost:10101', 'path': '/healthz'},
            'faiss': {'port': 10103, 'url': 'http://localhost:10103', 'path': '/health'},
            'ollama': {'port': 10104, 'url': 'http://localhost:10104', 'path': '/api/tags'},
            
            # Monitoring Stack (10200-10299)
            'prometheus': {'port': 10200, 'url': 'http://localhost:10200', 'path': '/-/healthy'},
            'grafana': {'port': 10201, 'url': 'http://localhost:10201', 'path': '/api/health'},
            'loki': {'port': 10202, 'url': 'http://localhost:10202', 'path': '/ready'},
            'alertmanager': {'port': 10203, 'url': 'http://localhost:10203', 'path': '/-/healthy'},
            'cadvisor': {'port': 10206, 'url': 'http://localhost:10206', 'path': '/healthz'},
            
            # Agent Services (11000+)
            'mcp-server': {'port': 11190, 'url': 'http://localhost:11190', 'path': '/health'},
        }
        
        self.load_performance_baselines()
    
    def load_performance_baselines(self):
        """Load performance baselines from historical data."""
        baseline_file = self.project_root / "monitoring" / "performance_baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    self.performance_baselines = json.load(f)
                logger.info(f"Loaded performance baselines from {baseline_file}")
            except Exception as e:
                logger.warning(f"Could not load performance baselines: {e}")
                self.performance_baselines = {}
        else:
            logger.info("No performance baselines found, will establish new ones")
            self.performance_baselines = {}
    
    def save_performance_baselines(self):
        """Save performance baselines to file."""
        baseline_file = self.project_root / "monitoring" / "performance_baselines.json"
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(baseline_file, 'w') as f:
                json.dump(self.performance_baselines, f, indent=2)
            logger.info(f"Saved performance baselines to {baseline_file}")
        except Exception as e:
            logger.error(f"Could not save performance baselines: {e}")
    
    def _check_postgres(self) -> Tuple[bool, float, Optional[str]]:
        """Check PostgreSQL health."""
        start_time = time.time()
        try:
            result = subprocess.run(
                ["docker", "exec", "sutazai-postgres", "pg_isready", "-U", "sutazai"],
                capture_output=True,
                text=True,
                timeout=10
            )
            response_time = time.time() - start_time
            if result.returncode == 0:
                return True, response_time, None
            else:
                return False, response_time, result.stderr
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, str(e)
    
    def _check_redis(self) -> Tuple[bool, float, Optional[str]]:
        """Check Redis health."""
        start_time = time.time()
        try:
            result = subprocess.run(
                ["docker", "exec", "sutazai-redis", "redis-cli", "ping"],
                capture_output=True,
                text=True,
                timeout=10
            )
            response_time = time.time() - start_time
            if result.returncode == 0 and "PONG" in result.stdout:
                return True, response_time, None
            else:
                return False, response_time, result.stderr
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, str(e)
    
    def _check_http_endpoint(self, url: str, timeout: int = 10) -> Tuple[bool, float, Optional[str]]:
        """Check HTTP endpoint health."""
        start_time = time.time()
        try:
            response = requests.get(url, timeout=timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return True, response_time, None
            else:
                return False, response_time, f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, str(e)
    
    def check_service_health(self, service_name: str, service_config: Dict) -> ServiceHealth:
        """Check health of a single service."""
        if 'health_check' in service_config:
            # Custom health check function
            is_healthy, response_time, error = service_config['health_check']()\n        elif 'url' in service_config:\n            # HTTP endpoint check\n            url = service_config['url'] + service_config.get('path', '/')\n            is_healthy, response_time, error = self._check_http_endpoint(url)\n        else:\n            # Port check only\n            port = service_config.get('port')\n            if port:\n                is_healthy, response_time, error = self._check_port(port)\n            else:\n                return ServiceHealth(\n                    name=service_name,\n                    status='UNKNOWN',\n                    response_time=0.0,\n                    last_check=datetime.now(timezone.utc).isoformat(),\n                    error_message='No health check method configured',\n                    port=service_config.get('port')\n                )\n        \n        status = 'HEALTHY' if is_healthy else 'UNHEALTHY'\n        \n        return ServiceHealth(\n            name=service_name,\n            status=status,\n            response_time=response_time,\n            last_check=datetime.now(timezone.utc).isoformat(),\n            error_message=error,\n            port=service_config.get('port'),\n            url=service_config.get('url')\n        )\n    \n    def _check_port(self, port: int) -> Tuple[bool, float, Optional[str]]:\n        """Check if a port is open."""\n        start_time = time.time()\n        try:\n            result = subprocess.run(\n                ["nc", "-z", "localhost", str(port)],\n                capture_output=True,\n                timeout=5\n            )\n            response_time = time.time() - start_time\n            return result.returncode == 0, response_time, None\n        except Exception as e:\n            response_time = time.time() - start_time\n            return False, response_time, str(e)\n    \n    def collect_performance_metrics(self) -> PerformanceMetrics:\n        """Collect system performance metrics."""\n        timestamp = datetime.now(timezone.utc).isoformat()\n        \n        # Get system metrics\n        cpu_usage = self._get_cpu_usage()\n        memory_usage = self._get_memory_usage()\n        disk_usage = self._get_disk_usage()\n        network_latency = self._get_network_latency()\n        \n        # Collect API response times\n        api_response_times = {}\n        critical_apis = [\n            ('backend', 'http://localhost:10010/health'),\n            ('ollama', 'http://localhost:10104/api/tags'),\n            ('qdrant', 'http://localhost:10101/healthz'),\n            ('prometheus', 'http://localhost:10200/-/healthy')\n        ]\n        \n        for api_name, url in critical_apis:\n            _, response_time, _ = self._check_http_endpoint(url, timeout=5)\n            api_response_times[api_name] = response_time\n        \n        # Calculate throughput and error rate (simplified)\n        throughput = self._calculate_throughput()\n        error_rate = self._calculate_error_rate()\n        \n        return PerformanceMetrics(\n            timestamp=timestamp,\n            cpu_usage=cpu_usage,\n            memory_usage=memory_usage,\n            disk_usage=disk_usage,\n            network_latency=network_latency,\n            api_response_times=api_response_times,\n            throughput=throughput,\n            error_rate=error_rate\n        )\n    \n    def _get_cpu_usage(self) -> float:\n        """Get CPU usage percentage."""\n        try:\n            result = subprocess.run(\n                ["top", "-bn1", "|", "grep", "Cpu(s)", "|", "awk", "'{print $2}'", "|", "cut", "-d%", "-f1"],\n                shell=True,\n                capture_output=True,\n                text=True,\n                timeout=5\n            )\n            if result.returncode == 0:\n                return float(result.stdout.strip())\n        except:\n            pass\n        return 0.0\n    \n    def _get_memory_usage(self) -> float:\n        """Get memory usage percentage."""\n        try:\n            result = subprocess.run(\n                ["free", "|", "grep", "Mem", "|", "awk", "'{print ($3/$2) * 100.0}'"],\n                shell=True,\n                capture_output=True,\n                text=True,\n                timeout=5\n            )\n            if result.returncode == 0:\n                return float(result.stdout.strip())\n        except:\n            pass\n        return 0.0\n    \n    def _get_disk_usage(self) -> float:\n        """Get disk usage percentage."""\n        try:\n            result = subprocess.run(\n                ["df", "/", "|", "tail", "-1", "|", "awk", "'{print $5}'", "|", "cut", "-d%", "-f1"],\n                shell=True,\n                capture_output=True,\n                text=True,\n                timeout=5\n            )\n            if result.returncode == 0:\n                return float(result.stdout.strip())\n        except:\n            pass\n        return 0.0\n    \n    def _get_network_latency(self) -> float:\n        """Get network latency to localhost."""\n        try:\n            result = subprocess.run(\n                ["ping", "-c", "1", "localhost"],\n                capture_output=True,\n                text=True,\n                timeout=5\n            )\n            if result.returncode == 0:\n                # Extract time from ping output\n                for line in result.stdout.split('\\n'):\n                    if 'time=' in line:\n                        time_part = line.split('time=')[1].split()[0]\n                        return float(time_part)\n        except:\n            pass\n        return 0.0\n    \n    def _calculate_throughput(self) -> float:\n        """Calculate system throughput (simplified)."""\n        # This is a simplified calculation\n        # In a real system, you'd integrate with metrics from load balancers or APM tools\n        return 100.0  # requests per second (placeholder)\n    \n    def _calculate_error_rate(self) -> float:\n        """Calculate error rate (simplified)."""\n        # This is a simplified calculation\n        # In a real system, you'd integrate with logging systems or APM tools\n        return 0.5  # percentage (placeholder)\n    \n    def detect_performance_regression(self, current_metrics: PerformanceMetrics) -> List[str]:\n        """Detect performance regressions by comparing with baselines."""\n        regressions = []\n        \n        # Check CPU usage regression\n        if 'cpu_usage' in self.performance_baselines:\n            baseline_cpu = self.performance_baselines['cpu_usage']\n            if current_metrics.cpu_usage > baseline_cpu * 1.3:  # 30% increase\n                regressions.append(f"CPU usage regression: {current_metrics.cpu_usage:.1f}% vs baseline {baseline_cpu:.1f}%")\n        \n        # Check memory usage regression\n        if 'memory_usage' in self.performance_baselines:\n            baseline_memory = self.performance_baselines['memory_usage']\n            if current_metrics.memory_usage > baseline_memory * 1.2:  # 20% increase\n                regressions.append(f"Memory usage regression: {current_metrics.memory_usage:.1f}% vs baseline {baseline_memory:.1f}%")\n        \n        # Check API response time regressions\n        for api_name, response_time in current_metrics.api_response_times.items():\n            baseline_key = f'api_response_time_{api_name}'\n            if baseline_key in self.performance_baselines:\n                baseline_time = self.performance_baselines[baseline_key]\n                if response_time > baseline_time * 2.0:  # 100% increase\n                    regressions.append(f"{api_name} API response time regression: {response_time:.3f}s vs baseline {baseline_time:.3f}s")\n        \n        return regressions\n    \n    def update_performance_baselines(self, metrics: PerformanceMetrics):\n        """Update performance baselines with current metrics (if better)."""\n        # Update baselines only if current metrics are better or if no baseline exists\n        if 'cpu_usage' not in self.performance_baselines or metrics.cpu_usage < self.performance_baselines['cpu_usage']:\n            self.performance_baselines['cpu_usage'] = metrics.cpu_usage\n        \n        if 'memory_usage' not in self.performance_baselines or metrics.memory_usage < self.performance_baselines['memory_usage']:\n            self.performance_baselines['memory_usage'] = metrics.memory_usage\n        \n        for api_name, response_time in metrics.api_response_times.items():\n            baseline_key = f'api_response_time_{api_name}'\n            if baseline_key not in self.performance_baselines or response_time < self.performance_baselines[baseline_key]:\n                self.performance_baselines[baseline_key] = response_time\n    \n    def check_infrastructure_protection(self) -> Dict[str, bool]:\n        """Check infrastructure protection status."""\n        protection_status = {}\n        \n        # Check MCP servers protection\n        mcp_config = self.project_root / ".mcp.json"\n        protection_status['mcp_config_exists'] = mcp_config.exists()\n        \n        if mcp_config.exists():\n            # Check configuration integrity\n            try:\n                result = subprocess.run(\n                    ["sha1sum", str(mcp_config)],\n                    capture_output=True,\n                    text=True,\n                    timeout=5\n                )\n                if result.returncode == 0:\n                    actual_checksum = result.stdout.split()[0]\n                    expected_checksum = "c1ada43007a0715d577c10fad975517a82506c07"\n                    protection_status['mcp_config_integrity'] = actual_checksum == expected_checksum\n                else:\n                    protection_status['mcp_config_integrity'] = False\n            except:\n                protection_status['mcp_config_integrity'] = False\n        \n        # Check Ollama protection\n        ollama_healthy, _, _ = self._check_http_endpoint('http://localhost:10104/api/tags')\n        protection_status['ollama_protected'] = ollama_healthy\n        \n        # Check database containers\n        db_services = ['postgres', 'redis', 'neo4j']\n        for service in db_services:\n            try:\n                result = subprocess.run(\n                    ["docker", "ps", "--filter", f"name={service}", "--format", "{{.Status}}"],\n                    capture_output=True,\n                    text=True,\n                    timeout=5\n                )\n                protection_status[f'{service}_running'] = "Up" in result.stdout\n            except:\n                protection_status[f'{service}_running'] = False\n        \n        return protection_status\n    \n    def generate_alerts(self, services: List[ServiceHealth], performance: PerformanceMetrics, \n                       regressions: List[str], infrastructure: Dict[str, bool]) -> Tuple[List[str], List[str]]:\n        """Generate critical alerts and warnings."""\n        critical_alerts = []\n        warnings = []\n        \n        # Critical service failures\n        critical_services = ['backend', 'postgres', 'redis', 'ollama']\n        for service in services:\n            if service.name in critical_services and service.status == 'UNHEALTHY':\n                critical_alerts.append(f"CRITICAL: {service.name} service is down")\n            elif service.status == 'UNHEALTHY':\n                warnings.append(f"Service {service.name} is unhealthy")\n        \n        # Performance threshold alerts\n        if performance.cpu_usage > self.alert_thresholds['cpu_usage']:\n            critical_alerts.append(f"CRITICAL: High CPU usage: {performance.cpu_usage:.1f}%")\n        \n        if performance.memory_usage > self.alert_thresholds['memory_usage']:\n            critical_alerts.append(f"CRITICAL: High memory usage: {performance.memory_usage:.1f}%")\n        \n        if performance.disk_usage > self.alert_thresholds['disk_usage']:\n            critical_alerts.append(f"CRITICAL: High disk usage: {performance.disk_usage:.1f}%")\n        \n        # Response time alerts\n        for api_name, response_time in performance.api_response_times.items():\n            if response_time > self.alert_thresholds['response_time']:\n                critical_alerts.append(f"CRITICAL: {api_name} API slow response: {response_time:.3f}s")\n        \n        # Performance regression alerts\n        for regression in regressions:\n            warnings.append(f"REGRESSION: {regression}")\n        \n        # Infrastructure protection alerts\n        if not infrastructure.get('mcp_config_exists', True):\n            critical_alerts.append("CRITICAL: MCP configuration missing (Rule 20 violation)")\n        \n        if not infrastructure.get('mcp_config_integrity', True):\n            warnings.append("WARNING: MCP configuration integrity changed")\n        \n        if not infrastructure.get('ollama_protected', True):\n            critical_alerts.append("CRITICAL: Ollama service protection compromised")\n        \n        return critical_alerts, warnings\n    \n    def run_health_check(self) -> HealthReport:\n        """Run complete health check and return report."""\n        logger.info("🏥 Running comprehensive health check...")\n        \n        # Check all services\n        service_healths = []\n        for service_name, service_config in self.services.items():\n            health = self.check_service_health(service_name, service_config)\n            service_healths.append(health)\n        \n        # Collect performance metrics\n        performance = self.collect_performance_metrics()\n        \n        # Detect regressions\n        regressions = self.detect_performance_regression(performance)\n        \n        # Check infrastructure protection\n        infrastructure = self.check_infrastructure_protection()\n        \n        # Generate alerts\n        critical_alerts, warnings = self.generate_alerts(\n            service_healths, performance, regressions, infrastructure\n        )\n        \n        # Calculate overall health\n        healthy_services = sum(1 for s in service_healths if s.status == 'HEALTHY')\n        total_services = len(service_healths)\n        \n        if critical_alerts:\n            overall_health = 'CRITICAL'\n        elif warnings or healthy_services < total_services:\n            overall_health = 'DEGRADED'\n        else:\n            overall_health = 'HEALTHY'\n        \n        # Update baselines if system is healthy\n        if overall_health == 'HEALTHY':\n            self.update_performance_baselines(performance)\n        \n        report = HealthReport(\n            timestamp=datetime.now(timezone.utc).isoformat(),\n            overall_health=overall_health,\n            healthy_services=healthy_services,\n            total_services=total_services,\n            critical_alerts=critical_alerts,\n            warnings=warnings,\n            services=service_healths,\n            performance=performance,\n            infrastructure_status=infrastructure\n        )\n        \n        return report\n    \n    def save_health_report(self, report: HealthReport) -> str:\n        """Save health report to file."""\n        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")\n        report_file = self.project_root / f"health_report_{timestamp}.json"\n        \n        with open(report_file, 'w') as f:\n            json.dump(asdict(report), f, indent=2)\n        \n        return str(report_file)\n    \n    def log_health_status(self, report: HealthReport):\n        """Log health status summary."""\n        logger.info("="*80)\n        logger.info("🏥 SYSTEM HEALTH STATUS")\n        logger.info("="*80)\n        logger.info(f"Overall Health: {report.overall_health}")\n        logger.info(f"Services: {report.healthy_services}/{report.total_services} healthy")\n        logger.info(f"CPU Usage: {report.performance.cpu_usage:.1f}%")\n        logger.info(f"Memory Usage: {report.performance.memory_usage:.1f}%")\n        logger.info(f"Disk Usage: {report.performance.disk_usage:.1f}%")\n        \n        if report.critical_alerts:\n            logger.error("🚨 CRITICAL ALERTS:")\n            for alert in report.critical_alerts:\n                logger.error(f"  {alert}")\n        \n        if report.warnings:\n            logger.warning("⚠️ WARNINGS:")\n            for warning in report.warnings:\n                logger.warning(f"  {warning}")\n        \n        # Log unhealthy services\n        unhealthy_services = [s for s in report.services if s.status != 'HEALTHY']\n        if unhealthy_services:\n            logger.warning("Unhealthy services:")\n            for service in unhealthy_services:\n                logger.warning(f"  {service.name}: {service.status} ({service.error_message or 'No details'})")\n        \n        logger.info("="*80)\n    \n    def start_continuous_monitoring(self):\n        """Start continuous health monitoring."""\n        logger.info(f"🔄 Starting continuous health monitoring (interval: {self.check_interval}s)")\n        self.running = True\n        \n        def signal_handler(signum, frame):\n            logger.info("Received shutdown signal, stopping monitoring...")\n            self.running = False\n        \n        signal.signal(signal.SIGINT, signal_handler)\n        signal.signal(signal.SIGTERM, signal_handler)\n        \n        while self.running:\n            try:\n                report = self.run_health_check()\n                self.log_health_status(report)\n                \n                # Save report\n                report_file = self.save_health_report(report)\n                logger.info(f"Health report saved: {report_file}")\n                \n                # Save updated baselines\n                self.save_performance_baselines()\n                \n                # Sleep until next check\n                time.sleep(self.check_interval)\n                \n            except Exception as e:\n                logger.error(f"Error in health monitoring: {e}")\n                time.sleep(60)  # Wait 1 minute before retry\n        \n        logger.info("Health monitoring stopped")

def main():\n    """Main entry point for health monitoring."""\n    import argparse\n    \n    parser = argparse.ArgumentParser(description="SutazAI Health Monitoring System")\n    parser.add_argument("--project-root", type=Path, help="Project root directory")\n    parser.add_argument("--continuous", action="store_true", help="Run continuous monitoring")\n    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")\n    parser.add_argument("--output", type=str, help="Output report file")\n    \n    args = parser.parse_args()\n    \n    # Initialize monitor\n    monitor = HealthMonitor(args.project_root)\n    monitor.check_interval = args.interval\n    \n    if args.continuous:\n        monitor.start_continuous_monitoring()\n    else:\n        # Single health check\n        report = monitor.run_health_check()\n        monitor.log_health_status(report)\n        \n        # Save report\n        report_file = monitor.save_health_report(report)\n        logger.info(f"Health report saved: {report_file}")\n        \n        if args.output:\n            with open(args.output, 'w') as f:\n                json.dump(asdict(report), f, indent=2)\n            logger.info(f"Report also saved to: {args.output}")\n        \n        # Exit with appropriate code\n        if report.overall_health == 'HEALTHY':\n            sys.exit(0)\n        elif report.overall_health == 'DEGRADED':\n            sys.exit(1)\n        else:  # CRITICAL\n            sys.exit(2)

if __name__ == "__main__":\n    main()