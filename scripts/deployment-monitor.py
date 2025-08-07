#!/usr/bin/env python3
"""
Ollama Integration Deployment Monitor
Real-time monitoring and automated rollback trigger system
"""

import os
import sys
import json
import time
import logging
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import docker
import psutil
import requests
import yaml
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    deployment_id: str
    check_interval: int = 30
    rollback_timeout: int = 180
    memory_threshold: float = 0.90
    error_rate_threshold: float = 0.05
    response_time_multiplier: float = 2.0
    health_check_failures: int = 3
    circuit_breaker_trips: int = 3

@dataclass
class MetricValue:
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None

class DeploymentMonitor:
    def __init__(self, config: MonitoringConfig, config_file: str):
        self.config = config
        self.config_file = config_file
        self.docker_client = docker.from_env()
        self.rollback_triggered = False
        self.baseline_metrics = {}
        self.current_metrics = {}
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.setup_prometheus_metrics()
        
        # Load monitoring configuration
        self.monitoring_config = self.load_monitoring_config()
        
        logger.info(f"Deployment monitor initialized for: {config.deployment_id}")

    def load_monitoring_config(self) -> Dict[str, Any]:
        """Load monitoring configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load monitoring config: {e}")
            return {}

    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors"""
        self.deployment_phase_gauge = Gauge(
            'deployment_phase_info',
            'Current deployment phase',
            ['deployment_id', 'phase'],
            registry=self.registry
        )
        
        self.agents_deployed_counter = Counter(
            'agents_deployed_total',
            'Total agents deployed',
            ['deployment_id', 'phase', 'status'],
            registry=self.registry
        )
        
        self.system_memory_gauge = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            ['deployment_id'],
            registry=self.registry
        )
        
        self.agent_health_gauge = Gauge(
            'agent_health_score',
            'Agent health score',
            ['deployment_id', 'agent_name'],
            registry=self.registry
        )
        
        self.rollback_triggers_counter = Counter(
            'rollback_triggers_total',
            'Rollback triggers fired',
            ['deployment_id', 'trigger_type'],
            registry=self.registry
        )
        
        self.response_time_histogram = Histogram(
            'task_response_time_seconds',
            'Task response time in seconds',
            ['deployment_id', 'agent_name'],
            registry=self.registry
        )

    def collect_system_metrics(self) -> Dict[str, MetricValue]:
        """Collect system-level metrics"""
        metrics = {}
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics['memory_usage_percent'] = MetricValue(
            value=memory.percent,
            timestamp=datetime.now()
        )
        
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        metrics['cpu_usage_percent'] = MetricValue(
            value=cpu_usage,
            timestamp=datetime.now()
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        metrics['disk_usage_percent'] = MetricValue(
            value=(disk.used / disk.total) * 100,
            timestamp=datetime.now()
        )
        
        return metrics

    def collect_docker_metrics(self) -> Dict[str, MetricValue]:
        """Collect Docker container metrics"""
        metrics = {}
        
        try:
            # Get all agent containers
            containers = self.docker_client.containers.list(
                filters={"name": "*agent*"}
            )
            
            total_containers = len(containers)
            healthy_containers = 0
            
            for container in containers:
                try:
                    # Get container stats
                    stats = container.stats(stream=False)
                    
                    # Memory usage
                    if 'memory' in stats:
                        memory_usage = stats['memory']['usage']
                        memory_limit = stats['memory']['limit']
                        memory_percent = (memory_usage / memory_limit) * 100
                        
                        metrics[f'container_memory_{container.name}'] = MetricValue(
                            value=memory_percent,
                            timestamp=datetime.now(),
                            labels={'container': container.name}
                        )
                    
                    # Health status
                    health = container.attrs.get('State', {}).get('Health', {})
                    if health.get('Status') == 'healthy':
                        healthy_containers += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to collect stats for {container.name}: {e}")
            
            # Overall container health
            if total_containers > 0:
                health_ratio = healthy_containers / total_containers
                metrics['container_health_ratio'] = MetricValue(
                    value=health_ratio,
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Failed to collect Docker metrics: {e}")
        
        return metrics

    def collect_agent_metrics(self) -> Dict[str, MetricValue]:
        """Collect agent-specific metrics"""
        metrics = {}
        
        try:
            # Get agent health from backend API
            response = requests.get(
                "http://localhost:8000/api/agents/health",
                timeout=10
            )
            
            if response.status_code == 200:
                agents_health = response.json()
                
                for agent_name, health_data in agents_health.items():
                    health_score = health_data.get('health_score', 0.0)
                    metrics[f'agent_health_{agent_name}'] = MetricValue(
                        value=health_score,
                        timestamp=datetime.now(),
                        labels={'agent': agent_name}
                    )
                    
                    # Update Prometheus metrics
                    self.agent_health_gauge.labels(
                        deployment_id=self.config.deployment_id,
                        agent_name=agent_name
                    ).set(health_score)
                    
        except Exception as e:
            logger.warning(f"Failed to collect agent metrics: {e}")
        
        return metrics

    def collect_ollama_metrics(self) -> Dict[str, MetricValue]:
        """Collect Ollama integration metrics"""
        metrics = {}
        
        try:
            # Check Ollama service health
            response = requests.get("http://localhost:10104/api/tags", timeout=10)
            ollama_healthy = response.status_code == 200
            
            metrics['ollama_health'] = MetricValue(
                value=1.0 if ollama_healthy else 0.0,
                timestamp=datetime.now()
            )
            
            # Get circuit breaker metrics from Redis if available
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379, decode_responses=True)
                
                # Count circuit breaker trips
                circuit_breaker_keys = r.keys("circuit_breaker:*:trips")
                total_trips = 0
                
                for key in circuit_breaker_keys:
                    trips = r.get(key)
                    if trips:
                        total_trips += int(trips)
                
                metrics['ollama_circuit_breaker_trips'] = MetricValue(
                    value=total_trips,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                logger.debug(f"Redis metrics collection failed: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to collect Ollama metrics: {e}")
        
        return metrics

    def collect_all_metrics(self) -> Dict[str, MetricValue]:
        """Collect all monitoring metrics"""
        all_metrics = {}
        
        # Collect from all sources
        all_metrics.update(self.collect_system_metrics())
        all_metrics.update(self.collect_docker_metrics())
        all_metrics.update(self.collect_agent_metrics())
        all_metrics.update(self.collect_ollama_metrics())
        
        # Update Prometheus metrics
        if 'memory_usage_percent' in all_metrics:
            self.system_memory_gauge.labels(
                deployment_id=self.config.deployment_id
            ).set(all_metrics['memory_usage_percent'].value)
        
        self.current_metrics = all_metrics
        return all_metrics

    def check_rollback_conditions(self, metrics: Dict[str, MetricValue]) -> Optional[str]:
        """Check if any rollback conditions are met"""
        
        # 1. Memory usage check
        if 'memory_usage_percent' in metrics:
            memory_usage = metrics['memory_usage_percent'].value
            if memory_usage > (self.config.memory_threshold * 100):
                return f"memory_exhaustion:{memory_usage:.1f}%"
        
        # 2. Container health check
        if 'container_health_ratio' in metrics:
            health_ratio = metrics['container_health_ratio'].value
            if health_ratio < 0.8:  # Less than 80% healthy
                return f"container_health_failure:{health_ratio:.2f}"
        
        # 3. Ollama service check
        if 'ollama_health' in metrics:
            ollama_health = metrics['ollama_health'].value
            if ollama_health < 1.0:
                return "ollama_service_failure"
        
        # 4. Circuit breaker trips
        if 'ollama_circuit_breaker_trips' in metrics:
            trips = metrics['ollama_circuit_breaker_trips'].value
            if trips > self.config.circuit_breaker_trips:
                return f"circuit_breaker_trips:{int(trips)}"
        
        # 5. Error rate check (if available from logs)
        error_rate = self.calculate_error_rate()
        if error_rate and error_rate > self.config.error_rate_threshold:
            return f"high_error_rate:{error_rate:.3f}"
        
        return None

    def calculate_error_rate(self) -> Optional[float]:
        """Calculate error rate from logs or metrics"""
        try:
            # This would integrate with your logging system
            # For now, return None as placeholder
            return None
        except Exception as e:
            logger.debug(f"Error rate calculation failed: {e}")
            return None

    def trigger_rollback(self, reason: str):
        """Trigger emergency rollback"""
        if self.rollback_triggered:
            logger.warning("Rollback already triggered, ignoring additional trigger")
            return
        
        self.rollback_triggered = True
        
        # Create rollback trigger file
        trigger_file = f"/tmp/rollback_triggered_{self.config.deployment_id}"
        with open(trigger_file, 'w') as f:
            json.dump({
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'deployment_id': self.config.deployment_id,
                'triggered_by': 'deployment_monitor'
            }, f, indent=2)
        
        # Update Prometheus metrics
        self.rollback_triggers_counter.labels(
            deployment_id=self.config.deployment_id,
            trigger_type=reason.split(':')[0]
        ).inc()
        
        # Send alerts
        self.send_alert("CRITICAL", f"Rollback Triggered: {reason}")
        
        # Execute rollback script
        try:
            import subprocess
            rollback_script = "/opt/sutazaiapp/scripts/rollback-ollama-integration.sh"
            subprocess.Popen([rollback_script, "emergency", reason])
            logger.critical(f"Rollback initiated due to: {reason}")
        except Exception as e:
            logger.error(f"Failed to execute rollback script: {e}")

    def send_alert(self, severity: str, message: str):
        """Send alert notifications"""
        alert_data = {
            'severity': severity,
            'message': message,
            'deployment_id': self.config.deployment_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Slack notification
        slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        if slack_webhook:
            try:
                payload = {
                    'text': f"🚨 {severity}: {message}",
                    'username': 'DeploymentMonitor',
                    'icon_emoji': ':warning:'
                }
                requests.post(slack_webhook, json=payload, timeout=5)
            except Exception as e:
                logger.warning(f"Failed to send Slack alert: {e}")
        
        logger.info(f"Alert sent: {severity} - {message}")

    def log_metrics(self, metrics: Dict[str, MetricValue]):
        """Log current metrics"""
        summary = []
        
        if 'memory_usage_percent' in metrics:
            summary.append(f"Memory: {metrics['memory_usage_percent'].value:.1f}%")
        
        if 'cpu_usage_percent' in metrics:
            summary.append(f"CPU: {metrics['cpu_usage_percent'].value:.1f}%")
        
        if 'container_health_ratio' in metrics:
            summary.append(f"Health: {metrics['container_health_ratio'].value:.2f}")
        
        if 'ollama_health' in metrics:
            ollama_status = "OK" if metrics['ollama_health'].value > 0 else "FAIL"
            summary.append(f"Ollama: {ollama_status}")
        
        logger.info(f"Metrics - {' | '.join(summary)}")

    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting deployment monitoring loop")
        
        # Start Prometheus metrics server
        start_http_server(8000, registry=self.registry)
        logger.info("Prometheus metrics server started on :8000")
        
        while not self.rollback_triggered:
            try:
                # Collect metrics
                metrics = self.collect_all_metrics()
                
                # Log metrics
                self.log_metrics(metrics)
                
                # Check rollback conditions
                rollback_reason = self.check_rollback_conditions(metrics)
                if rollback_reason:
                    logger.critical(f"Rollback condition detected: {rollback_reason}")
                    self.trigger_rollback(rollback_reason)
                    break
                
                # Wait for next check
                await asyncio.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay on error
        
        logger.info("Monitoring loop ended")

    def run(self):
        """Run the deployment monitor"""
        try:
            asyncio.run(self.monitor_loop())
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitor crashed: {e}")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description='Ollama Integration Deployment Monitor')
    parser.add_argument('--config', required=True, help='Monitoring configuration file')
    parser.add_argument('--deployment-id', required=True, help='Deployment ID')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--check-interval', type=int, default=30, help='Check interval in seconds')
    parser.add_argument('--memory-threshold', type=float, default=0.90, help='Memory threshold (0-1)')
    parser.add_argument('--error-threshold', type=float, default=0.05, help='Error rate threshold (0-1)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging to file if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    # Create monitoring configuration
    config = MonitoringConfig(
        deployment_id=args.deployment_id,
        check_interval=args.check_interval,
        memory_threshold=args.memory_threshold,
        error_rate_threshold=args.error_threshold
    )
    
    # Create and run monitor
    monitor = DeploymentMonitor(config, args.config)
    monitor.run()

if __name__ == "__main__":
    main()