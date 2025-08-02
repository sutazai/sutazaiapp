#!/usr/bin/env python3
"""
SutazAI System Logger
Comprehensive logging and monitoring system
"""

import os
import json
import time
import psutil
import docker
import logging
import subprocess
from datetime import datetime
from typing import Dict, List
from logging.handlers import RotatingFileHandler
import threading

class SystemLogger:
    def __init__(self):
        self.setup_logging()
        self.docker_client = docker.from_env()
        self.log_dir = "/var/log/sutazai"
        self.ensure_log_directory()
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create logger
        self.logger = logging.getLogger('SutazAI-System')
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            '/var/log/sutazai-system.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
    def ensure_log_directory(self):
        """Ensure log directory exists"""
        os.makedirs(self.log_dir, exist_ok=True)
        
    def log_system_metrics(self):
        """Log comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            processes = len(psutil.pids())
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None
                },
                "memory": {
                    "total_mb": memory.total // (1024 * 1024),
                    "available_mb": memory.available // (1024 * 1024),
                    "used_mb": memory.used // (1024 * 1024),
                    "percent": memory.percent,
                    "cached_mb": memory.cached // (1024 * 1024)
                },
                "swap": {
                    "total_mb": swap.total // (1024 * 1024),
                    "used_mb": swap.used // (1024 * 1024),
                    "percent": swap.percent
                },
                "disk": {
                    "total_gb": disk.total // (1024**3),
                    "used_gb": disk.used // (1024**3),
                    "free_gb": disk.free // (1024**3),
                    "percent": (disk.used / disk.total) * 100,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "processes": processes
            }
            
            # Log to file
            with open(f"{self.log_dir}/system-metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
                
            # Log warnings for high usage
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                self.logger.warning(f"High memory usage: {memory.percent}%")
            if disk.percent > 90:
                self.logger.warning(f"High disk usage: {(disk.used / disk.total) * 100:.1f}%")
                
        except Exception as e:
            self.logger.error(f"Failed to log system metrics: {e}")
    
    def log_docker_metrics(self):
        """Log Docker container metrics"""
        try:
            containers = self.docker_client.containers.list()
            container_metrics = []
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
                    
                    # Memory metrics
                    memory_usage = stats['memory_stats']['usage']
                    memory_limit = stats['memory_stats']['limit']
                    memory_percent = (memory_usage / memory_limit) * 100.0
                    
                    # Network metrics
                    network_rx = 0
                    network_tx = 0
                    if 'networks' in stats:
                        for interface in stats['networks'].values():
                            network_rx += interface['rx_bytes']
                            network_tx += interface['tx_bytes']
                    
                    metric = {
                        "container_name": container.name,
                        "container_id": container.short_id,
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                        "status": container.status,
                        "cpu_percent": cpu_percent,
                        "memory_usage_mb": memory_usage // (1024 * 1024),
                        "memory_limit_mb": memory_limit // (1024 * 1024),
                        "memory_percent": memory_percent,
                        "network_rx_bytes": network_rx,
                        "network_tx_bytes": network_tx
                    }
                    
                    container_metrics.append(metric)
                    
                    # Log warnings for high usage
                    if cpu_percent > 90:
                        self.logger.warning(f"High CPU usage in {container.name}: {cpu_percent:.1f}%")
                    if memory_percent > 90:
                        self.logger.warning(f"High memory usage in {container.name}: {memory_percent:.1f}%")
                        
                except Exception as e:
                    self.logger.error(f"Failed to get stats for container {container.name}: {e}")
            
            # Save metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "containers": container_metrics
            }
            
            with open(f"{self.log_dir}/docker-metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to log Docker metrics: {e}")
    
    def log_application_logs(self):
        """Collect and centralize application logs"""
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                try:
                    # Get recent logs
                    logs = container.logs(tail=100, since=int(time.time()) - 300)  # Last 5 minutes
                    
                    if logs:
                        log_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "container": container.name,
                            "logs": logs.decode('utf-8', errors='ignore')
                        }
                        
                        # Save to container-specific log file
                        log_file = f"{self.log_dir}/{container.name}.log"
                        with open(log_file, "a") as f:
                            f.write(f"[{log_entry['timestamp']}]\n{log_entry['logs']}\n")
                        
                        # Check for error patterns
                        error_patterns = ['ERROR', 'FATAL', 'Exception', 'Traceback', 'Out of memory']
                        for pattern in error_patterns:
                            if pattern in logs.decode('utf-8', errors='ignore'):
                                self.logger.error(f"Error pattern '{pattern}' found in {container.name} logs")
                                break
                                
                except Exception as e:
                    self.logger.error(f"Failed to get logs for container {container.name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to collect application logs: {e}")
    
    def log_service_health(self):
        """Log service health status"""
        try:
            services = {
                "postgresql": "localhost:5432",
                "redis": "localhost:6379",
                "chromadb": "localhost:8001",
                "qdrant": "localhost:6333",
                "ollama": "localhost:11434",
                "fastapi-backend": "localhost:8000",
                "streamlit-frontend": "localhost:8501",
                "prometheus": "localhost:9090",
                "grafana": "localhost:3000"
            }
            
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "services": {}
            }
            
            for service_name, endpoint in services.items():
                try:
                    host, port = endpoint.split(':')
                    port = int(port)
                    
                    # Try to connect
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    is_healthy = result == 0
                    health_status["services"][service_name] = {
                        "healthy": is_healthy,
                        "endpoint": endpoint,
                        "response_time_ms": None
                    }
                    
                    if not is_healthy:
                        self.logger.warning(f"Service {service_name} is not responding on {endpoint}")
                    
                except Exception as e:
                    health_status["services"][service_name] = {
                        "healthy": False,
                        "endpoint": endpoint,
                        "error": str(e)
                    }
                    self.logger.error(f"Health check failed for {service_name}: {e}")
            
            # Save health status
            with open(f"{self.log_dir}/service-health.jsonl", "a") as f:
                f.write(json.dumps(health_status) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to log service health: {e}")
    
    def analyze_logs(self):
        """Analyze recent logs for patterns and issues"""
        try:
            # Analyze system metrics for trends
            recent_metrics = []
            try:
                with open(f"{self.log_dir}/system-metrics.jsonl", "r") as f:
                    lines = f.readlines()
                    for line in lines[-60:]:  # Last 60 entries (5 hours at 5min intervals)
                        recent_metrics.append(json.loads(line))
            except FileNotFoundError:
                return
            
            if len(recent_metrics) < 10:
                return
            
            # Analyze memory trend
            memory_usage = [m["memory"]["percent"] for m in recent_metrics]
            avg_memory = sum(memory_usage) / len(memory_usage)
            
            if avg_memory > 85:
                self.logger.warning(f"Sustained high memory usage: {avg_memory:.1f}% average")
            
            # Analyze CPU trend
            cpu_usage = [m["cpu"]["percent"] for m in recent_metrics]
            avg_cpu = sum(cpu_usage) / len(cpu_usage)
            
            if avg_cpu > 80:
                self.logger.warning(f"Sustained high CPU usage: {avg_cpu:.1f}% average")
            
            # Check for memory leaks (consistently increasing memory)
            if len(memory_usage) >= 20:
                early_avg = sum(memory_usage[:10]) / 10
                late_avg = sum(memory_usage[-10:]) / 10
                
                if late_avg - early_avg > 10:
                    self.logger.warning(f"Potential memory leak detected: "
                                      f"memory increased from {early_avg:.1f}% to {late_avg:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze logs: {e}")
    
    def cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            current_time = time.time()
            retention_days = 7
            retention_seconds = retention_days * 24 * 3600
            
            for filename in os.listdir(self.log_dir):
                filepath = os.path.join(self.log_dir, filename)
                
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    
                    if file_age > retention_seconds:
                        os.remove(filepath)
                        self.logger.info(f"Removed old log file: {filename}")
                        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
    
    def generate_summary_report(self):
        """Generate daily summary report"""
        try:
            report = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "summary": {
                    "total_containers": len(self.docker_client.containers.list()),
                    "system_health": "good",  # Will be determined by analysis
                    "alerts_generated": 0,
                    "avg_memory_usage": 0,
                    "avg_cpu_usage": 0,
                    "uptime_hours": 0
                },
                "recommendations": []
            }
            
            # Analyze recent metrics for summary
            try:
                with open(f"{self.log_dir}/system-metrics.jsonl", "r") as f:
                    lines = f.readlines()
                    if lines:
                        recent_metrics = [json.loads(line) for line in lines[-288:]]  # Last 24 hours
                        
                        memory_usage = [m["memory"]["percent"] for m in recent_metrics]
                        cpu_usage = [m["cpu"]["percent"] for m in recent_metrics]
                        
                        report["summary"]["avg_memory_usage"] = sum(memory_usage) / len(memory_usage)
                        report["summary"]["avg_cpu_usage"] = sum(cpu_usage) / len(cpu_usage)
                        
                        # Determine health status
                        if report["summary"]["avg_memory_usage"] > 90:
                            report["summary"]["system_health"] = "critical"
                            report["recommendations"].append("High memory usage detected - consider adding more RAM")
                        elif report["summary"]["avg_memory_usage"] > 80:
                            report["summary"]["system_health"] = "warning"
                            report["recommendations"].append("Memory usage is high - monitor closely")
                        
                        if report["summary"]["avg_cpu_usage"] > 80:
                            report["recommendations"].append("High CPU usage - consider optimizing workloads")
            except:
                pass
            
            # Save report
            with open(f"{self.log_dir}/daily-report-{report['date']}.json", "w") as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Generated daily report: {report['summary']['system_health']} health status")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
    
    def run_monitoring_cycle(self):
        """Run one complete monitoring cycle"""
        self.logger.info("Running monitoring cycle")
        
        # Collect metrics
        self.log_system_metrics()
        self.log_docker_metrics()
        self.log_service_health()
        
        # Analyze and cleanup
        self.analyze_logs()
        
        # Cleanup old logs daily
        if datetime.now().hour == 3 and datetime.now().minute < 5:  # 3 AM daily
            self.cleanup_old_logs()
            self.generate_summary_report()
    
    def run(self):
        """Main monitoring loop"""
        self.logger.info("SutazAI System Logger started")
        
        while True:
            try:
                self.run_monitoring_cycle()
                time.sleep(300)  # Run every 5 minutes
                
            except KeyboardInterrupt:
                self.logger.info("System logger stopped")
                break
            except Exception as e:
                self.logger.error(f"Monitoring cycle error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    logger = SystemLogger()
    logger.run()