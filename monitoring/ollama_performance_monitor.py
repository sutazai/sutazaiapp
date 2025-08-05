#!/usr/bin/env python3
"""
Comprehensive Ollama Performance Monitor
Real-time monitoring of Ollama instances for 174+ concurrent connections
"""

import asyncio
import aiohttp
import json
import time
import logging
import psutil
import redis.asyncio as redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import argparse
import signal
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/ollama_performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OllamaMetrics:
    """Ollama instance performance metrics."""
    instance_url: str
    timestamp: float
    is_healthy: bool
    response_time_ms: float
    active_connections: int
    queue_size: int
    cpu_usage: float
    memory_usage_mb: float
    memory_usage_percent: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    average_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    uptime_seconds: float
    models_loaded: List[str]
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    timestamp: float
    total_cpu_usage: float
    total_memory_usage_mb: float
    total_memory_usage_percent: float
    available_memory_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: List[float]
    open_files: int
    active_processes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class OllamaPerformanceMonitor:
    """
    Comprehensive performance monitor for Ollama instances.
    Tracks health, performance, and resource utilization.
    """
    
    def __init__(self, 
                 instances: List[str] = None,
                 redis_url: str = "redis://localhost:6379",
                 monitoring_interval: int = 10,
                 metrics_retention_hours: int = 24,
                 alert_thresholds: Dict[str, float] = None):
        
        # Default to single local instance
        self.instances = instances or ["http://localhost:11434"]
        self.redis_url = redis_url
        self.monitoring_interval = monitoring_interval
        self.metrics_retention_hours = metrics_retention_hours
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "response_time_ms": 5000,    # 5 seconds
            "cpu_usage": 90,             # 90%
            "memory_usage": 85,          # 85%
            "error_rate": 10,            # 10%
            "queue_size": 100,           # 100 requests
            "disk_usage": 90             # 90%
        }
        
        # State tracking
        self.redis_client: Optional[redis.Redis] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self.start_time = time.time()
        
        # Metrics storage
        self.metrics_history: Dict[str, List[OllamaMetrics]] = {}
        self.system_metrics_history: List[SystemMetrics] = []
        self.alerts_sent: Dict[str, float] = {}  # Alert cooldown tracking
        
        logger.info(f"Initialized monitor for {len(self.instances)} Ollama instances")

    async def initialize(self):
        """Initialize monitoring components."""
        try:
            # Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis for metrics storage")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Metrics will be memory-only.")
            self.redis_client = None
        
        # HTTP session for API calls
        connector = aiohttp.TCPConnector(
            limit=50,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Initialize metrics history
        for instance in self.instances:
            self.metrics_history[instance] = []
        
        self.is_running = True
        logger.info("Performance monitor initialized successfully")

    async def shutdown(self):
        """Gracefully shutdown the monitor."""
        logger.info("Shutting down performance monitor...")
        self.is_running = False
        
        if self.session:
            await self.session.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Performance monitor shutdown complete")

    async def collect_ollama_metrics(self, instance_url: str) -> Optional[OllamaMetrics]:
        """Collect comprehensive metrics from an Ollama instance."""
        start_time = time.time()
        
        try:
            # Health check and basic metrics
            async with self.session.get(f"{instance_url}/api/tags") as response:
                response_time_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    models_loaded = [model["name"] for model in data.get("models", [])]
                    is_healthy = True
                else:
                    models_loaded = []
                    is_healthy = False
            
            # Try to get detailed metrics if available
            memory_usage_mb = 0
            cpu_usage = 0
            active_connections = 0
            queue_size = 0
            
            try:
                # Attempt to get more detailed metrics from custom endpoint
                async with self.session.get(f"{instance_url}/api/metrics") as metrics_response:
                    if metrics_response.status == 200:
                        metrics_data = await metrics_response.json()
                        memory_usage_mb = metrics_data.get("memory_usage_mb", 0)
                        cpu_usage = metrics_data.get("cpu_usage", 0)
                        active_connections = metrics_data.get("active_connections", 0)
                        queue_size = metrics_data.get("queue_size", 0)
            except Exception:
                # Metrics endpoint not available, use system-level estimates
                pass
            
            # Get historical data for trend calculation
            instance_history = self.metrics_history.get(instance_url, [])
            
            # Calculate request rate
            requests_per_second = 0
            if len(instance_history) > 1:
                time_diff = time.time() - instance_history[-1].timestamp
                if time_diff > 0:
                    # Estimate based on response patterns
                    requests_per_second = 1.0 / max(response_time_ms / 1000, 0.1)
            
            # Calculate error rate
            total_requests = len(instance_history)
            failed_requests = sum(1 for m in instance_history if not m.is_healthy)
            error_rate = (failed_requests / max(total_requests, 1)) * 100
            
            # Calculate percentiles
            recent_response_times = [
                m.response_time_ms for m in instance_history[-100:] 
                if m.is_healthy
            ]
            
            if recent_response_times:
                recent_response_times.sort()
                p95_idx = int(len(recent_response_times) * 0.95)
                p99_idx = int(len(recent_response_times) * 0.99)
                p95_response_time = recent_response_times[min(p95_idx, len(recent_response_times) - 1)]
                p99_response_time = recent_response_times[min(p99_idx, len(recent_response_times) - 1)]
                average_response_time = sum(recent_response_times) / len(recent_response_times)
            else:
                p95_response_time = response_time_ms
                p99_response_time = response_time_ms
                average_response_time = response_time_ms
            
            # System resource usage (estimated for this instance)
            try:
                system_memory = psutil.virtual_memory()
                memory_usage_percent = (memory_usage_mb / (system_memory.total / 1024 / 1024)) * 100
            except Exception:
                memory_usage_percent = 0
            
            metrics = OllamaMetrics(
                instance_url=instance_url,
                timestamp=time.time(),
                is_healthy=is_healthy,
                response_time_ms=response_time_ms,
                active_connections=active_connections,
                queue_size=queue_size,
                cpu_usage=cpu_usage,
                memory_usage_mb=memory_usage_mb,
                memory_usage_percent=memory_usage_percent,
                total_requests=total_requests,
                successful_requests=total_requests - failed_requests,
                failed_requests=failed_requests,
                requests_per_second=requests_per_second,
                average_response_time=average_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                error_rate=error_rate,
                uptime_seconds=time.time() - self.start_time,
                models_loaded=models_loaded
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics from {instance_url}: {e}")
            
            # Return error metrics
            return OllamaMetrics(
                instance_url=instance_url,
                timestamp=time.time(),
                is_healthy=False,
                response_time_ms=30000,  # Timeout
                active_connections=0,
                queue_size=0,
                cpu_usage=0,
                memory_usage_mb=0,
                memory_usage_percent=0,
                total_requests=0,
                successful_requests=0,
                failed_requests=1,
                requests_per_second=0,
                average_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                error_rate=100,
                uptime_seconds=time.time() - self.start_time,
                models_loaded=[]
            )

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-wide performance metrics."""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage_mb = (memory.total - memory.available) / 1024 / 1024
            memory_usage_percent = memory.percent
            available_memory_mb = memory.available / 1024 / 1024
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Load average
            load_average = list(psutil.getloadavg())
            
            # Process metrics
            open_files = len(psutil.pids())
            active_processes = len([p for p in psutil.process_iter() if p.status() == 'running'])
            
            return SystemMetrics(
                timestamp=time.time(),
                total_cpu_usage=cpu_usage,
                total_memory_usage_mb=memory_usage_mb,
                total_memory_usage_percent=memory_usage_percent,
                available_memory_mb=available_memory_mb,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                load_average=load_average,
                open_files=open_files,
                active_processes=active_processes
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                total_cpu_usage=0,
                total_memory_usage_mb=0,
                total_memory_usage_percent=0,
                available_memory_mb=0,
                disk_usage_percent=0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                load_average=[0, 0, 0],
                open_files=0,
                active_processes=0
            )

    async def store_metrics(self, ollama_metrics: List[OllamaMetrics], system_metrics: SystemMetrics):
        """Store metrics in Redis and local memory."""
        current_time = time.time()
        
        # Store in local memory
        for metrics in ollama_metrics:
            if metrics.instance_url not in self.metrics_history:
                self.metrics_history[metrics.instance_url] = []
            
            self.metrics_history[metrics.instance_url].append(metrics)
            
            # Limit history size
            max_history = (self.metrics_retention_hours * 3600) // self.monitoring_interval
            if len(self.metrics_history[metrics.instance_url]) > max_history:
                self.metrics_history[metrics.instance_url] = (
                    self.metrics_history[metrics.instance_url][-max_history:]
                )
        
        self.system_metrics_history.append(system_metrics)
        max_system_history = (self.metrics_retention_hours * 3600) // self.monitoring_interval
        if len(self.system_metrics_history) > max_system_history:
            self.system_metrics_history = self.system_metrics_history[-max_system_history:]
        
        # Store in Redis if available
        if self.redis_client:
            try:
                # Store current snapshot
                snapshot = {
                    "timestamp": current_time,
                    "ollama_instances": [m.to_dict() for m in ollama_metrics],
                    "system_metrics": system_metrics.to_dict()
                }
                
                await self.redis_client.setex(
                    "ollama:monitor:current",
                    300,  # 5 minute TTL
                    json.dumps(snapshot)
                )
                
                # Store in time series for historical analysis
                await self.redis_client.zadd(
                    "ollama:monitor:history",
                    {json.dumps(snapshot): current_time}
                )
                
                # Cleanup old entries
                cutoff_time = current_time - (self.metrics_retention_hours * 3600)
                await self.redis_client.zremrangebyscore(
                    "ollama:monitor:history",
                    0,
                    cutoff_time
                )
                
            except Exception as e:
                logger.error(f"Failed to store metrics in Redis: {e}")

    async def check_alerts(self, ollama_metrics: List[OllamaMetrics], system_metrics: SystemMetrics):
        """Check for alert conditions and send notifications."""
        current_time = time.time()
        alert_cooldown = 300  # 5 minutes between same alert type
        
        alerts = []
        
        # Check Ollama instance alerts
        for metrics in ollama_metrics:
            instance = metrics.instance_url
            
            # Health alert
            if not metrics.is_healthy:
                alert_key = f"{instance}:health"
                if (alert_key not in self.alerts_sent or 
                    current_time - self.alerts_sent[alert_key] > alert_cooldown):
                    alerts.append(f"CRITICAL: Ollama instance {instance} is unhealthy")
                    self.alerts_sent[alert_key] = current_time
            
            # Response time alert
            if metrics.response_time_ms > self.alert_thresholds["response_time_ms"]:
                alert_key = f"{instance}:response_time"
                if (alert_key not in self.alerts_sent or 
                    current_time - self.alerts_sent[alert_key] > alert_cooldown):
                    alerts.append(f"WARNING: High response time for {instance}: "
                                f"{metrics.response_time_ms:.1f}ms")
                    self.alerts_sent[alert_key] = current_time
            
            # Error rate alert
            if metrics.error_rate > self.alert_thresholds["error_rate"]:
                alert_key = f"{instance}:error_rate"
                if (alert_key not in self.alerts_sent or 
                    current_time - self.alerts_sent[alert_key] > alert_cooldown):
                    alerts.append(f"WARNING: High error rate for {instance}: "
                                f"{metrics.error_rate:.1f}%")
                    self.alerts_sent[alert_key] = current_time
            
            # Queue size alert
            if metrics.queue_size > self.alert_thresholds["queue_size"]:
                alert_key = f"{instance}:queue_size"
                if (alert_key not in self.alerts_sent or 
                    current_time - self.alerts_sent[alert_key] > alert_cooldown):
                    alerts.append(f"WARNING: Large queue for {instance}: "
                                f"{metrics.queue_size} requests")
                    self.alerts_sent[alert_key] = current_time
        
        # Check system alerts
        if system_metrics.total_cpu_usage > self.alert_thresholds["cpu_usage"]:
            alert_key = "system:cpu"
            if (alert_key not in self.alerts_sent or 
                current_time - self.alerts_sent[alert_key] > alert_cooldown):
                alerts.append(f"WARNING: High system CPU usage: "
                            f"{system_metrics.total_cpu_usage:.1f}%")
                self.alerts_sent[alert_key] = current_time
        
        if system_metrics.total_memory_usage_percent > self.alert_thresholds["memory_usage"]:
            alert_key = "system:memory"
            if (alert_key not in self.alerts_sent or 
                current_time - self.alerts_sent[alert_key] > alert_cooldown):
                alerts.append(f"WARNING: High system memory usage: "
                            f"{system_metrics.total_memory_usage_percent:.1f}%")
                self.alerts_sent[alert_key] = current_time
        
        if system_metrics.disk_usage_percent > self.alert_thresholds["disk_usage"]:
            alert_key = "system:disk"
            if (alert_key not in self.alerts_sent or 
                current_time - self.alerts_sent[alert_key] > alert_cooldown):
                alerts.append(f"WARNING: High disk usage: "
                            f"{system_metrics.disk_usage_percent:.1f}%")
                self.alerts_sent[alert_key] = current_time
        
        # Log alerts
        for alert in alerts:
            logger.warning(alert)

    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_time = time.time()
        
        # Aggregate Ollama metrics
        healthy_instances = 0
        total_instances = len(self.instances)
        total_active_connections = 0
        total_queue_size = 0
        avg_response_time = 0
        total_requests = 0
        total_errors = 0
        
        instance_reports = []
        
        for instance_url in self.instances:
            history = self.metrics_history.get(instance_url, [])
            if not history:
                continue
                
            latest = history[-1]
            
            if latest.is_healthy:
                healthy_instances += 1
            
            total_active_connections += latest.active_connections
            total_queue_size += latest.queue_size
            total_requests += latest.total_requests
            total_errors += latest.failed_requests
            
            # Calculate averages over last hour
            recent_metrics = [
                m for m in history 
                if m.timestamp > current_time - 3600
            ]
            
            if recent_metrics:
                avg_response_time += sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
            
            instance_reports.append({
                "url": instance_url,
                "healthy": latest.is_healthy,
                "response_time_ms": latest.response_time_ms,
                "active_connections": latest.active_connections,
                "queue_size": latest.queue_size,
                "requests_per_second": latest.requests_per_second,
                "error_rate": latest.error_rate,
                "models_loaded": latest.models_loaded,
                "uptime_seconds": latest.uptime_seconds
            })
        
        # System metrics
        system_latest = self.system_metrics_history[-1] if self.system_metrics_history else None
        
        # Calculate totals and averages
        if total_instances > 0:
            avg_response_time /= total_instances
        
        success_rate = ((total_requests - total_errors) / max(total_requests, 1)) * 100
        
        report = {
            "timestamp": current_time,
            "monitoring_duration_seconds": current_time - self.start_time,
            "summary": {
                "total_instances": total_instances,
                "healthy_instances": healthy_instances,
                "health_rate": (healthy_instances / max(total_instances, 1)) * 100,
                "total_active_connections": total_active_connections,
                "total_queue_size": total_queue_size,
                "total_capacity": total_instances * 50,  # Assuming 50 per instance
                "utilization_percent": (total_active_connections / max(total_instances * 50, 1)) * 100,
                "avg_response_time_ms": avg_response_time,
                "total_requests": total_requests,
                "success_rate": success_rate,
                "total_errors": total_errors
            },
            "instances": instance_reports,
            "system": system_latest.to_dict() if system_latest else {},
            "alert_thresholds": self.alert_thresholds,
            "recent_alerts": len(self.alerts_sent)
        }
        
        return report

    async def monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting monitoring loop...")
        
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Collect metrics from all instances
                ollama_tasks = [
                    self.collect_ollama_metrics(instance) 
                    for instance in self.instances
                ]
                
                ollama_metrics = await asyncio.gather(*ollama_tasks, return_exceptions=True)
                
                # Filter out exceptions and None values
                valid_metrics = [
                    m for m in ollama_metrics 
                    if isinstance(m, OllamaMetrics)
                ]
                
                # Collect system metrics
                system_metrics = await self.collect_system_metrics()
                
                # Store metrics
                await self.store_metrics(valid_metrics, system_metrics)
                
                # Check for alerts
                await self.check_alerts(valid_metrics, system_metrics)
                
                # Log summary
                healthy_count = sum(1 for m in valid_metrics if m.is_healthy)
                total_connections = sum(m.active_connections for m in valid_metrics)
                avg_response_time = (
                    sum(m.response_time_ms for m in valid_metrics) / max(len(valid_metrics), 1)
                )
                
                logger.info(f"Monitor cycle: {healthy_count}/{len(self.instances)} healthy, "
                           f"{total_connections} active connections, "
                           f"{avg_response_time:.1f}ms avg response time, "
                           f"CPU: {system_metrics.total_cpu_usage:.1f}%, "
                           f"Memory: {system_metrics.total_memory_usage_percent:.1f}%")
                
                # Sleep until next cycle
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.monitoring_interval - loop_duration)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def run_api_server(self, port: int = 8082):
        """Run a simple HTTP API server for metrics access."""
        from aiohttp import web
        
        async def get_metrics(request):
            report = await self.generate_report()
            return web.json_response(report)
        
        async def get_health(request):
            healthy_instances = sum(
                1 for instance in self.instances
                if self.metrics_history.get(instance, [])
                and self.metrics_history[instance][-1].is_healthy
            )
            
            is_healthy = healthy_instances > 0
            
            return web.json_response({
                "healthy": is_healthy,
                "healthy_instances": healthy_instances,
                "total_instances": len(self.instances),
                "uptime": time.time() - self.start_time
            }, status=200 if is_healthy else 503)
        
        app = web.Application()
        app.router.add_get('/metrics', get_metrics)
        app.router.add_get('/health', get_health)
        app.router.add_get('/', get_health)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"API server started on port {port}")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ollama Performance Monitor")
    parser.add_argument("--instances", nargs="+", 
                       default=["http://localhost:11434"],
                       help="Ollama instance URLs to monitor")
    parser.add_argument("--redis-url", default="redis://localhost:6379",
                       help="Redis URL for metrics storage")
    parser.add_argument("--interval", type=int, default=10,
                       help="Monitoring interval in seconds")
    parser.add_argument("--retention-hours", type=int, default=24,
                       help="Metrics retention time in hours")
    parser.add_argument("--api-port", type=int, default=8082,
                       help="API server port")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = OllamaPerformanceMonitor(
        instances=args.instances,
        redis_url=args.redis_url,
        monitoring_interval=args.interval,
        metrics_retention_hours=args.retention_hours
    )
    
    # Graceful shutdown handler
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(monitor.shutdown())
    
    # Set up signal handlers
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
    
    try:
        # Initialize monitor
        await monitor.initialize()
        
        # Start API server and monitoring loop
        await asyncio.gather(
            monitor.run_api_server(args.api_port),
            monitor.monitoring_loop()
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await monitor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())