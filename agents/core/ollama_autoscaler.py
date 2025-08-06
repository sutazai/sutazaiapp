#!/usr/bin/env python3
"""
Ollama Auto-Scaler for High-Concurrency Load Management
Automatically scales Ollama instances based on demand patterns
"""

import asyncio
import json
import time
import logging
import docker
import redis.asyncio as redis
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import os
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/ollama_autoscaler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: float
    total_instances: int
    healthy_instances: int
    active_connections: int
    total_capacity: int
    utilization_percent: float
    queue_size: int
    avg_response_time: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    requests_per_second: float
    load_trend: str  # 'increasing', 'decreasing', 'stable'

@dataclass
class ScalingEvent:
    """Record of scaling actions."""
    timestamp: float
    action: str  # 'scale_up', 'scale_down', 'no_action'
    from_instances: int
    to_instances: int
    reason: str
    metrics: ScalingMetrics

class OllamaAutoscaler:
    """
    Intelligent auto-scaler for Ollama instances.
    Monitors load patterns and automatically adjusts capacity.
    """
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 min_instances: int = 1,
                 max_instances: int = 4,
                 target_utilization: float = 70.0,
                 scale_up_threshold: float = 80.0,
                 scale_down_threshold: float = 40.0,
                 scale_up_cooldown: int = 300,    # 5 minutes
                 scale_down_cooldown: int = 600,  # 10 minutes
                 check_interval: int = 30,        # 30 seconds
                 stability_window: int = 180):    # 3 minutes
        
        self.redis_url = redis_url
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        self.check_interval = check_interval
        self.stability_window = stability_window
        
        # Docker client for container management
        self.docker_client = docker.from_env()
        
        # Redis client for metrics and coordination
        self.redis_client: Optional[redis.Redis] = None
        
        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        
        # State tracking
        self.is_running = False
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.current_instances = 0
        self.scaling_history: List[ScalingEvent] = []
        self.metrics_history: List[ScalingMetrics] = []
        
        # Instance configuration templates
        self.instance_configs = {
            "primary": {
                "image": "ollama/ollama:latest",
                "environment": {
                    "OLLAMA_NUM_PARALLEL": "50",
                    "OLLAMA_MAX_LOADED_MODELS": "3",
                    "OLLAMA_KEEP_ALIVE": "10m",
                    "OLLAMA_HOST": "0.0.0.0:10104",
                    "OLLAMA_MAX_MEMORY": "16384"
                },
                "ports": {"10104/tcp": 10104},
                "mem_limit": "16g",
                "cpu_count": 8,
                "priority": 1
            },
            "secondary": {
                "image": "ollama/ollama:latest",
                "environment": {
                    "OLLAMA_NUM_PARALLEL": "30",
                    "OLLAMA_MAX_LOADED_MODELS": "2",
                    "OLLAMA_KEEP_ALIVE": "8m",
                    "OLLAMA_HOST": "0.0.0.0:10104",
                    "OLLAMA_MAX_MEMORY": "10240"
                },
                "ports": {"10104/tcp": 11435},
                "mem_limit": "10g",
                "cpu_count": 6,
                "priority": 2
            },
            "tertiary": {
                "image": "ollama/ollama:latest",
                "environment": {
                    "OLLAMA_NUM_PARALLEL": "20",
                    "OLLAMA_MAX_LOADED_MODELS": "1",
                    "OLLAMA_KEEP_ALIVE": "5m",
                    "OLLAMA_HOST": "0.0.0.0:10104",
                    "OLLAMA_MAX_MEMORY": "6144"
                },
                "ports": {"10104/tcp": 11436},
                "mem_limit": "6g",
                "cpu_count": 4,
                "priority": 3
            }
        }
        
        logger.info(f"Autoscaler initialized: {min_instances}-{max_instances} instances, "
                   f"target utilization: {target_utilization}%")

    async def initialize(self):
        """Initialize the autoscaler components."""
        try:
            # Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis for coordination")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
        
        # HTTP session
        connector = aiohttp.TCPConnector(
            limit=20,
            ttl_dns_cache=300
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Get current instance count
        self.current_instances = await self.get_current_instance_count()
        
        self.is_running = True
        logger.info(f"Autoscaler initialized with {self.current_instances} current instances")

    async def shutdown(self):
        """Gracefully shutdown the autoscaler."""
        logger.info("Shutting down autoscaler...")
        self.is_running = False
        
        if self.session:
            await self.session.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Autoscaler shutdown complete")

    async def get_current_instance_count(self) -> int:
        """Get the current number of running Ollama instances."""
        try:
            containers = self.docker_client.containers.list(
                filters={"label": "app=ollama"}
            )
            return len(containers)
        except Exception as e:
            logger.error(f"Failed to get current instance count: {e}")
            return 1  # Default to 1

    async def get_scaling_metrics(self) -> Optional[ScalingMetrics]:
        """Gather metrics for scaling decisions."""
        try:
            # Get metrics from Redis (populated by performance monitor)
            metrics_data = await self.redis_client.get("ollama:monitor:current")
            if not metrics_data:
                logger.warning("No metrics data available from monitor")
                return None
            
            data = json.loads(metrics_data)
            
            # Extract key metrics
            instances = data.get("ollama_instances", [])
            system_metrics = data.get("system_metrics", {})
            
            total_instances = len(instances)
            healthy_instances = sum(1 for i in instances if i.get("is_healthy", False))
            active_connections = sum(i.get("active_connections", 0) for i in instances)
            queue_size = sum(i.get("queue_size", 0) for i in instances)
            
            # Calculate capacity (assuming 50 connections per primary, scaled down for others)
            total_capacity = sum(
                50 if i == 0 else (30 if i == 1 else 20)
                for i in range(total_instances)
            )
            
            utilization_percent = (active_connections / max(total_capacity, 1)) * 100
            
            # Response time and error rate
            response_times = [i.get("response_time_ms", 0) for i in instances if i.get("is_healthy")]
            avg_response_time = sum(response_times) / max(len(response_times), 1)
            
            error_rates = [i.get("error_rate", 0) for i in instances]
            error_rate = sum(error_rates) / max(len(error_rates), 1)
            
            # System metrics
            cpu_usage = system_metrics.get("total_cpu_usage", 0)
            memory_usage = system_metrics.get("total_memory_usage_percent", 0)
            
            # Calculate request rate
            requests_per_second = sum(i.get("requests_per_second", 0) for i in instances)
            
            # Determine load trend
            load_trend = await self.calculate_load_trend(utilization_percent)
            
            metrics = ScalingMetrics(
                timestamp=time.time(),
                total_instances=total_instances,
                healthy_instances=healthy_instances,
                active_connections=active_connections,
                total_capacity=total_capacity,
                utilization_percent=utilization_percent,
                queue_size=queue_size,
                avg_response_time=avg_response_time,
                error_rate=error_rate,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                requests_per_second=requests_per_second,
                load_trend=load_trend
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get scaling metrics: {e}")
            return None

    async def calculate_load_trend(self, current_utilization: float) -> str:
        """Calculate load trend based on historical data."""
        if len(self.metrics_history) < 3:
            return "stable"
        
        # Get recent utilization values
        recent_utilizations = [
            m.utilization_percent for m in self.metrics_history[-5:]
        ]
        recent_utilizations.append(current_utilization)
        
        # Simple trend calculation
        if len(recent_utilizations) >= 3:
            early_avg = sum(recent_utilizations[:2]) / 2
            late_avg = sum(recent_utilizations[-2:]) / 2
            
            if late_avg > early_avg + 10:
                return "increasing"
            elif late_avg < early_avg - 10:
                return "decreasing"
        
        return "stable"

    async def should_scale_up(self, metrics: ScalingMetrics) -> Tuple[bool, str]:
        """Determine if we should scale up."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scale_up < self.scale_up_cooldown:
            return False, f"Scale-up cooldown active ({self.scale_up_cooldown}s)"
        
        # Check if at maximum instances
        if metrics.total_instances >= self.max_instances:
            return False, f"Already at maximum instances ({self.max_instances})"
        
        # Check utilization threshold
        if metrics.utilization_percent > self.scale_up_threshold:
            return True, f"High utilization: {metrics.utilization_percent:.1f}% > {self.scale_up_threshold}%"
        
        # Check queue size
        if metrics.queue_size > 50:
            return True, f"Large queue size: {metrics.queue_size} requests"
        
        # Check response time
        if metrics.avg_response_time > 3000:  # 3 seconds
            return True, f"High response time: {metrics.avg_response_time:.1f}ms"
        
        # Check load trend
        if (metrics.load_trend == "increasing" and 
            metrics.utilization_percent > self.target_utilization):
            return True, f"Increasing load trend at {metrics.utilization_percent:.1f}% utilization"
        
        # Check system resources
        if metrics.cpu_usage > 85 or metrics.memory_usage > 90:
            return True, f"High system resource usage: CPU {metrics.cpu_usage:.1f}%, Memory {metrics.memory_usage:.1f}%"
        
        return False, "No scale-up conditions met"

    async def should_scale_down(self, metrics: ScalingMetrics) -> Tuple[bool, str]:
        """Determine if we should scale down."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scale_down < self.scale_down_cooldown:
            return False, f"Scale-down cooldown active ({self.scale_down_cooldown}s)"
        
        # Check if at minimum instances
        if metrics.total_instances <= self.min_instances:
            return False, f"Already at minimum instances ({self.min_instances})"
        
        # Check utilization threshold (must be stable low utilization)
        if metrics.utilization_percent < self.scale_down_threshold:
            # Check if utilization has been consistently low
            recent_low_utilization = all(
                m.utilization_percent < self.scale_down_threshold
                for m in self.metrics_history[-3:]  # Last 3 measurements
            )
            
            if recent_low_utilization:
                return True, f"Sustained low utilization: {metrics.utilization_percent:.1f}% < {self.scale_down_threshold}%"
        
        # Check if load is decreasing and we have excess capacity
        if (metrics.load_trend == "decreasing" and 
            metrics.utilization_percent < self.target_utilization and
            metrics.queue_size == 0):
            return True, f"Decreasing load trend with low utilization: {metrics.utilization_percent:.1f}%"
        
        return False, "No scale-down conditions met"

    async def scale_up(self, metrics: ScalingMetrics) -> bool:
        """Scale up by adding an Ollama instance."""
        try:
            target_instances = min(metrics.total_instances + 1, self.max_instances)
            instance_type = self.get_instance_type_for_scaling(target_instances)
            
            logger.info(f"Scaling up: {metrics.total_instances} -> {target_instances} instances")
            
            # Get configuration for new instance
            config = self.instance_configs[instance_type]
            
            # Create new container
            container_name = f"ollama-{instance_type}-auto"
            
            try:
                # Remove existing container if it exists
                try:
                    existing = self.docker_client.containers.get(container_name)
                    existing.remove(force=True)
                    logger.info(f"Removed existing container: {container_name}")
                except docker.errors.NotFound:
                    pass
                
                # Create and start new container
                container = self.docker_client.containers.run(
                    image=config["image"],
                    name=container_name,
                    environment=config["environment"],
                    ports=config["ports"],
                    mem_limit=config["mem_limit"],
                    cpu_count=config["cpu_count"],
                    detach=True,
                    restart_policy={"Name": "unless-stopped"},
                    labels={"app": "ollama", "autoscaled": "true", "priority": str(config["priority"])},
                    volumes={
                        f"ollama-models-{instance_type}": {"bind": "/root/.ollama", "mode": "rw"}
                    }
                )
                
                logger.info(f"Created new instance: {container_name}")
                
                # Wait for health check
                await self.wait_for_instance_health(container_name, timeout=60)
                
                # Update tracking
                self.current_instances = target_instances
                self.last_scale_up = time.time()
                
                # Record scaling event
                event = ScalingEvent(
                    timestamp=time.time(),
                    action="scale_up",
                    from_instances=metrics.total_instances,
                    to_instances=target_instances,
                    reason=f"Added {instance_type} instance",
                    metrics=metrics
                )
                self.scaling_history.append(event)
                
                # Store event in Redis
                await self.store_scaling_event(event)
                
                logger.info(f"Successfully scaled up to {target_instances} instances")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create container {container_name}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Scale-up failed: {e}")
            return False

    async def scale_down(self, metrics: ScalingMetrics) -> bool:
        """Scale down by removing an Ollama instance."""
        try:
            target_instances = max(metrics.total_instances - 1, self.min_instances)
            
            logger.info(f"Scaling down: {metrics.total_instances} -> {target_instances} instances")
            
            # Find lowest priority instance to remove
            containers = self.docker_client.containers.list(
                filters={"label": "app=ollama"}
            )
            
            # Sort by priority (highest priority number = lowest priority)
            autoscaled_containers = [
                c for c in containers 
                if c.labels.get("autoscaled") == "true"
            ]
            
            if not autoscaled_containers:
                logger.warning("No autoscaled containers to remove")
                return False
            
            # Sort by priority (remove highest priority number first)
            autoscaled_containers.sort(
                key=lambda c: int(c.labels.get("priority", "0")),
                reverse=True
            )
            
            container_to_remove = autoscaled_containers[0]
            
            logger.info(f"Removing instance: {container_to_remove.name}")
            
            # Gracefully stop and remove container
            container_to_remove.stop(timeout=30)
            container_to_remove.remove()
            
            # Update tracking
            self.current_instances = target_instances
            self.last_scale_down = time.time()
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                action="scale_down",
                from_instances=metrics.total_instances,
                to_instances=target_instances,
                reason=f"Removed {container_to_remove.name}",
                metrics=metrics
            )
            self.scaling_history.append(event)
            
            # Store event in Redis
            await self.store_scaling_event(event)
            
            logger.info(f"Successfully scaled down to {target_instances} instances")
            return True
            
        except Exception as e:
            logger.error(f"Scale-down failed: {e}")
            return False

    def get_instance_type_for_scaling(self, target_count: int) -> str:
        """Get the appropriate instance type for scaling."""
        if target_count == 1:
            return "primary"
        elif target_count == 2:
            return "secondary"
        else:
            return "tertiary"

    async def wait_for_instance_health(self, container_name: str, timeout: int = 60):
        """Wait for an instance to become healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                container = self.docker_client.containers.get(container_name)
                
                if container.status == "running":
                    # Try to connect to the API
                    port_mapping = container.ports.get("10104/tcp")
                    if port_mapping:
                        port = port_mapping[0]["HostPort"]
                        
                        async with self.session.get(f"http://localhost:{port}/api/tags",
                                                   timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                logger.info(f"Instance {container_name} is healthy")
                                return
                
            except Exception:
                pass  # Continue waiting
            
            await asyncio.sleep(2)
        
        raise Exception(f"Instance {container_name} failed to become healthy within {timeout}s")

    async def store_scaling_event(self, event: ScalingEvent):
        """Store scaling event in Redis."""
        try:
            if self.redis_client:
                event_data = {
                    "timestamp": event.timestamp,
                    "action": event.action,
                    "from_instances": event.from_instances,
                    "to_instances": event.to_instances,
                    "reason": event.reason,
                    "metrics": {
                        "utilization_percent": event.metrics.utilization_percent,
                        "queue_size": event.metrics.queue_size,
                        "avg_response_time": event.metrics.avg_response_time,
                        "load_trend": event.metrics.load_trend
                    }
                }
                
                await self.redis_client.zadd(
                    "ollama:autoscaler:events",
                    {json.dumps(event_data): event.timestamp}
                )
                
                # Keep only last 100 events
                await self.redis_client.zremrangebyrank("ollama:autoscaler:events", 0, -101)
                
        except Exception as e:
            logger.error(f"Failed to store scaling event: {e}")

    async def autoscaling_loop(self):
        """Main autoscaling loop."""
        logger.info("Starting autoscaling loop...")
        
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Get current metrics
                metrics = await self.get_scaling_metrics()
                if not metrics:
                    logger.warning("No metrics available, skipping scaling decision")
                    await asyncio.sleep(self.check_interval)
                    continue
                
                # Store metrics for trend analysis
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 20:  # Keep last 20 measurements
                    self.metrics_history = self.metrics_history[-20:]
                
                # Check scaling conditions
                should_up, up_reason = await self.should_scale_up(metrics)
                should_down, down_reason = await self.should_scale_down(metrics)
                
                action_taken = "no_action"
                
                if should_up and not should_down:
                    success = await self.scale_up(metrics)
                    if success:
                        action_taken = "scale_up"
                        logger.info(f"Scaled up: {up_reason}")
                    else:
                        logger.error(f"Scale-up failed: {up_reason}")
                
                elif should_down and not should_up:
                    success = await self.scale_down(metrics)
                    if success:
                        action_taken = "scale_down"
                        logger.info(f"Scaled down: {down_reason}")
                    else:
                        logger.error(f"Scale-down failed: {down_reason}")
                
                elif should_up and should_down:
                    logger.warning("Conflicting scaling signals - taking no action")
                
                # Log current state
                logger.info(f"Autoscaler status: {metrics.total_instances} instances, "
                           f"{metrics.utilization_percent:.1f}% utilization, "
                           f"{metrics.queue_size} queued, "
                           f"trend: {metrics.load_trend}, "
                           f"action: {action_taken}")
                
                # Sleep until next check
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.check_interval - loop_duration)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in autoscaling loop: {e}")
                await asyncio.sleep(self.check_interval)

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current autoscaling status."""
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "target_utilization": self.target_utilization,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "last_scale_up": self.last_scale_up,
            "last_scale_down": self.last_scale_down,
            "scaling_events": len(self.scaling_history),
            "recent_events": [
                {
                    "timestamp": event.timestamp,
                    "action": event.action,
                    "from_instances": event.from_instances,
                    "to_instances": event.to_instances,
                    "reason": event.reason
                }
                for event in self.scaling_history[-5:]  # Last 5 events
            ]
        }

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama Autoscaler")
    parser.add_argument("--redis-url", default="redis://localhost:6379",
                       help="Redis URL for coordination")
    parser.add_argument("--min-instances", type=int, default=1,
                       help="Minimum number of instances")
    parser.add_argument("--max-instances", type=int, default=4,
                       help="Maximum number of instances")
    parser.add_argument("--target-utilization", type=float, default=70.0,
                       help="Target utilization percentage")
    
    args = parser.parse_args()
    
    # Create autoscaler
    autoscaler = OllamaAutoscaler(
        redis_url=args.redis_url,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        target_utilization=args.target_utilization
    )
    
    # Graceful shutdown handler
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(autoscaler.shutdown())
    
    # Set up signal handlers
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
    
    try:
        await autoscaler.initialize()
        await autoscaler.autoscaling_loop()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await autoscaler.shutdown()

if __name__ == "__main__":
    asyncio.run(main())