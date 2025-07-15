"""
Auto-scaling and Load Balancing for SutazAI
Dynamic resource scaling based on performance metrics
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class ScalingAction(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"

@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    cpu_scale_up_threshold: float = 75.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 40.0
    response_time_threshold: float = 1000.0  # ms
    min_instances: int = 1
    max_instances: int = 10
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    evaluation_period: float = 60.0  # 1 minute

class AutoScaler:
    """Automatic scaling system"""
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
        self.current_instances = self.config.min_instances
        self.last_scale_action = 0.0
        self.last_action_type = None
        
        # Metrics tracking
        self.metrics_history = []
        self.scaling_history = []
        
        # Scaling handlers
        self.scale_up_handlers = []
        self.scale_down_handlers = []
        
        # State
        self.scaling_active = False
        self.scaling_task = None
    
    async def initialize(self):
        """Initialize auto-scaler"""
        logger.info("🔄 Initializing Auto-Scaler")
        
        self.scaling_active = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        
        logger.info(f"✅ Auto-Scaler initialized (instances: {self.current_instances})")
    
    async def _scaling_loop(self):
        """Main scaling evaluation loop"""
        while self.scaling_active:
            try:
                await asyncio.sleep(self.config.evaluation_period)
                
                # Collect current metrics
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics
                self.metrics_history = self.metrics_history[-100:]
                
                # Evaluate scaling decision
                action = await self._evaluate_scaling_decision(metrics)
                
                if action != ScalingAction.MAINTAIN:
                    await self._execute_scaling_action(action, metrics)
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(self.config.evaluation_period)
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        try:
            process = psutil.Process()
            
            # Get performance metrics
            cpu_percent = process.cpu_percent(interval=1)
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get response time from performance monitor if available
            response_time = 0.0
            try:
                from backend.monitoring.performance_monitor import performance_monitor
                current_metrics = performance_monitor.get_current_metrics()
                response_time = current_metrics.get("avg_response_time", 0.0)
            except ImportError:
                pass
            
            metrics = {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_mb": memory_info.rss / 1024 / 1024,
                "response_time_ms": response_time,
                "current_instances": self.current_instances
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {
                "timestamp": time.time(),
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "memory_mb": 0.0,
                "response_time_ms": 0.0,
                "current_instances": self.current_instances
            }
    
    async def _evaluate_scaling_decision(self, metrics: Dict[str, Any]) -> ScalingAction:
        """Evaluate whether to scale up, down, or maintain"""
        try:
            # Check cooldown periods
            current_time = time.time()
            time_since_last_action = current_time - self.last_scale_action
            
            # Scale up conditions
            scale_up_needed = (
                metrics["cpu_percent"] > self.config.cpu_scale_up_threshold or
                metrics["memory_percent"] > self.config.memory_scale_up_threshold or
                metrics["response_time_ms"] > self.config.response_time_threshold
            )
            
            # Scale down conditions
            scale_down_possible = (
                metrics["cpu_percent"] < self.config.cpu_scale_down_threshold and
                metrics["memory_percent"] < self.config.memory_scale_down_threshold and
                metrics["response_time_ms"] < self.config.response_time_threshold / 2
            )
            
            # Check if we can scale up
            if (scale_up_needed and 
                self.current_instances < self.config.max_instances and
                (self.last_action_type != ScalingAction.SCALE_UP or 
                 time_since_last_action > self.config.scale_up_cooldown)):
                return ScalingAction.SCALE_UP
            
            # Check if we can scale down
            if (scale_down_possible and 
                self.current_instances > self.config.min_instances and
                (self.last_action_type != ScalingAction.SCALE_DOWN or 
                 time_since_last_action > self.config.scale_down_cooldown)):
                return ScalingAction.SCALE_DOWN
            
            return ScalingAction.MAINTAIN
            
        except Exception as e:
            logger.error(f"Scaling evaluation error: {e}")
            return ScalingAction.MAINTAIN
    
    async def _execute_scaling_action(self, action: ScalingAction, metrics: Dict[str, Any]):
        """Execute scaling action"""
        try:
            logger.info(f"Executing scaling action: {action}")
            
            old_instances = self.current_instances
            
            if action == ScalingAction.SCALE_UP:
                new_instances = min(self.current_instances + 1, self.config.max_instances)
                await self._scale_up(new_instances - self.current_instances)
                
            elif action == ScalingAction.SCALE_DOWN:
                new_instances = max(self.current_instances - 1, self.config.min_instances)
                await self._scale_down(self.current_instances - new_instances)
            
            self.current_instances = new_instances if action != ScalingAction.MAINTAIN else self.current_instances
            self.last_scale_action = time.time()
            self.last_action_type = action
            
            # Record scaling event
            scaling_event = {
                "timestamp": time.time(),
                "action": action,
                "old_instances": old_instances,
                "new_instances": self.current_instances,
                "trigger_metrics": metrics.copy(),
                "reason": self._get_scaling_reason(action, metrics)
            }
            
            self.scaling_history.append(scaling_event)
            self.scaling_history = self.scaling_history[-50:]  # Keep last 50 events
            
            logger.info(f"Scaling completed: {old_instances} -> {self.current_instances} instances")
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
    
    def _get_scaling_reason(self, action: ScalingAction, metrics: Dict[str, Any]) -> str:
        """Get human-readable scaling reason"""
        if action == ScalingAction.SCALE_UP:
            reasons = []
            if metrics["cpu_percent"] > self.config.cpu_scale_up_threshold:
                reasons.append(f"High CPU: {metrics['cpu_percent']:.1f}%")
            if metrics["memory_percent"] > self.config.memory_scale_up_threshold:
                reasons.append(f"High Memory: {metrics['memory_percent']:.1f}%")
            if metrics["response_time_ms"] > self.config.response_time_threshold:
                reasons.append(f"High Response Time: {metrics['response_time_ms']:.1f}ms")
            return ", ".join(reasons)
        
        elif action == ScalingAction.SCALE_DOWN:
            return f"Low resource utilization - CPU: {metrics['cpu_percent']:.1f}%, Memory: {metrics['memory_percent']:.1f}%"
        
        return "Maintaining current scale"
    
    async def _scale_up(self, instances_to_add: int):
        """Scale up by adding instances"""
        for handler in self.scale_up_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(instances_to_add)
                else:
                    handler(instances_to_add)
            except Exception as e:
                logger.error(f"Scale up handler failed: {e}")
    
    async def _scale_down(self, instances_to_remove: int):
        """Scale down by removing instances"""
        for handler in self.scale_down_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(instances_to_remove)
                else:
                    handler(instances_to_remove)
            except Exception as e:
                logger.error(f"Scale down handler failed: {e}")
    
    def register_scale_up_handler(self, handler: Callable):
        """Register handler for scale up events"""
        self.scale_up_handlers.append(handler)
    
    def register_scale_down_handler(self, handler: Callable):
        """Register handler for scale down events"""
        self.scale_down_handlers.append(handler)
    
    async def manual_scale(self, target_instances: int) -> bool:
        """Manually scale to target number of instances"""
        try:
            if target_instances < self.config.min_instances:
                target_instances = self.config.min_instances
            elif target_instances > self.config.max_instances:
                target_instances = self.config.max_instances
            
            if target_instances == self.current_instances:
                return True
            
            old_instances = self.current_instances
            
            if target_instances > self.current_instances:
                await self._scale_up(target_instances - self.current_instances)
            else:
                await self._scale_down(self.current_instances - target_instances)
            
            self.current_instances = target_instances
            self.last_scale_action = time.time()
            self.last_action_type = ScalingAction.SCALE_UP if target_instances > old_instances else ScalingAction.SCALE_DOWN
            
            # Record manual scaling event
            scaling_event = {
                "timestamp": time.time(),
                "action": "manual_scale",
                "old_instances": old_instances,
                "new_instances": self.current_instances,
                "trigger_metrics": {},
                "reason": "Manual scaling request"
            }
            
            self.scaling_history.append(scaling_event)
            
            logger.info(f"Manual scaling completed: {old_instances} -> {self.current_instances} instances")
            return True
            
        except Exception as e:
            logger.error(f"Manual scaling failed: {e}")
            return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        current_time = time.time()
        
        # Calculate average metrics from recent history
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        
        avg_cpu = sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        avg_memory = sum(m["memory_percent"] for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        avg_response_time = sum(m["response_time_ms"] for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        
        return {
            "current_instances": self.current_instances,
            "target_range": {
                "min": self.config.min_instances,
                "max": self.config.max_instances
            },
            "last_scaling_action": {
                "action": self.last_action_type.value if self.last_action_type else "none",
                "timestamp": self.last_scale_action,
                "time_ago": current_time - self.last_scale_action
            },
            "current_metrics": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
                "avg_response_time_ms": avg_response_time
            },
            "thresholds": {
                "cpu_scale_up": self.config.cpu_scale_up_threshold,
                "cpu_scale_down": self.config.cpu_scale_down_threshold,
                "memory_scale_up": self.config.memory_scale_up_threshold,
                "memory_scale_down": self.config.memory_scale_down_threshold,
                "response_time": self.config.response_time_threshold
            },
            "scaling_active": self.scaling_active,
            "recent_events": self.scaling_history[-5:]
        }
    
    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get scaling history"""
        return self.scaling_history.copy()
    
    async def shutdown(self):
        """Shutdown auto-scaler"""
        logger.info("🛑 Shutting down Auto-Scaler")
        
        self.scaling_active = False
        
        if self.scaling_task:
            self.scaling_task.cancel()
        
        logger.info("✅ Auto-Scaler shutdown complete")

# Load balancer for distributing requests
class LoadBalancer:
    """Simple load balancer for request distribution"""
    
    def __init__(self):
        self.backends = []
        self.current_backend = 0
        self.request_counts = {}
        self.backend_health = {}
    
    def add_backend(self, backend_id: str, weight: int = 1):
        """Add backend to load balancer"""
        self.backends.append({"id": backend_id, "weight": weight})
        self.request_counts[backend_id] = 0
        self.backend_health[backend_id] = True
        logger.info(f"Added backend: {backend_id} (weight: {weight})")
    
    def remove_backend(self, backend_id: str):
        """Remove backend from load balancer"""
        self.backends = [b for b in self.backends if b["id"] != backend_id]
        self.request_counts.pop(backend_id, None)
        self.backend_health.pop(backend_id, None)
        logger.info(f"Removed backend: {backend_id}")
    
    def get_next_backend(self) -> Optional[str]:
        """Get next backend using round-robin"""
        healthy_backends = [b for b in self.backends if self.backend_health.get(b["id"], True)]
        
        if not healthy_backends:
            return None
        
        # Simple round-robin
        backend = healthy_backends[self.current_backend % len(healthy_backends)]
        self.current_backend += 1
        
        # Update request count
        self.request_counts[backend["id"]] += 1
        
        return backend["id"]
    
    def mark_backend_unhealthy(self, backend_id: str):
        """Mark backend as unhealthy"""
        self.backend_health[backend_id] = False
        logger.warning(f"Backend marked unhealthy: {backend_id}")
    
    def mark_backend_healthy(self, backend_id: str):
        """Mark backend as healthy"""
        self.backend_health[backend_id] = True
        logger.info(f"Backend marked healthy: {backend_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status"""
        return {
            "total_backends": len(self.backends),
            "healthy_backends": len([b for b in self.backends if self.backend_health.get(b["id"], True)]),
            "backends": [
                {
                    "id": b["id"],
                    "weight": b["weight"],
                    "healthy": self.backend_health.get(b["id"], True),
                    "requests": self.request_counts.get(b["id"], 0)
                }
                for b in self.backends
            ]
        }

# Global instances
auto_scaler = AutoScaler()
load_balancer = LoadBalancer()
