"""
Edge Failover System - Advanced failover mechanisms and health checking for edge inference
"""

import asyncio
import time
import threading
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import aiohttp
import psutil
import weakref

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class FailoverStrategy(Enum):
    """Failover strategies"""
    IMMEDIATE = "immediate"        # Immediate failover on failure
    GRACEFUL = "graceful"         # Graceful degradation
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker pattern
    ADAPTIVE = "adaptive"         # Adaptive based on conditions
    MANUAL = "manual"            # Manual failover only

class RecoveryAction(Enum):
    """Recovery actions"""
    RESTART_SERVICE = "restart_service"
    RELOAD_MODEL = "reload_model"
    RESET_CONNECTION = "reset_connection"
    SCALE_UP = "scale_up"
    EVACUATE_TRAFFIC = "evacuate_traffic"
    ALERT_OPERATOR = "alert_operator"

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_func: Callable[[], bool]
    interval_seconds: float
    timeout_seconds: float
    failure_threshold: int = 3
    success_threshold: int = 2
    enabled: bool = True
    critical: bool = False

@dataclass
class NodeHealth:
    """Node health information"""
    node_id: str
    status: HealthStatus
    last_check: datetime
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    response_time_ms: float = 0.0
    checks: Dict[str, bool] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    recovery_actions: List[RecoveryAction] = field(default_factory=list)

@dataclass
class FailoverEvent:
    """Failover event record"""
    event_id: str
    event_type: str
    source_node: str
    target_node: Optional[str]
    timestamp: datetime
    reason: str
    success: bool
    recovery_time_sec: Optional[float] = None
    affected_requests: int = 0

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_seconds: float = 60.0,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == "open":
                # Check if timeout has passed
                if (time.time() - self.last_failure_time) > self.timeout_seconds:
                    self.state = "half-open"
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == "half-open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.failure_count = 0
        elif self.state == "closed":
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.node_health: Dict[str, NodeHealth] = {}
        self.check_results: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._running = False
        self._check_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    def register_health_check(self, node_id: str, health_check: HealthCheck) -> None:
        """Register a health check for a node"""
        check_key = f"{node_id}_{health_check.name}"
        self.health_checks[check_key] = health_check
        
        if node_id not in self.node_health:
            self.node_health[node_id] = NodeHealth(
                node_id=node_id,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now()
            )
        
        logger.info(f"Registered health check {health_check.name} for node {node_id}")
    
    def unregister_health_check(self, node_id: str, check_name: str) -> None:
        """Unregister a health check"""
        check_key = f"{node_id}_{check_name}"
        if check_key in self.health_checks:
            del self.health_checks[check_key]
            
            # Cancel task if running
            if check_key in self._check_tasks:
                self._check_tasks[check_key].cancel()
                del self._check_tasks[check_key]
    
    async def start_health_checking(self) -> None:
        """Start health checking for all registered checks"""
        if self._running:
            return
        
        self._running = True
        
        # Start health check tasks
        for check_key, health_check in self.health_checks.items():
            if health_check.enabled:
                self._check_tasks[check_key] = asyncio.create_task(
                    self._health_check_loop(check_key, health_check)
                )
        
        logger.info(f"Started health checking for {len(self._check_tasks)} checks")
    
    async def stop_health_checking(self) -> None:
        """Stop all health checking"""
        self._running = False
        
        # Cancel all check tasks
        for task in self._check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._check_tasks:
            await asyncio.gather(*self._check_tasks.values(), return_exceptions=True)
        
        self._check_tasks.clear()
        logger.info("Stopped health checking")
    
    async def _health_check_loop(self, check_key: str, health_check: HealthCheck) -> None:
        """Health check loop for a specific check"""
        node_id = check_key.split('_')[0]  # Extract node_id from key
        
        while self._running:
            try:
                # Perform health check
                start_time = time.time()
                
                try:
                    # Run check with timeout
                    is_healthy = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, health_check.check_func),
                        timeout=health_check.timeout_seconds
                    )
                    response_time = (time.time() - start_time) * 1000  # ms
                except asyncio.TimeoutError:
                    is_healthy = False
                    response_time = health_check.timeout_seconds * 1000
                except Exception as e:
                    logger.warning(f"Health check {health_check.name} failed: {e}")
                    is_healthy = False
                    response_time = (time.time() - start_time) * 1000
                
                # Update node health
                await self._update_node_health(node_id, health_check.name, is_healthy, response_time)
                
                # Record result
                self.check_results[check_key].append({
                    "timestamp": datetime.now(),
                    "healthy": is_healthy,
                    "response_time_ms": response_time
                })
                
                await asyncio.sleep(health_check.interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error for {check_key}: {e}")
                await asyncio.sleep(health_check.interval_seconds)
    
    async def _update_node_health(self, 
                                 node_id: str, 
                                 check_name: str, 
                                 is_healthy: bool,
                                 response_time_ms: float) -> None:
        """Update node health based on check result"""
        async with self._lock:
            if node_id not in self.node_health:
                return
            
            node_health = self.node_health[node_id]
            node_health.last_check = datetime.now()
            node_health.total_checks += 1
            node_health.response_time_ms = response_time_ms
            node_health.checks[check_name] = is_healthy
            
            if is_healthy:
                node_health.consecutive_successes += 1
                node_health.consecutive_failures = 0
                
                # Remove any previous failure reason for this check
                node_health.failure_reasons = [
                    reason for reason in node_health.failure_reasons 
                    if not reason.startswith(f"{check_name}:")
                ]
            else:
                node_health.consecutive_failures += 1
                node_health.consecutive_successes = 0
                node_health.total_failures += 1
                
                # Add failure reason
                failure_reason = f"{check_name}: Health check failed"
                if failure_reason not in node_health.failure_reasons:
                    node_health.failure_reasons.append(failure_reason)
            
            # Update overall health status
            await self._calculate_node_health_status(node_id)
    
    async def _calculate_node_health_status(self, node_id: str) -> None:
        """Calculate overall health status for a node"""
        node_health = self.node_health[node_id]
        
        # Get critical and non-critical check results
        critical_checks = []
        non_critical_checks = []
        
        for check_key, health_check in self.health_checks.items():
            if check_key.startswith(f"{node_id}_"):
                check_name = check_key.split('_', 1)[1]
                if check_name in node_health.checks:
                    if health_check.critical:
                        critical_checks.append(node_health.checks[check_name])
                    else:
                        non_critical_checks.append(node_health.checks[check_name])
        
        # Determine status
        if critical_checks and not all(critical_checks):
            # Any critical check failing means critical status
            node_health.status = HealthStatus.CRITICAL
        elif not critical_checks and not non_critical_checks:
            # No checks means unknown
            node_health.status = HealthStatus.UNKNOWN
        elif node_health.consecutive_failures >= 5:
            # Many consecutive failures
            node_health.status = HealthStatus.UNHEALTHY
        elif node_health.consecutive_failures >= 2:
            # Some failures
            node_health.status = HealthStatus.DEGRADED
        elif all(critical_checks) and (not non_critical_checks or sum(non_critical_checks) / len(non_critical_checks) > 0.8):
            # All critical checks pass and most non-critical pass
            node_health.status = HealthStatus.HEALTHY
        else:
            node_health.status = HealthStatus.DEGRADED
    
    def get_node_health(self, node_id: str) -> Optional[NodeHealth]:
        """Get health information for a node"""
        return self.node_health.get(node_id)
    
    def get_all_node_health(self) -> Dict[str, NodeHealth]:
        """Get health information for all nodes"""
        return self.node_health.copy()
    
    def is_node_healthy(self, node_id: str) -> bool:
        """Check if a node is healthy"""
        node_health = self.node_health.get(node_id)
        return node_health and node_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

class FailoverManager:
    """Manages failover operations and recovery"""
    
    def __init__(self, 
                 strategy: FailoverStrategy = FailoverStrategy.ADAPTIVE,
                 health_checker: Optional[HealthChecker] = None):
        self.strategy = strategy
        self.health_checker = health_checker or HealthChecker()
        
        # Node management
        self.nodes: Dict[str, Dict[str, Any]] = {}  # node_id -> node info
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failover_groups: Dict[str, List[str]] = {}  # group -> [node_ids]
        
        # Event tracking
        self.failover_events: deque = deque(maxlen=1000)
        self.recovery_callbacks: List[Callable] = []
        
        # Background tasks
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._recovery_task: Optional[asyncio.Task] = None
        
        self._lock = asyncio.Lock()
        
        logger.info(f"FailoverManager initialized with {strategy.value} strategy")
    
    def register_node(self, 
                     node_id: str, 
                     node_info: Dict[str, Any],
                     failover_group: Optional[str] = None) -> None:
        """Register a node for failover management"""
        self.nodes[node_id] = {
            **node_info,
            "registered_at": datetime.now(),
            "active": True,
            "failover_group": failover_group
        }
        
        # Create circuit breaker
        self.circuit_breakers[node_id] = CircuitBreaker()
        
        # Add to failover group
        if failover_group:
            if failover_group not in self.failover_groups:
                self.failover_groups[failover_group] = []
            self.failover_groups[failover_group].append(node_id)
        
        # Register basic health checks
        self._register_basic_health_checks(node_id, node_info)
        
        logger.info(f"Registered node {node_id} for failover management")
    
    def unregister_node(self, node_id: str) -> None:
        """Unregister a node"""
        if node_id in self.nodes:
            node_info = self.nodes[node_id]
            failover_group = node_info.get("failover_group")
            
            # Remove from failover group
            if failover_group and failover_group in self.failover_groups:
                self.failover_groups[failover_group] = [
                    nid for nid in self.failover_groups[failover_group] if nid != node_id
                ]
            
            # Remove circuit breaker
            if node_id in self.circuit_breakers:
                del self.circuit_breakers[node_id]
            
            # Remove node
            del self.nodes[node_id]
            
            logger.info(f"Unregistered node {node_id}")
    
    def _register_basic_health_checks(self, node_id: str, node_info: Dict[str, Any]) -> None:
        """Register basic health checks for a node"""
        endpoint = node_info.get("endpoint")
        if not endpoint:
            return
        
        # HTTP health check
        def http_health_check():
            try:
                import requests
                response = requests.get(f"{endpoint}/health", timeout=5)
                return response.status_code == 200
            except Exception as e:
                logger.warning(f"Exception caught, returning: {e}")
                return False
        
        self.health_checker.register_health_check(
            node_id,
            HealthCheck(
                name="http_health",
                check_func=http_health_check,
                interval_seconds=10.0,
                timeout_seconds=5.0,
                failure_threshold=3,
                critical=True
            )
        )
        
        # Response time check
        def response_time_check():
            try:
                import requests
                start_time = time.time()
                response = requests.get(f"{endpoint}/health", timeout=10)
                response_time = (time.time() - start_time) * 1000
                return response.status_code == 200 and response_time < 5000  # 5 second max
            except Exception as e:
                logger.warning(f"Exception caught, returning: {e}")
                return False
        
        self.health_checker.register_health_check(
            node_id,
            HealthCheck(
                name="response_time",
                check_func=response_time_check,
                interval_seconds=30.0,
                timeout_seconds=10.0,
                failure_threshold=2,
                critical=False
            )
        )
    
    async def start(self) -> None:
        """Start failover management"""
        if self._running:
            return
        
        self._running = True
        
        # Start health checker
        await self.health_checker.start_health_checking()
        
        # Start monitoring and recovery tasks
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._recovery_task = asyncio.create_task(self._recovery_loop())
        
        logger.info("FailoverManager started")
    
    async def stop(self) -> None:
        """Stop failover management"""
        self._running = False
        
        # Stop health checker
        await self.health_checker.stop_health_checking()
        
        # Cancel tasks
        for task in [self._monitor_task, self._recovery_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("FailoverManager stopped")
    
    async def _monitor_loop(self) -> None:
        """Monitor nodes and trigger failover when needed"""
        while self._running:
            try:
                async with self._lock:
                    for node_id in list(self.nodes.keys()):
                        await self._check_node_for_failover(node_id)
                
                await asyncio.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(30)
    
    async def _check_node_for_failover(self, node_id: str) -> None:
        """Check if a node needs failover"""
        if not self.nodes[node_id]["active"]:
            return
        
        node_health = self.health_checker.get_node_health(node_id)
        if not node_health:
            return
        
        # Determine if failover is needed
        should_failover = False
        failover_reason = ""
        
        if node_health.status == HealthStatus.CRITICAL:
            should_failover = True
            failover_reason = "Node health is critical"
        elif (self.strategy == FailoverStrategy.IMMEDIATE and 
              node_health.status == HealthStatus.UNHEALTHY):
            should_failover = True
            failover_reason = "Node is unhealthy (immediate strategy)"
        elif (self.strategy == FailoverStrategy.CIRCUIT_BREAKER and
              self.circuit_breakers[node_id].get_state()["state"] == "open"):
            should_failover = True
            failover_reason = "Circuit breaker is open"
        elif (self.strategy == FailoverStrategy.ADAPTIVE and
              node_health.consecutive_failures >= 5):
            should_failover = True
            failover_reason = "Too many consecutive failures"
        
        if should_failover:
            await self._execute_failover(node_id, failover_reason)
    
    async def _execute_failover(self, failed_node_id: str, reason: str) -> None:
        """Execute failover for a failed node"""
        logger.warning(f"Executing failover for node {failed_node_id}: {reason}")
        
        # Mark node as inactive
        self.nodes[failed_node_id]["active"] = False
        
        # Find replacement node
        replacement_node = await self._find_replacement_node(failed_node_id)
        
        # Create failover event
        event = FailoverEvent(
            event_id=f"failover_{int(time.time())}_{failed_node_id}",
            event_type="node_failover",
            source_node=failed_node_id,
            target_node=replacement_node,
            timestamp=datetime.now(),
            reason=reason,
            success=replacement_node is not None
        )
        
        self.failover_events.append(event)
        
        # Notify recovery callbacks
        for callback in self.recovery_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")
        
        logger.info(f"Failover {'successful' if event.success else 'failed'} for node {failed_node_id}")
    
    async def _find_replacement_node(self, failed_node_id: str) -> Optional[str]:
        """Find a healthy replacement node"""
        failed_node = self.nodes[failed_node_id]
        failover_group = failed_node.get("failover_group")
        
        # Look for healthy nodes in the same failover group
        if failover_group and failover_group in self.failover_groups:
            for node_id in self.failover_groups[failover_group]:
                if (node_id != failed_node_id and 
                    self.nodes[node_id]["active"] and
                    self.health_checker.is_node_healthy(node_id)):
                    return node_id
        
        # Look for any healthy node
        for node_id, node_info in self.nodes.items():
            if (node_id != failed_node_id and
                node_info["active"] and
                self.health_checker.is_node_healthy(node_id)):
                return node_id
        
        return None
    
    async def _recovery_loop(self) -> None:
        """Recovery loop for failed nodes"""
        while self._running:
            try:
                async with self._lock:
                    await self._attempt_node_recovery()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
                await asyncio.sleep(60)
    
    async def _attempt_node_recovery(self) -> None:
        """Attempt to recover failed nodes"""
        for node_id, node_info in self.nodes.items():
            if not node_info["active"]:
                # Check if node has recovered
                node_health = self.health_checker.get_node_health(node_id)
                if (node_health and 
                    node_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] and
                    node_health.consecutive_successes >= 3):
                    
                    # Node has recovered
                    node_info["active"] = True
                    
                    # Reset circuit breaker
                    if node_id in self.circuit_breakers:
                        self.circuit_breakers[node_id] = CircuitBreaker()
                    
                    # Create recovery event
                    event = FailoverEvent(
                        event_id=f"recovery_{int(time.time())}_{node_id}",
                        event_type="node_recovery",
                        source_node=node_id,
                        target_node=None,
                        timestamp=datetime.now(),
                        reason="Node health restored",
                        success=True
                    )
                    
                    self.failover_events.append(event)
                    
                    logger.info(f"Node {node_id} has recovered and is back online")
    
    def add_recovery_callback(self, callback: Callable) -> None:
        """Add a callback for failover/recovery events"""
        self.recovery_callbacks.append(callback)
    
    def force_failover(self, node_id: str, reason: str = "Manual failover") -> None:
        """Manually trigger failover for a node"""
        if node_id in self.nodes:
            asyncio.create_task(self._execute_failover(node_id, reason))
    
    def get_node_status(self, node_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a node"""
        if node_id not in self.nodes:
            return {}
        
        node_info = self.nodes[node_id]
        node_health = self.health_checker.get_node_health(node_id)
        circuit_breaker_state = self.circuit_breakers[node_id].get_state() if node_id in self.circuit_breakers else {}
        
        return {
            "node_id": node_id,
            "active": node_info["active"],
            "endpoint": node_info.get("endpoint"),
            "failover_group": node_info.get("failover_group"),
            "health_status": node_health.status.value if node_health else "unknown",
            "consecutive_failures": node_health.consecutive_failures if node_health else 0,
            "consecutive_successes": node_health.consecutive_successes if node_health else 0,
            "last_check": node_health.last_check if node_health else None,
            "circuit_breaker": circuit_breaker_state,
            "failure_reasons": node_health.failure_reasons if node_health else []
        }
    
    def get_failover_stats(self) -> Dict[str, Any]:
        """Get failover statistics"""
        total_nodes = len(self.nodes)
        active_nodes = len([n for n in self.nodes.values() if n["active"]])
        total_events = len(self.failover_events)
        recent_events = [e for e in self.failover_events if e.timestamp > datetime.now() - timedelta(hours=24)]
        
        return {
            "strategy": self.strategy.value,
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "inactive_nodes": total_nodes - active_nodes,
            "total_failover_events": total_events,
            "recent_failover_events": len(recent_events),
            "failover_groups": len(self.failover_groups),
            "avg_recovery_time": self._calculate_avg_recovery_time()
        }
    
    def _calculate_avg_recovery_time(self) -> Optional[float]:
        """Calculate average recovery time"""
        recovery_times = []
        
        # Group events by node to calculate recovery times
        node_events = defaultdict(list)
        for event in self.failover_events:
            node_events[event.source_node].append(event)
        
        for events in node_events.values():
            events.sort(key=lambda e: e.timestamp)
            
            for i in range(len(events) - 1):
                if (events[i].event_type == "node_failover" and 
                    events[i + 1].event_type == "node_recovery"):
                    recovery_time = (events[i + 1].timestamp - events[i].timestamp).total_seconds()
                    recovery_times.append(recovery_time)
        
        return sum(recovery_times) / len(recovery_times) if recovery_times else None

# Global failover manager instance
_global_failover_manager: Optional[FailoverManager] = None

def get_global_failover_manager(**kwargs) -> FailoverManager:
    """Get or create global failover manager instance"""
    global _global_failover_manager
    if _global_failover_manager is None:
        _global_failover_manager = FailoverManager(**kwargs)
    return _global_failover_manager