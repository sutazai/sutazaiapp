"""
MCP-Specific Load Balancing Strategy
Implements intelligent load balancing for MCP servers based on their unique characteristics
"""
from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

from .service_mesh import ServiceInstance, LoadBalancerStrategy

logger = logging.getLogger(__name__)

@dataclass
class MCPInstanceMetrics:
    """Metrics for MCP instance load balancing decisions"""
    instance_id: str
    response_time_avg: float = 0.0
    error_rate: float = 0.0
    active_requests: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    capability_score: float = 1.0
    last_used: float = 0.0
    total_requests: int = 0
    consecutive_errors: int = 0

class MCPLoadBalancer:
    """
    MCP-aware load balancer with intelligent instance selection
    Considers MCP-specific factors like capability matching and resource usage
    """
    
    def __init__(self):
        self.instance_metrics: Dict[str, MCPInstanceMetrics] = {}
        self.service_capabilities: Dict[str, Dict[str, List[str]]] = {}
        self.sticky_sessions: Dict[str, str] = {}  # client_id -> instance_id
        self.session_timeout = 300  # 5 minutes
        
    def select_instance(self, 
                       instances: List[ServiceInstance],
                       service_name: str,
                       context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """
        Select best MCP instance based on multiple factors
        
        Args:
            instances: Available service instances
            service_name: Name of the MCP service
            context: Request context with optional hints
        
        Returns:
            Selected instance or None if no suitable instance
        """
        if not instances:
            return None
        
        # Filter healthy instances
        healthy_instances = [
            inst for inst in instances 
            if inst.state.value in ["healthy", "degraded"]
        ]
        
        if not healthy_instances:
            logger.warning(f"No healthy instances for {service_name}")
            return None
        
        # Check for sticky session
        if context and "client_id" in context:
            instance_id = self._check_sticky_session(context["client_id"], healthy_instances)
            if instance_id:
                return self._find_instance_by_id(healthy_instances, instance_id)
        
        # Apply selection strategy based on service type
        strategy = self._determine_strategy(service_name, context)
        
        if strategy == "capability":
            return self._select_by_capability(healthy_instances, service_name, context)
        elif strategy == "least_loaded":
            return self._select_least_loaded(healthy_instances)
        elif strategy == "fastest_response":
            return self._select_fastest_response(healthy_instances)
        elif strategy == "resource_aware":
            return self._select_resource_aware(healthy_instances)
        else:
            return self._select_weighted_random(healthy_instances)
    
    def _determine_strategy(self, service_name: str, context: Optional[Dict[str, Any]]) -> str:
        """Determine selection strategy based on service type and context"""
        
        # Language services need capability matching
        if "language" in service_name or "coder" in service_name:
            return "capability"
        
        # Database services need least connections
        if "postgres" in service_name or "memory" in service_name:
            return "least_loaded"
        
        # Browser automation needs resource awareness
        if "puppeteer" in service_name or "playwright" in service_name:
            return "resource_aware"
        
        # HTTP/Search services need fastest response
        if "http" in service_name or "ddg" in service_name:
            return "fastest_response"
        
        # Default to weighted random
        return "weighted_random"
    
    def _select_by_capability(self, 
                             instances: List[ServiceInstance],
                             service_name: str,
                             context: Optional[Dict[str, Any]]) -> ServiceInstance:
        """Select instance based on capability matching"""
        
        if not context or "required_capabilities" not in context:
            return random.choice(instances)
        
        required_caps = set(context["required_capabilities"])
        best_match = None
        best_score = 0
        
        for instance in instances:
            # Get instance capabilities from metadata
            caps = set(instance.metadata.get("capabilities", []))
            
            # Calculate match score
            score = len(caps.intersection(required_caps))
            
            # Bonus for exact match
            if caps == required_caps:
                score += 10
            
            # Consider instance metrics
            metrics = self._get_or_create_metrics(instance.service_id)
            score *= (1 - metrics.error_rate)  # Penalize errors
            score *= metrics.capability_score  # Custom capability score
            
            if score > best_score:
                best_score = score
                best_match = instance
        
        return best_match or random.choice(instances)
    
    def _select_least_loaded(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least active connections/requests"""
        
        least_loaded = None
        min_load = float('inf')
        
        for instance in instances:
            metrics = self._get_or_create_metrics(instance.service_id)
            
            # Calculate load score
            load = metrics.active_requests + (metrics.error_rate * 10)
            
            # Prefer instances not recently used
            if time.time() - metrics.last_used > 1:
                load *= 0.9
            
            if load < min_load:
                min_load = load
                least_loaded = instance
        
        return least_loaded or instances[0]
    
    def _select_fastest_response(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with fastest average response time"""
        
        fastest = None
        best_time = float('inf')
        
        for instance in instances:
            metrics = self._get_or_create_metrics(instance.service_id)
            
            # New instances get a chance
            if metrics.total_requests == 0:
                return instance
            
            # Calculate effective response time
            effective_time = metrics.response_time_avg
            
            # Penalize errors
            if metrics.error_rate > 0:
                effective_time *= (1 + metrics.error_rate)
            
            if effective_time < best_time:
                best_time = effective_time
                fastest = instance
        
        return fastest or instances[0]
    
    def _select_resource_aware(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance based on resource availability"""
        
        best = None
        best_score = 0
        
        for instance in instances:
            metrics = self._get_or_create_metrics(instance.service_id)
            
            # Calculate resource score (lower usage is better)
            cpu_score = 1 - min(metrics.cpu_usage, 1.0)
            mem_score = 1 - min(metrics.memory_usage, 1.0)
            
            # Combined score with weights
            score = (cpu_score * 0.4) + (mem_score * 0.4) + (0.2 * (1 - metrics.error_rate))
            
            # Boost for idle instances
            if metrics.active_requests == 0:
                score *= 1.2
            
            if score > best_score:
                best_score = score
                best = instance
        
        return best or instances[0]
    
    def _select_weighted_random(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance using weighted random selection"""
        
        weights = []
        for instance in instances:
            metrics = self._get_or_create_metrics(instance.service_id)
            
            # Base weight from instance configuration
            weight = instance.weight
            
            # Adjust based on error rate
            if metrics.error_rate > 0:
                weight *= (1 - metrics.error_rate)
            
            # Adjust based on consecutive errors
            if metrics.consecutive_errors > 0:
                weight *= (0.5 ** metrics.consecutive_errors)
            
            weights.append(max(weight, 0.1))  # Minimum weight
        
        # Weighted random selection
        total = sum(weights)
        r = random.uniform(0, total)
        
        current = 0
        for instance, weight in zip(instances, weights):
            current += weight
            if r <= current:
                return instance
        
        return instances[-1]
    
    def _check_sticky_session(self, client_id: str, instances: List[ServiceInstance]) -> Optional[str]:
        """Check for sticky session and validate instance still available"""
        
        if client_id not in self.sticky_sessions:
            return None
        
        instance_id = self.sticky_sessions[client_id]
        
        # Check if instance still available
        for instance in instances:
            if instance.service_id == instance_id:
                return instance_id
        
        # Instance no longer available, clear session
        del self.sticky_sessions[client_id]
        return None
    
    def _find_instance_by_id(self, instances: List[ServiceInstance], instance_id: str) -> Optional[ServiceInstance]:
        """Find instance by ID"""
        for instance in instances:
            if instance.service_id == instance_id:
                return instance
        return None
    
    def _get_or_create_metrics(self, instance_id: str) -> MCPInstanceMetrics:
        """Get or create metrics for instance"""
        if instance_id not in self.instance_metrics:
            self.instance_metrics[instance_id] = MCPInstanceMetrics(instance_id=instance_id)
        return self.instance_metrics[instance_id]
    
    def update_metrics(self, 
                      instance_id: str,
                      response_time: Optional[float] = None,
                      success: bool = True,
                      resource_usage: Optional[Dict[str, float]] = None):
        """Update metrics for an instance after request"""
        
        metrics = self._get_or_create_metrics(instance_id)
        
        # Update request count
        metrics.total_requests += 1
        metrics.last_used = time.time()
        
        # Update response time
        if response_time is not None:
            # Exponential moving average
            alpha = 0.3
            if metrics.response_time_avg == 0:
                metrics.response_time_avg = response_time
            else:
                metrics.response_time_avg = (alpha * response_time + 
                                            (1 - alpha) * metrics.response_time_avg)
        
        # Update error tracking
        if success:
            metrics.consecutive_errors = 0
        else:
            metrics.consecutive_errors += 1
        
        # Update error rate (sliding window)
        window_size = 100
        if metrics.total_requests <= window_size:
            # Simple average for initial requests
            metrics.error_rate = (metrics.error_rate * (metrics.total_requests - 1) + 
                                 (0 if success else 1)) / metrics.total_requests
        else:
            # Exponential decay for long-running instances
            metrics.error_rate = metrics.error_rate * 0.99 + (0 if success else 0.01)
        
        # Update resource usage if provided
        if resource_usage:
            if "cpu" in resource_usage:
                metrics.cpu_usage = resource_usage["cpu"]
            if "memory" in resource_usage:
                metrics.memory_usage = resource_usage["memory"]
    
    def set_capability_score(self, instance_id: str, score: float):
        """Set capability score for an instance"""
        metrics = self._get_or_create_metrics(instance_id)
        metrics.capability_score = max(0.1, min(score, 10.0))  # Clamp between 0.1 and 10
    
    def create_sticky_session(self, client_id: str, instance_id: str):
        """Create sticky session for client"""
        self.sticky_sessions[client_id] = instance_id
    
    def clear_expired_sessions(self):
        """Clear expired sticky sessions"""
        # This should be called periodically
        # For now, we'll keep it simple without timestamp tracking
        pass
    
    def get_instance_stats(self, instance_id: str) -> Dict[str, Any]:
        """Get statistics for an instance"""
        metrics = self._get_or_create_metrics(instance_id)
        return {
            "instance_id": instance_id,
            "total_requests": metrics.total_requests,
            "response_time_avg": metrics.response_time_avg,
            "error_rate": metrics.error_rate,
            "active_requests": metrics.active_requests,
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "capability_score": metrics.capability_score,
            "consecutive_errors": metrics.consecutive_errors,
            "last_used": metrics.last_used
        }

# Global MCP load balancer instance
_mcp_load_balancer = MCPLoadBalancer()

def get_mcp_load_balancer() -> MCPLoadBalancer:
    """Get the global MCP load balancer instance"""
    return _mcp_load_balancer