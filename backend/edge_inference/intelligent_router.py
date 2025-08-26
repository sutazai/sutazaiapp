"""
Intelligent Router - Advanced routing logic with performance optimization for edge inference
"""

import asyncio
import time
import math
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import heapq
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class RoutingObjective(Enum):
    """Routing optimization objectives"""
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_COST = "minimize_cost"
    OPTIMIZE_ACCURACY = "optimize_accuracy"
    MULTI_OBJECTIVE = "multi_objective"

class NodeStatus(Enum):
    """Node status for routing decisions"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    UNREACHABLE = "unreachable"
    MAINTENANCE = "maintenance"

@dataclass
class RoutingNode:
    """Node information for routing decisions"""
    node_id: str
    endpoint: str
    status: NodeStatus
    current_load: float  # 0.0 to 1.0
    avg_latency_ms: float
    error_rate: float  # 0.0 to 1.0
    throughput_rps: float  # Requests per second
    memory_usage: float  # 0.0 to 1.0
    cpu_usage: float  # 0.0 to 1.0
    model_capabilities: Set[str]
    hardware_score: float  # Relative performance score
    cost_per_request: float
    location: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    # Dynamic metrics
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=50))
    request_count: int = 0
    success_count: int = 0

@dataclass
class RoutingRequest:
    """Request information for routing decisions"""
    request_id: str
    model_name: str
    priority_level: int  # 1 (highest) to 10 (lowest)
    estimated_complexity: float  # Relative complexity score
    latency_requirement: Optional[float]  # Max acceptable latency in ms
    accuracy_requirement: Optional[float]  # Min required accuracy
    client_location: Optional[str] = None
    context_length: int = 2048
    requires_gpu: bool = False
    batch_size: int = 1

@dataclass
class RoutingDecision:
    """Routing decision result"""
    selected_node: RoutingNode
    confidence_score: float  # 0.0 to 1.0
    expected_latency_ms: float
    expected_accuracy: float
    reasoning: str
    alternative_nodes: List[RoutingNode] = field(default_factory=list)

class PerformancePredictor:
    """Predicts performance metrics for routing decisions"""
    
    def __init__(self):
        self.historical_data: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)  # node_id -> [(load, latency, timestamp)]
        self.model_performance: Dict[Tuple[str, str], List[float]] = defaultdict(list)  # (node_id, model) -> [latencies]
        self._lock = threading.RLock()
    
    def record_performance(self, 
                         node_id: str, 
                         model_name: str,
                         latency_ms: float,
                         load: float,
                         success: bool) -> None:
        """Record performance data for learning"""
        with self._lock:
            timestamp = time.time()
            self.historical_data[node_id].append((load, latency_ms, timestamp))
            
            # Keep only recent data (last 1000 entries)
            if len(self.historical_data[node_id]) > 1000:
                self.historical_data[node_id] = self.historical_data[node_id][-1000:]
            
            if success:
                self.model_performance[(node_id, model_name)].append(latency_ms)
                # Keep only recent model performance data
                if len(self.model_performance[(node_id, model_name)]) > 100:
                    self.model_performance[(node_id, model_name)] = self.model_performance[(node_id, model_name)][-100:]
    
    def predict_latency(self, node: RoutingNode, request: RoutingRequest) -> float:
        """Predict latency for a request on a node"""
        node_id = node.node_id
        model_name = request.model_name
        
        with self._lock:
            # Use model-specific data if available
            model_key = (node_id, model_name)
            if model_key in self.model_performance and self.model_performance[model_key]:
                recent_latencies = self.model_performance[model_key][-20:]  # Last 20 requests
                base_latency = np.median(recent_latencies)
            else:
                # Fall back to general node performance
                base_latency = node.avg_latency_ms
            
            # Adjust for current load
            load_factor = 1.0 + (node.current_load ** 2) * 2.0  # Quadratic load impact
            
            # Adjust for request complexity
            complexity_factor = 1.0 + request.estimated_complexity * 0.5
            
            # Adjust for context length
            context_factor = 1.0 + max(0, (request.context_length - 2048) / 2048) * 0.3
            
            predicted_latency = base_latency * load_factor * complexity_factor * context_factor
            
            return max(predicted_latency, 10.0)  # Minimum 10ms
    
    def predict_success_probability(self, node: RoutingNode, request: RoutingRequest) -> float:
        """Predict probability of successful request completion"""
        # Base success rate
        base_success = 1.0 - node.error_rate
        
        # Adjust for node health
        if node.status == NodeStatus.HEALTHY:
            health_factor = 1.0
        elif node.status == NodeStatus.DEGRADED:
            health_factor = 0.8
        elif node.status == NodeStatus.OVERLOADED:
            health_factor = 0.5
        else:
            health_factor = 0.1  # Unreachable or maintenance
        
        # Adjust for load
        load_factor = max(0.3, 1.0 - node.current_load * 0.5)
        
        # Adjust for model capability
        capability_factor = 1.0 if request.model_name in node.model_capabilities else 0.7
        
        return min(1.0, base_success * health_factor * load_factor * capability_factor)

class LoadBalancer:
    """Advanced load balancing with multiple algorithms"""
    
    def __init__(self):
        self.round_robin_counter = 0
        self.weighted_round_robin_state: Dict[str, int] = {}
        self.least_connections_state: Dict[str, int] = defaultdict(int)
    
    def select_by_round_robin(self, nodes: List[RoutingNode]) -> Optional[RoutingNode]:
        """Simple round-robin selection"""
        if not nodes:
            return None
        
        selected = nodes[self.round_robin_counter % len(nodes)]
        self.round_robin_counter += 1
        return selected
    
    def select_by_weighted_round_robin(self, nodes: List[RoutingNode]) -> Optional[RoutingNode]:
        """Weighted round-robin based on hardware scores"""
        if not nodes:
            return None
        
        # Calculate weights based on hardware scores
        total_weight = sum(node.hardware_score for node in nodes)
        if total_weight <= 0:
            return self.select_by_round_robin(nodes)
        
        # Select based on weighted probability
        target = np.random.random() * total_weight
        current = 0.0
        
        for node in nodes:
            current += node.hardware_score
            if current >= target:
                return node
        
        return nodes[-1]  # Fallback
    
    def select_by_least_connections(self, nodes: List[RoutingNode]) -> Optional[RoutingNode]:
        """Least connections algorithm"""
        if not nodes:
            return None
        
        # Find node with least active connections
        min_connections = float('inf')
        selected_node = None
        
        for node in nodes:
            connections = self.least_connections_state.get(node.node_id, 0)
            if connections < min_connections:
                min_connections = connections
                selected_node = node
        
        return selected_node
    
    def select_by_response_time(self, nodes: List[RoutingNode]) -> Optional[RoutingNode]:
        """Select node with best average response time"""
        if not nodes:
            return None
        
        return min(nodes, key=lambda n: n.avg_latency_ms)
    
    def select_by_load(self, nodes: List[RoutingNode]) -> Optional[RoutingNode]:
        """Select node with lowest current load"""
        if not nodes:
            return None
        
        return min(nodes, key=lambda n: n.current_load)
    
    def update_connections(self, node_id: str, delta: int) -> None:
        """Update connection count for a node"""
        self.least_connections_state[node_id] = max(0, self.least_connections_state[node_id] + delta)

class MultiObjectiveOptimizer:
    """Multi-objective optimization for routing decisions"""
    
    def __init__(self, weights: Dict[str, float] = None):
        # Default weights for different objectives
        self.weights = weights or {
            "latency": 0.3,
            "throughput": 0.2,
            "load": 0.2,
            "reliability": 0.15,
            "cost": 0.1,
            "accuracy": 0.05
        }
    
    def calculate_node_score(self, 
                           node: RoutingNode, 
                           request: RoutingRequest,
                           predicted_latency: float,
                           success_probability: float) -> Tuple[float, str]:
        """Calculate overall score for a node"""
        scores = {}
        reasoning_parts = []
        
        # Latency score (lower is better)
        if request.latency_requirement:
            latency_score = max(0, 1.0 - (predicted_latency / request.latency_requirement))
        else:
            latency_score = max(0, 1.0 - (predicted_latency / 1000.0))  # Normalize to 1s
        scores["latency"] = latency_score
        reasoning_parts.append(f"latency: {latency_score:.2f}")
        
        # Throughput score (higher is better)
        max_throughput = 100.0  # Assume max 100 RPS
        throughput_score = min(1.0, node.throughput_rps / max_throughput)
        scores["throughput"] = throughput_score
        reasoning_parts.append(f"throughput: {throughput_score:.2f}")
        
        # Load score (lower load is better)
        load_score = 1.0 - node.current_load
        scores["load"] = load_score
        reasoning_parts.append(f"load: {load_score:.2f}")
        
        # Reliability score
        reliability_score = success_probability
        scores["reliability"] = reliability_score
        reasoning_parts.append(f"reliability: {reliability_score:.2f}")
        
        # Cost score (lower cost is better)
        max_cost = 1.0  # Assume max cost per request
        cost_score = 1.0 - min(1.0, node.cost_per_request / max_cost)
        scores["cost"] = cost_score
        reasoning_parts.append(f"cost: {cost_score:.2f}")
        
        # Accuracy score (model capability)
        accuracy_score = 1.0 if request.model_name in node.model_capabilities else 0.5
        scores["accuracy"] = accuracy_score
        reasoning_parts.append(f"accuracy: {accuracy_score:.2f}")
        
        # Calculate weighted sum
        total_score = sum(scores[metric] * self.weights[metric] for metric in scores)
        reasoning = f"weighted_score({', '.join(reasoning_parts)}) = {total_score:.3f}"
        
        return total_score, reasoning

class IntelligentRouter:
    """Main intelligent routing system with advanced optimization"""
    
    def __init__(self,
                 routing_objective: RoutingObjective = RoutingObjective.MULTI_OBJECTIVE,
                 enable_prediction: bool = True,
                 enable_learning: bool = True,
                 update_interval: float = 30.0):
        
        self.routing_objective = routing_objective
        self.enable_prediction = enable_prediction
        self.enable_learning = enable_learning
        self.update_interval = update_interval
        
        # Node management
        self.nodes: Dict[str, RoutingNode] = {}
        self.node_health_scores: Dict[str, float] = {}
        
        # Components
        self.performance_predictor = PerformancePredictor() if enable_prediction else None
        self.load_balancer = LoadBalancer()
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        
        # Routing statistics
        self.routing_stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "avg_decision_time_ms": 0.0,
            "objective_scores": defaultdict(list)
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._learning_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Threading
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._lock = asyncio.Lock()
        
        logger.info(f"IntelligentRouter initialized with {routing_objective.value} objective")
    
    async def start(self) -> None:
        """Start the intelligent router"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        if self.enable_learning:
            self._learning_task = asyncio.create_task(self._learning_loop())
        
        logger.info("IntelligentRouter started")
    
    async def stop(self) -> None:
        """Stop the intelligent router"""
        self._running = False
        
        # Cancel background tasks
        for task in [self._monitoring_task, self._learning_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        logger.info("IntelligentRouter stopped")
    
    def register_node(self, node: RoutingNode) -> None:
        """Register a node for routing"""
        self.nodes[node.node_id] = node
        self.node_health_scores[node.node_id] = 1.0
        logger.info(f"Registered node {node.node_id} at {node.endpoint}")
    
    def unregister_node(self, node_id: str) -> None:
        """Unregister a node"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            if node_id in self.node_health_scores:
                del self.node_health_scores[node_id]
            logger.info(f"Unregistered node {node_id}")
    
    def update_node_metrics(self, 
                          node_id: str,
                          latency_ms: float,
                          error_rate: float,
                          throughput_rps: float,
                          load: float,
                          memory_usage: float,
                          cpu_usage: float) -> None:
        """Update node performance metrics"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.avg_latency_ms = latency_ms
        node.error_rate = error_rate
        node.throughput_rps = throughput_rps
        node.current_load = load
        node.memory_usage = memory_usage
        node.cpu_usage = cpu_usage
        node.last_heartbeat = datetime.now()
        
        # Update status based on metrics
        self._update_node_status(node)
    
    def _update_node_status(self, node: RoutingNode) -> None:
        """Update node status based on current metrics"""
        # Check if node is reachable (recent heartbeat)
        if (datetime.now() - node.last_heartbeat).total_seconds() > 60:
            node.status = NodeStatus.UNREACHABLE
            return
        
        # Check if overloaded
        if node.current_load > 0.9 or node.cpu_usage > 0.95 or node.memory_usage > 0.95:
            node.status = NodeStatus.OVERLOADED
        elif node.error_rate > 0.1 or node.current_load > 0.8:
            node.status = NodeStatus.DEGRADED
        else:
            node.status = NodeStatus.HEALTHY
        
        # Update health score
        health_factors = [
            1.0 - node.error_rate,
            1.0 - node.current_load,
            1.0 - node.cpu_usage,
            1.0 - node.memory_usage
        ]
        self.node_health_scores[node.node_id] = np.mean(health_factors)
    
    async def route_request(self, request: RoutingRequest) -> Optional[RoutingDecision]:
        """Make intelligent routing decision for a request"""
        start_time = time.time()
        self.routing_stats["total_requests"] += 1
        
        try:
            # Filter available nodes
            available_nodes = await self._filter_available_nodes(request)
            if not available_nodes:
                logger.warning(f"No available nodes for request {request.request_id}")
                self.routing_stats["failed_routes"] += 1
                return None
            
            # Select optimal node based on objective
            decision = await self._select_optimal_node(available_nodes, request)
            
            if decision:
                self.routing_stats["successful_routes"] += 1
                
                # Update load balancer state
                self.load_balancer.update_connections(decision.selected_node.node_id, 1)
                
                # Record decision time
                decision_time = (time.time() - start_time) * 1000
                self.routing_stats["avg_decision_time_ms"] = (
                    self.routing_stats["avg_decision_time_ms"] * 0.9 + decision_time * 0.1
                )
                
                logger.debug(f"Routed request {request.request_id} to {decision.selected_node.node_id} "
                           f"(confidence: {decision.confidence_score:.2f})")
                return decision
            else:
                self.routing_stats["failed_routes"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Routing failed for request {request.request_id}: {e}")
            self.routing_stats["failed_routes"] += 1
            return None
    
    async def _filter_available_nodes(self, request: RoutingRequest) -> List[RoutingNode]:
        """Filter nodes based on request requirements"""
        available_nodes = []
        
        for node in self.nodes.values():
            # Check basic availability
            if node.status in [NodeStatus.UNREACHABLE, NodeStatus.MAINTENANCE]:
                continue
            
            # Check model capability
            if request.model_name not in node.model_capabilities:
                continue
            
            # Check GPU requirement
            if request.requires_gpu and not node.model_capabilities.intersection({"gpu", "cuda"}):
                continue
            
            # Check if node can handle the load
            if node.current_load > 0.95:  # 95% load threshold
                continue
            
            available_nodes.append(node)
        
        return available_nodes
    
    async def _select_optimal_node(self, 
                                  nodes: List[RoutingNode], 
                                  request: RoutingRequest) -> Optional[RoutingDecision]:
        """Select optimal node based on routing objective"""
        if not nodes:
            return None
        
        if self.routing_objective == RoutingObjective.MINIMIZE_LATENCY:
            return await self._select_by_latency(nodes, request)
        elif self.routing_objective == RoutingObjective.MAXIMIZE_THROUGHPUT:
            return await self._select_by_throughput(nodes, request)
        elif self.routing_objective == RoutingObjective.BALANCE_LOAD:
            return await self._select_by_load_balance(nodes, request)
        elif self.routing_objective == RoutingObjective.MINIMIZE_COST:
            return await self._select_by_cost(nodes, request)
        elif self.routing_objective == RoutingObjective.MULTI_OBJECTIVE:
            return await self._select_by_multi_objective(nodes, request)
        else:
            # Default: simple load balancing
            selected_node = self.load_balancer.select_by_load(nodes)
            return RoutingDecision(
                selected_node=selected_node,
                confidence_score=0.5,
                expected_latency_ms=selected_node.avg_latency_ms,
                expected_accuracy=0.9,
                reasoning="Default load-based selection"
            ) if selected_node else None
    
    async def _select_by_latency(self, 
                               nodes: List[RoutingNode], 
                               request: RoutingRequest) -> Optional[RoutingDecision]:
        """Select node with lowest predicted latency"""
        if not nodes:
            return None
        
        best_node = None
        best_latency = float('inf')
        
        for node in nodes:
            if self.performance_predictor:
                predicted_latency = self.performance_predictor.predict_latency(node, request)
            else:
                predicted_latency = node.avg_latency_ms
            
            if predicted_latency < best_latency:
                best_latency = predicted_latency
                best_node = node
        
        return RoutingDecision(
            selected_node=best_node,
            confidence_score=0.8,
            expected_latency_ms=best_latency,
            expected_accuracy=0.9,
            reasoning=f"Selected for minimum latency: {best_latency:.1f}ms"
        ) if best_node else None
    
    async def _select_by_throughput(self, 
                                  nodes: List[RoutingNode], 
                                  request: RoutingRequest) -> Optional[RoutingDecision]:
        """Select node with highest throughput capacity"""
        best_node = max(nodes, key=lambda n: n.throughput_rps * (1.0 - n.current_load))
        
        return RoutingDecision(
            selected_node=best_node,
            confidence_score=0.7,
            expected_latency_ms=best_node.avg_latency_ms,
            expected_accuracy=0.9,
            reasoning=f"Selected for throughput: {best_node.throughput_rps:.1f} RPS"
        )
    
    async def _select_by_load_balance(self, 
                                    nodes: List[RoutingNode], 
                                    request: RoutingRequest) -> Optional[RoutingDecision]:
        """Select node for optimal load balancing"""
        best_node = self.load_balancer.select_by_least_connections(nodes)
        
        return RoutingDecision(
            selected_node=best_node,
            confidence_score=0.6,
            expected_latency_ms=best_node.avg_latency_ms,
            expected_accuracy=0.9,
            reasoning=f"Selected for load balancing: {best_node.current_load:.2f} load"
        ) if best_node else None
    
    async def _select_by_cost(self, 
                            nodes: List[RoutingNode], 
                            request: RoutingRequest) -> Optional[RoutingDecision]:
        """Select node with lowest cost"""
        best_node = min(nodes, key=lambda n: n.cost_per_request)
        
        return RoutingDecision(
            selected_node=best_node,
            confidence_score=0.5,
            expected_latency_ms=best_node.avg_latency_ms,
            expected_accuracy=0.9,
            reasoning=f"Selected for cost: ${best_node.cost_per_request:.4f} per request"
        )
    
    async def _select_by_multi_objective(self, 
                                       nodes: List[RoutingNode], 
                                       request: RoutingRequest) -> Optional[RoutingDecision]:
        """Select node using multi-objective optimization"""
        best_node = None
        best_score = -1.0
        best_reasoning = ""
        alternatives = []
        
        for node in nodes:
            # Predict performance
            if self.performance_predictor:
                predicted_latency = self.performance_predictor.predict_latency(node, request)
                success_probability = self.performance_predictor.predict_success_probability(node, request)
            else:
                predicted_latency = node.avg_latency_ms
                success_probability = 1.0 - node.error_rate
            
            # Calculate multi-objective score
            score, reasoning = self.multi_objective_optimizer.calculate_node_score(
                node, request, predicted_latency, success_probability
            )
            
            if score > best_score:
                if best_node:  # Add previous best to alternatives
                    alternatives.append(best_node)
                best_score = score
                best_node = node
                best_reasoning = reasoning
            else:
                alternatives.append(node)
        
        # Sort alternatives by score
        alternatives.sort(key=lambda n: self.multi_objective_optimizer.calculate_node_score(
            n, request, 
            self.performance_predictor.predict_latency(n, request) if self.performance_predictor else n.avg_latency_ms,
            self.performance_predictor.predict_success_probability(n, request) if self.performance_predictor else 1.0 - n.error_rate
        )[0], reverse=True)
        
        return RoutingDecision(
            selected_node=best_node,
            confidence_score=min(1.0, best_score),
            expected_latency_ms=self.performance_predictor.predict_latency(best_node, request) if self.performance_predictor else best_node.avg_latency_ms,
            expected_accuracy=0.9,
            reasoning=f"Multi-objective optimization: {best_reasoning}",
            alternative_nodes=alternatives[:3]  # Top 3 alternatives
        ) if best_node else None
    
    def record_request_completion(self, 
                                node_id: str,
                                model_name: str,
                                latency_ms: float,
                                success: bool) -> None:
        """Record completion of a request for learning"""
        if self.performance_predictor and self.enable_learning:
            node = self.nodes.get(node_id)
            if node:
                self.performance_predictor.record_performance(
                    node_id, model_name, latency_ms, node.current_load, success
                )
        
        # Update load balancer
        self.load_balancer.update_connections(node_id, -1)
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self._running:
            try:
                # Update node health scores
                for node_id, node in self.nodes.items():
                    self._update_node_status(node)
                
                # Clean up old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _learning_loop(self) -> None:
        """Background learning loop"""
        while self._running:
            try:
                # Adjust routing weights based on performance
                await self._adaptive_weight_adjustment()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old performance data"""
        cutoff_time = time.time() - 3600  # 1 hour ago
        
        if self.performance_predictor:
            for node_id in list(self.performance_predictor.historical_data.keys()):
                old_data = self.performance_predictor.historical_data[node_id]
                # Remove entries older than cutoff
                recent_data = [(load, latency, ts) for load, latency, ts in old_data if ts > cutoff_time]
                self.performance_predictor.historical_data[node_id] = recent_data
    
    async def _adaptive_weight_adjustment(self) -> None:
        """Adaptively adjust routing weights based on performance"""
        # Analyze recent routing performance
        success_rate = self.routing_stats["successful_routes"] / max(self.routing_stats["total_requests"], 1)
        
        if success_rate < 0.9:  # Low success rate
            # Increase reliability weight
            self.multi_objective_optimizer.weights["reliability"] *= 1.1
            self.multi_objective_optimizer.weights["latency"] *= 0.9
            logger.info("Adjusted weights to prioritize reliability")
        elif success_rate > 0.98:  # High success rate
            # Increase performance weights
            self.multi_objective_optimizer.weights["latency"] *= 1.1
            self.multi_objective_optimizer.weights["throughput"] *= 1.05
            logger.info("Adjusted weights to prioritize performance")
        
        # Normalize weights
        total_weight = sum(self.multi_objective_optimizer.weights.values())
        for key in self.multi_objective_optimizer.weights:
            self.multi_objective_optimizer.weights[key] /= total_weight
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        total_requests = self.routing_stats["total_requests"]
        success_rate = self.routing_stats["successful_routes"] / max(total_requests, 1)
        
        return {
            **self.routing_stats,
            "success_rate": success_rate,
            "node_count": len(self.nodes),
            "healthy_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.HEALTHY]),
            "routing_objective": self.routing_objective.value,
            "weights": self.multi_objective_optimizer.weights.copy()
        }
    
    def get_node_stats(self) -> Dict[str, Any]:
        """Get node statistics"""
        return {
            node_id: {
                "status": node.status.value,
                "current_load": node.current_load,
                "avg_latency_ms": node.avg_latency_ms,
                "error_rate": node.error_rate,
                "throughput_rps": node.throughput_rps,
                "health_score": self.node_health_scores.get(node_id, 0.0),
                "request_count": node.request_count,
                "success_count": node.success_count
            }
            for node_id, node in self.nodes.items()
        }

# Global router instance
_global_router: Optional[IntelligentRouter] = None

def get_global_router(**kwargs) -> IntelligentRouter:
    """Get or create global intelligent router instance"""
    global _global_router
    if _global_router is None:
        _global_router = IntelligentRouter(**kwargs)
    return _global_router