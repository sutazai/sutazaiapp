"""
Agent Discovery and Capability Matching System
Provides intelligent agent discovery, capability matching, and load balancing.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import aiohttp
from universal_client import (
    AgentInfo, AgentCapability, AgentStatus, Priority, TaskRequest,
    UniversalAgentClient, AgentType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchingAlgorithm(Enum):
    """Agent matching algorithms."""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_MATCH = "semantic_match"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_SCORE = "capability_score"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RESPONSE_TIME = "response_time"
    RANDOM = "random"
    PRIORITY_BASED = "priority_based"


@dataclass
class CapabilityRequirement:
    """Represents a capability requirement for task matching."""
    name: str
    required: bool = True
    weight: float = 1.0
    min_score: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRequirements:
    """Complete requirements specification for task matching."""
    capabilities: List[CapabilityRequirement]
    priority: Priority = Priority.MEDIUM
    max_response_time: Optional[float] = None
    min_success_rate: Optional[float] = None
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    exclude_agents: Set[str] = field(default_factory=set)
    prefer_agents: Set[str] = field(default_factory=set)


@dataclass
class AgentMatch:
    """Represents a matched agent with scoring information."""
    agent_info: AgentInfo
    match_score: float
    capability_scores: Dict[str, float]
    load_score: float
    priority_score: float
    response_time_score: float
    total_score: float
    reasoning: List[str] = field(default_factory=list)


@dataclass
class DiscoveryMetrics:
    """Metrics for agent discovery and performance."""
    agent_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    current_load: float = 0.0
    last_health_check: Optional[datetime] = None
    uptime_percentage: float = 100.0
    capability_usage: Dict[str, int] = field(default_factory=dict)


class AgentRegistry:
    """Enhanced agent registry with discovery and metrics."""
    
    def __init__(self):
        self.agents: Dict[str, AgentInfo] = {}
        self.metrics: Dict[str, DiscoveryMetrics] = {}
        self.capability_index: Dict[str, Set[str]] = defaultdict(set)
        self.load_balancer_state: Dict[str, int] = defaultdict(int)  # Round-robin counters
        self.semantic_cache: Dict[str, List[str]] = {}  # Cache for semantic matches
    
    def register_agent(self, agent_info: AgentInfo):
        """Register an agent and update indexes."""
        self.agents[agent_info.id] = agent_info
        
        # Initialize metrics if not exists
        if agent_info.id not in self.metrics:
            self.metrics[agent_info.id] = DiscoveryMetrics(agent_id=agent_info.id)
        
        # Update capability index
        for capability in agent_info.capabilities:
            self.capability_index[capability.name.lower()].add(agent_info.id)
            
            # Also index by keywords
            keywords = capability.name.lower().replace('_', ' ').split()
            for keyword in keywords:
                self.capability_index[keyword].add(agent_info.id)
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent and clean up indexes."""
        if agent_id in self.agents:
            agent_info = self.agents[agent_id]
            
            # Remove from capability index
            for capability in agent_info.capabilities:
                self.capability_index[capability.name.lower()].discard(agent_id)
                
                keywords = capability.name.lower().replace('_', ' ').split()
                for keyword in keywords:
                    self.capability_index[keyword].discard(agent_id)
            
            # Clean up
            del self.agents[agent_id]
            if agent_id in self.metrics:
                del self.metrics[agent_id]
            if agent_id in self.load_balancer_state:
                del self.load_balancer_state[agent_id]
    
    def update_agent_status(self, agent_id: str, status: AgentStatus):
        """Update agent status."""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].last_heartbeat = datetime.now()
    
    def update_metrics(
        self,
        agent_id: str,
        response_time: Optional[float] = None,
        success: Optional[bool] = None,
        load: Optional[float] = None,
        capability_used: Optional[str] = None
    ):
        """Update agent performance metrics."""
        if agent_id not in self.metrics:
            self.metrics[agent_id] = DiscoveryMetrics(agent_id=agent_id)
        
        metrics = self.metrics[agent_id]
        
        if response_time is not None:
            # Update average response time using exponential moving average
            if metrics.avg_response_time == 0:
                metrics.avg_response_time = response_time
            else:
                metrics.avg_response_time = 0.8 * metrics.avg_response_time + 0.2 * response_time
        
        if success is not None:
            metrics.total_requests += 1
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
        
        if load is not None:
            metrics.current_load = load
        
        if capability_used:
            metrics.capability_usage[capability_used] = metrics.capability_usage.get(capability_used, 0) + 1
    
    def get_success_rate(self, agent_id: str) -> float:
        """Get agent success rate."""
        if agent_id not in self.metrics:
            return 1.0
        
        metrics = self.metrics[agent_id]
        if metrics.total_requests == 0:
            return 1.0
        
        return metrics.successful_requests / metrics.total_requests
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """Get agents that have a specific capability."""
        capability_lower = capability.lower()
        
        # Direct match
        direct_matches = self.capability_index.get(capability_lower, set())
        
        # Fuzzy matches (partial keyword matching)
        fuzzy_matches = set()
        for indexed_capability, agent_ids in self.capability_index.items():
            if capability_lower in indexed_capability or indexed_capability in capability_lower:
                fuzzy_matches.update(agent_ids)
        
        return list(direct_matches.union(fuzzy_matches))
    
    def get_online_agents(self) -> List[str]:
        """Get all online agents."""
        return [
            agent_id for agent_id, agent_info in self.agents.items()
            if agent_info.status == AgentStatus.ONLINE
        ]
    
    def get_least_loaded_agents(self, agent_ids: List[str], limit: int = 5) -> List[str]:
        """Get least loaded agents from a list."""
        agent_loads = []
        
        for agent_id in agent_ids:
            if agent_id in self.metrics:
                load = self.metrics[agent_id].current_load
            else:
                load = 0.0
            agent_loads.append((agent_id, load))
        
        # Sort by load and return top N
        agent_loads.sort(key=lambda x: x[1])
        return [agent_id for agent_id, _ in agent_loads[:limit]]


class CapabilityMatcher:
    """Advanced capability matching engine."""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.semantic_similarity_cache: Dict[Tuple[str, str], float] = {}
    
    def calculate_capability_score(
        self,
        agent_info: AgentInfo,
        requirement: CapabilityRequirement
    ) -> float:
        """Calculate how well an agent matches a capability requirement."""
        best_score = 0.0
        
        for capability in agent_info.capabilities:
            score = self._score_capability_match(capability, requirement)
            best_score = max(best_score, score)
        
        return best_score
    
    def _score_capability_match(
        self,
        capability: AgentCapability,
        requirement: CapabilityRequirement
    ) -> float:
        """Score a single capability against a requirement."""
        # Exact name match
        if capability.name.lower() == requirement.name.lower():
            return 1.0
        
        # Keyword matching
        cap_keywords = set(capability.name.lower().replace('_', ' ').split())
        req_keywords = set(requirement.name.lower().replace('_', ' ').split())
        
        if cap_keywords.intersection(req_keywords):
            overlap = len(cap_keywords.intersection(req_keywords))
            total = len(cap_keywords.union(req_keywords))
            keyword_score = overlap / total
        else:
            keyword_score = 0.0
        
        # Semantic similarity (simplified - could use embeddings)
        semantic_score = self._calculate_semantic_similarity(
            capability.name, requirement.name
        )
        
        # Combined score
        return max(keyword_score, semantic_score * 0.8)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Simple implementation - could be enhanced with embeddings
        key = (text1.lower(), text2.lower())
        if key in self.semantic_similarity_cache:
            return self.semantic_similarity_cache[key]
        
        # Simple word overlap for now
        words1 = set(text1.lower().replace('_', ' ').split())
        words2 = set(text2.lower().replace('_', ' ').split())
        
        if not words1 or not words2:
            similarity = 0.0
        else:
            similarity = len(words1.intersection(words2)) / len(words1.union(words2))
        
        self.semantic_similarity_cache[key] = similarity
        return similarity
    
    def match_agents(
        self,
        requirements: TaskRequirements,
        algorithm: MatchingAlgorithm = MatchingAlgorithm.CAPABILITY_SCORE,
        limit: int = 5
    ) -> List[AgentMatch]:
        """Find and rank agents matching the requirements."""
        
        # Get candidate agents
        candidates = self._get_candidate_agents(requirements)
        
        # Score each candidate
        matches = []
        for agent_id in candidates:
            if agent_id not in self.registry.agents:
                continue
                
            agent_info = self.registry.agents[agent_id]
            match = self._score_agent_match(agent_info, requirements, algorithm)
            
            if match.total_score >= requirements.capabilities[0].min_score:
                matches.append(match)
        
        # Sort by total score and return top matches
        matches.sort(key=lambda x: x.total_score, reverse=True)
        return matches[:limit]
    
    def _get_candidate_agents(self, requirements: TaskRequirements) -> Set[str]:
        """Get initial set of candidate agents."""
        candidates = set()
        
        # Find agents with required capabilities
        for req in requirements.capabilities:
            if req.required:
                matching_agents = self.registry.get_agents_by_capability(req.name)
                if not candidates:
                    candidates = set(matching_agents)
                else:
                    candidates = candidates.intersection(set(matching_agents))
        
        # Add agents with optional capabilities
        for req in requirements.capabilities:
            if not req.required:
                matching_agents = self.registry.get_agents_by_capability(req.name)
                candidates.update(matching_agents)
        
        # Filter by status and constraints
        online_agents = set(self.registry.get_online_agents())
        candidates = candidates.intersection(online_agents)
        
        # Apply exclusions and preferences
        candidates = candidates - requirements.exclude_agents
        
        return candidates
    
    def _score_agent_match(
        self,
        agent_info: AgentInfo,
        requirements: TaskRequirements,
        algorithm: MatchingAlgorithm
    ) -> AgentMatch:
        """Score how well an agent matches the requirements."""
        
        # Calculate capability scores
        capability_scores = {}
        total_capability_score = 0.0
        total_weight = 0.0
        
        for req in requirements.capabilities:
            score = self.calculate_capability_score(agent_info, req)
            capability_scores[req.name] = score
            total_capability_score += score * req.weight
            total_weight += req.weight
        
        capability_score = total_capability_score / total_weight if total_weight > 0 else 0.0
        
        # Calculate load score
        load_score = self._calculate_load_score(agent_info.id)
        
        # Calculate priority score
        priority_score = self._calculate_priority_score(agent_info.priority, requirements.priority)
        
        # Calculate response time score
        response_time_score = self._calculate_response_time_score(
            agent_info.id, requirements.max_response_time
        )
        
        # Calculate total score based on algorithm
        if algorithm == MatchingAlgorithm.CAPABILITY_SCORE:
            total_score = (
                capability_score * 0.5 +
                load_score * 0.2 +
                priority_score * 0.15 +
                response_time_score * 0.15
            )
        elif algorithm == MatchingAlgorithm.LOAD_BALANCED:
            total_score = (
                capability_score * 0.3 +
                load_score * 0.4 +
                priority_score * 0.15 +
                response_time_score * 0.15
            )
        else:  # Default to capability score
            total_score = capability_score
        
        # Apply preference bonus
        if agent_info.id in requirements.prefer_agents:
            total_score *= 1.2
        
        # Generate reasoning
        reasoning = [
            f"Capability match: {capability_score:.2f}",
            f"Load score: {load_score:.2f}",
            f"Priority alignment: {priority_score:.2f}",
            f"Response time: {response_time_score:.2f}"
        ]
        
        return AgentMatch(
            agent_info=agent_info,
            match_score=capability_score,
            capability_scores=capability_scores,
            load_score=load_score,
            priority_score=priority_score,
            response_time_score=response_time_score,
            total_score=total_score,
            reasoning=reasoning
        )
    
    def _calculate_load_score(self, agent_id: str) -> float:
        """Calculate load-based score (higher is better)."""
        if agent_id not in self.registry.metrics:
            return 1.0
        
        load = self.registry.metrics[agent_id].current_load
        return max(0.0, 1.0 - load)  # Invert load so lower load = higher score
    
    def _calculate_priority_score(self, agent_priority: Priority, task_priority: Priority) -> float:
        """Calculate priority alignment score."""
        priority_values = {
            Priority.LOW: 1,
            Priority.MEDIUM: 2,
            Priority.HIGH: 3,
            Priority.CRITICAL: 4,
            Priority.URGENT: 5
        }
        
        agent_val = priority_values[agent_priority]
        task_val = priority_values[task_priority]
        
        # Perfect match gets 1.0, decreasing as priorities diverge
        diff = abs(agent_val - task_val)
        return max(0.0, 1.0 - diff * 0.2)
    
    def _calculate_response_time_score(self, agent_id: str, max_response_time: Optional[float]) -> float:
        """Calculate response time score."""
        if not max_response_time or agent_id not in self.registry.metrics:
            return 1.0
        
        avg_response_time = self.registry.metrics[agent_id].avg_response_time
        if avg_response_time == 0:
            return 1.0
        
        if avg_response_time <= max_response_time:
            return 1.0
        else:
            # Penalize agents that exceed max response time
            return max(0.0, max_response_time / avg_response_time)


class LoadBalancer:
    """Intelligent load balancer for agent selection."""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
    
    def select_agent(
        self,
        candidate_agents: List[str],
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED
    ) -> Optional[str]:
        """Select an agent using the specified load balancing strategy."""
        
        if not candidate_agents:
            return None
        
        # Filter to online agents only
        online_agents = [
            agent_id for agent_id in candidate_agents
            if agent_id in self.registry.agents and
            self.registry.agents[agent_id].status == AgentStatus.ONLINE
        ]
        
        if not online_agents:
            return None
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(online_agents)
        elif strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_select(online_agents)
        elif strategy == LoadBalancingStrategy.RESPONSE_TIME:
            return self._response_time_select(online_agents)
        elif strategy == LoadBalancingStrategy.PRIORITY_BASED:
            return self._priority_based_select(online_agents)
        else:  # RANDOM
            import random
            return random.choice(online_agents)
    
    def _round_robin_select(self, agents: List[str]) -> str:
        """Round-robin selection."""
        if not agents:
            return None
        
        # Use a global counter for all agents combined
        key = "global"
        current_index = self.registry.load_balancer_state[key] % len(agents)
        self.registry.load_balancer_state[key] += 1
        
        return agents[current_index]
    
    def _least_loaded_select(self, agents: List[str]) -> str:
        """Select the least loaded agent."""
        if not agents:
            return None
        
        min_load = float('inf')
        selected_agent = agents[0]
        
        for agent_id in agents:
            load = 0.0
            if agent_id in self.registry.metrics:
                load = self.registry.metrics[agent_id].current_load
            
            if load < min_load:
                min_load = load
                selected_agent = agent_id
        
        return selected_agent
    
    def _response_time_select(self, agents: List[str]) -> str:
        """Select agent with best response time."""
        if not agents:
            return None
        
        min_response_time = float('inf')
        selected_agent = agents[0]
        
        for agent_id in agents:
            response_time = float('inf')
            if agent_id in self.registry.metrics:
                response_time = self.registry.metrics[agent_id].avg_response_time
                if response_time == 0:
                    response_time = 1.0  # Default for new agents
            
            if response_time < min_response_time:
                min_response_time = response_time
                selected_agent = agent_id
        
        return selected_agent
    
    def _priority_based_select(self, agents: List[str]) -> str:
        """Select highest priority agent."""
        if not agents:
            return None
        
        priority_values = {
            Priority.LOW: 1,
            Priority.MEDIUM: 2,
            Priority.HIGH: 3,
            Priority.CRITICAL: 4,
            Priority.URGENT: 5
        }
        
        max_priority = 0
        selected_agent = agents[0]
        
        for agent_id in agents:
            if agent_id in self.registry.agents:
                priority = priority_values[self.registry.agents[agent_id].priority]
                if priority > max_priority:
                    max_priority = priority
                    selected_agent = agent_id
        
        return selected_agent


class DiscoveryService:
    """Main discovery service orchestrating all components."""
    
    def __init__(self, universal_client: Optional[UniversalAgentClient] = None):
        self.registry = AgentRegistry()
        self.matcher = CapabilityMatcher(self.registry)
        self.load_balancer = LoadBalancer(self.registry)
        self.universal_client = universal_client
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_update_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the discovery service."""
        logger.info("Starting Discovery Service...")
        
        # Initialize agents from universal client if available
        if self.universal_client:
            await self._sync_with_universal_client()
        
        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._metrics_update_task = asyncio.create_task(self._metrics_update_loop())
        
        logger.info("Discovery Service started successfully")
    
    async def stop(self):
        """Stop the discovery service."""
        logger.info("Stopping Discovery Service...")
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._metrics_update_task:
            self._metrics_update_task.cancel()
        
        logger.info("Discovery Service stopped")
    
    async def _sync_with_universal_client(self):
        """Sync agents from universal client."""
        agents = self.universal_client.list_agents()
        for agent_info in agents:
            self.registry.register_agent(agent_info)
        
        logger.info(f"Synced {len(agents)} agents from universal client")
    
    async def _health_check_loop(self):
        """Background health checking."""
        while True:
            try:
                if self.universal_client:
                    health_results = await self.universal_client.health_check_all()
                    
                    for agent_id, is_healthy in health_results.items():
                        status = AgentStatus.ONLINE if is_healthy else AgentStatus.OFFLINE
                        self.registry.update_agent_status(agent_id, status)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {str(e)}")
                await asyncio.sleep(10)
    
    async def _metrics_update_loop(self):
        """Background metrics updating."""
        while True:
            try:
                # Update load metrics for all agents
                # This could be enhanced to collect real metrics from agents
                for agent_id in self.registry.agents:
                    # Simulate load updates - replace with real metrics collection
                    import random
                    simulated_load = random.uniform(0.0, 1.0)
                    self.registry.update_metrics(agent_id, load=simulated_load)
                
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics update loop error: {str(e)}")
                await asyncio.sleep(30)
    
    def find_best_agent(
        self,
        task_description: str,
        capabilities: List[str],
        priority: Priority = Priority.MEDIUM,
        algorithm: MatchingAlgorithm = MatchingAlgorithm.CAPABILITY_SCORE,
        load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED
    ) -> Optional[AgentMatch]:
        """Find the best agent for a task."""
        
        # Convert capabilities to requirements
        requirements = TaskRequirements(
            capabilities=[
                CapabilityRequirement(name=cap, required=True)
                for cap in capabilities
            ],
            priority=priority
        )
        
        # Get matches
        matches = self.matcher.match_agents(requirements, algorithm, limit=10)
        
        if not matches:
            return None
        
        # Apply load balancing among top matches
        top_agents = [match.agent_info.id for match in matches[:5]]
        selected_agent_id = self.load_balancer.select_agent(top_agents, load_balancing)
        
        if not selected_agent_id:
            return None
        
        # Return the match for the selected agent
        for match in matches:
            if match.agent_info.id == selected_agent_id:
                return match
        
        return matches[0]  # Fallback
    
    def find_agents_by_requirements(
        self,
        requirements: TaskRequirements,
        algorithm: MatchingAlgorithm = MatchingAlgorithm.CAPABILITY_SCORE,
        limit: int = 5
    ) -> List[AgentMatch]:
        """Find agents matching specific requirements."""
        return self.matcher.match_agents(requirements, algorithm, limit)
    
    def get_agent_recommendations(
        self,
        task_description: str,
        num_recommendations: int = 3
    ) -> List[AgentMatch]:
        """Get agent recommendations for a natural language task description."""
        
        # Simple keyword extraction - could be enhanced with NLP
        keywords = self._extract_keywords(task_description)
        
        # Convert to requirements
        requirements = TaskRequirements(
            capabilities=[
                CapabilityRequirement(name=keyword, required=False, weight=1.0/len(keywords))
                for keyword in keywords
            ]
        )
        
        return self.matcher.match_agents(requirements, limit=num_recommendations)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract capability keywords from task description."""
        # Simple keyword extraction - could use NLP libraries
        common_capabilities = [
            "code", "coding", "development", "testing", "security", "analysis",
            "automation", "deployment", "monitoring", "optimization", "design",
            "documentation", "research", "planning", "coordination", "management"
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for capability in common_capabilities:
            if capability in text_lower:
                found_keywords.append(capability)
        
        return found_keywords if found_keywords else ["general"]
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_agents = len(self.registry.agents)
        online_agents = len(self.registry.get_online_agents())
        
        capability_counts = defaultdict(int)
        for agent_info in self.registry.agents.values():
            for capability in agent_info.capabilities:
                capability_counts[capability.name] += 1
        
        return {
            "total_agents": total_agents,
            "online_agents": online_agents,
            "offline_agents": total_agents - online_agents,
            "capabilities": dict(capability_counts),
            "most_common_capabilities": sorted(
                capability_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Example usage of the Discovery Service."""
        
        # Initialize discovery service
        async with UniversalAgentClient() as client:
            discovery = DiscoveryService(client)
            await discovery.start()
            
            try:
                # Find best agent for code analysis
                best_match = discovery.find_best_agent(
                    task_description="Analyze Python code quality",
                    capabilities=["code_analysis", "quality_assurance"],
                    priority=Priority.HIGH
                )
                
                if best_match:
                    print(f"Best match: {best_match.agent_info.name}")
                    print(f"Score: {best_match.total_score:.2f}")
                    print(f"Reasoning: {best_match.reasoning}")
                
                # Get recommendations for natural language task
                recommendations = discovery.get_agent_recommendations(
                    "I need help with security testing and vulnerability assessment"
                )
                
                print(f"\nRecommendations ({len(recommendations)}):")
                for i, match in enumerate(recommendations, 1):
                    print(f"{i}. {match.agent_info.name} (score: {match.total_score:.2f})")
                
                # Show registry stats
                stats = discovery.get_registry_stats()
                print(f"\nRegistry Stats:")
                print(f"Total agents: {stats['total_agents']}")
                print(f"Online agents: {stats['online_agents']}")
                print(f"Top capabilities: {stats['most_common_capabilities'][:5]}")
                
            finally:
                await discovery.stop()
    
    # Run the example
    asyncio.run(main())