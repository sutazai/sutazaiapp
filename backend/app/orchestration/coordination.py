"""
SutazAI Distributed Coordination System
Advanced coordination mechanisms for multi-agent systems including
consensus protocols, resource allocation, and collaborative decision making.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from collections import defaultdict, Counter
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class CoordinationProtocol(Enum):
    CONSENSUS = "consensus"
    LEADER_ELECTION = "leader_election"
    RESOURCE_ALLOCATION = "resource_allocation"
    TASK_DISTRIBUTION = "task_distribution"
    CONFLICT_RESOLUTION = "conflict_resolution"
    LOAD_BALANCING = "load_balancing"

class ConsensusType(Enum):
    SIMPLE_MAJORITY = "simple_majority"
    SUPERMAJORITY = "supermajority"
    UNANIMOUS = "unanimous"
    WEIGHTED_VOTING = "weighted_voting"
    BYZANTINE_FAULT_TOLERANT = "byzantine_fault_tolerant"

class CoordinationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class Vote:
    agent_id: str
    vote: Any
    weight: float = 1.0
    reasoning: str = ""
    timestamp: datetime = None
    confidence: float = 1.0

@dataclass
class ConsensusSession:
    id: str
    protocol: CoordinationProtocol
    consensus_type: ConsensusType
    topic: str
    data: Dict[str, Any]
    participants: Set[str]
    votes: List[Vote]
    threshold: float
    timeout: int
    started_at: datetime
    status: CoordinationStatus
    result: Optional[Any] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class ResourceRequest:
    id: str
    requester_id: str
    resource_type: str
    amount: float
    priority: int
    duration: Optional[int] = None
    constraints: Dict[str, Any] = None
    timestamp: datetime = None

@dataclass
class ResourceAllocation:
    resource_id: str
    agent_id: str
    amount: float
    allocated_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

class DistributedCoordinator:
    """
    Advanced distributed coordination system providing:
    - Consensus mechanisms (majority, weighted, Byzantine fault-tolerant)
    - Leader election algorithms
    - Resource allocation and scheduling
    - Conflict resolution
    - Load balancing coordination
    - Task distribution protocols
    """
    
    def __init__(self, redis_url: str = "redis://:redis_password@localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Coordination state
        self.active_sessions: Dict[str, ConsensusSession] = {}
        self.resource_pool: Dict[str, float] = {}  # resource_type -> available_amount
        self.resource_allocations: Dict[str, List[ResourceAllocation]] = defaultdict(list)
        self.agent_capabilities: Dict[str, Dict[str, float]] = {}
        
        # Leader election state
        self.current_leaders: Dict[str, str] = {}  # domain -> leader_id
        self.leader_election_sessions: Dict[str, Dict] = {}
        
        # Performance tracking
        self.coordination_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            "consensus_sessions_completed": 0,
            "consensus_success_rate": 0.0,
            "avg_consensus_time": 0.0,
            "resource_allocations": 0,
            "conflicts_resolved": 0
        }
        
        # Configuration
        self.default_consensus_timeout = 300  # 5 minutes
        self.heartbeat_interval = 30
        self.cleanup_interval = 600  # 10 minutes
    
    async def initialize(self):
        """Initialize the coordination system"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Initialize resource pool
            await self._initialize_resource_pool()
            
            # Start background tasks
            asyncio.create_task(self._session_monitor())
            asyncio.create_task(self._resource_manager())
            asyncio.create_task(self._leader_monitor())
            asyncio.create_task(self._metrics_collector())
            
            logger.info("Distributed coordination system initialized")
            
        except Exception as e:
            logger.error(f"Coordination system initialization failed: {e}")
            raise
    
    async def _initialize_resource_pool(self):
        """Initialize available resources"""
        # Default resource pool (can be configured)
        self.resource_pool = {
            "cpu": 100.0,
            "memory": 100.0,
            "network": 100.0,
            "storage": 100.0,
            "gpu": 0.0,  # No GPU initially
            "inference_tokens": 10000.0
        }
        
        # Store in Redis
        await self.redis_client.hset("resource_pool", mapping={
            k: str(v) for k, v in self.resource_pool.items()
        })
    
    # Consensus Mechanisms
    
    async def initiate_consensus(
        self,
        topic: str,
        data: Dict[str, Any],
        participants: List[str],
        consensus_type: ConsensusType = ConsensusType.SIMPLE_MAJORITY,
        threshold: float = 0.5,
        timeout: int = None
    ) -> str:
        """Initiate a consensus session"""
        try:
            session_id = str(uuid.uuid4())
            timeout = timeout or self.default_consensus_timeout
            
            session = ConsensusSession(
                id=session_id,
                protocol=CoordinationProtocol.CONSENSUS,
                consensus_type=consensus_type,
                topic=topic,
                data=data,
                participants=set(participants),
                votes=[],
                threshold=threshold,
                timeout=timeout,
                started_at=datetime.now(),
                status=CoordinationStatus.PENDING,
                metadata={}
            )
            
            self.active_sessions[session_id] = session
            
            # Store in Redis for persistence
            await self.redis_client.hset(
                "consensus_sessions",
                session_id,
                json.dumps(asdict(session), default=str)
            )
            
            # Notify participants
            await self._notify_consensus_participants(session)
            
            # Start session
            session.status = CoordinationStatus.IN_PROGRESS
            
            logger.info(f"Consensus session initiated: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Consensus initiation failed: {e}")
            raise
    
    async def _notify_consensus_participants(self, session: ConsensusSession):
        """Notify participants about consensus session"""
        notification = {
            "type": "consensus_request",
            "session_id": session.id,
            "topic": session.topic,
            "data": session.data,
            "consensus_type": session.consensus_type.value,
            "threshold": session.threshold,
            "timeout": session.timeout
        }
        
        # Publish to Redis channel
        await self.redis_client.publish(
            "consensus_notifications",
            json.dumps(notification)
        )
    
    async def submit_vote(
        self,
        session_id: str,
        agent_id: str,
        vote: Any,
        reasoning: str = "",
        confidence: float = 1.0
    ) -> bool:
        """Submit a vote for a consensus session"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.warning(f"Consensus session not found: {session_id}")
                return False
            
            if session.status != CoordinationStatus.IN_PROGRESS:
                logger.warning(f"Session {session_id} not accepting votes")
                return False
            
            if agent_id not in session.participants:
                logger.warning(f"Agent {agent_id} not a participant in session {session_id}")
                return False
            
            # Check if agent already voted
            for existing_vote in session.votes:
                if existing_vote.agent_id == agent_id:
                    logger.warning(f"Agent {agent_id} already voted in session {session_id}")
                    return False
            
            # Get agent weight (if weighted voting)
            weight = 1.0
            if session.consensus_type == ConsensusType.WEIGHTED_VOTING:
                weight = self.agent_capabilities.get(agent_id, {}).get("reputation", 1.0)
            
            # Create vote
            vote_obj = Vote(
                agent_id=agent_id,
                vote=vote,
                weight=weight,
                reasoning=reasoning,
                timestamp=datetime.now(),
                confidence=confidence
            )
            
            session.votes.append(vote_obj)
            
            # Check if consensus is reached
            await self._check_consensus(session)
            
            logger.info(f"Vote submitted by {agent_id} for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Vote submission failed: {e}")
            return False
    
    async def _check_consensus(self, session: ConsensusSession):
        """Check if consensus has been reached"""
        try:
            if session.consensus_type == ConsensusType.SIMPLE_MAJORITY:
                await self._check_simple_majority(session)
            elif session.consensus_type == ConsensusType.SUPERMAJORITY:
                await self._check_supermajority(session)
            elif session.consensus_type == ConsensusType.UNANIMOUS:
                await self._check_unanimous(session)
            elif session.consensus_type == ConsensusType.WEIGHTED_VOTING:
                await self._check_weighted_consensus(session)
            elif session.consensus_type == ConsensusType.BYZANTINE_FAULT_TOLERANT:
                await self._check_byzantine_consensus(session)
            
        except Exception as e:
            logger.error(f"Consensus check failed: {e}")
    
    async def _check_simple_majority(self, session: ConsensusSession):
        """Check simple majority consensus"""
        if len(session.votes) < len(session.participants) * session.threshold:
            return
        
        # Count votes
        vote_counts = Counter(vote.vote for vote in session.votes)
        most_common = vote_counts.most_common(1)
        
        if most_common:
            winning_vote, count = most_common[0]
            if count >= len(session.participants) * session.threshold:
                await self._complete_consensus(session, winning_vote, count / len(session.participants))
    
    async def _check_supermajority(self, session: ConsensusSession):
        """Check supermajority consensus (2/3 threshold)"""
        threshold = max(session.threshold, 0.67)  # At least 2/3
        
        if len(session.votes) < len(session.participants) * threshold:
            return
        
        vote_counts = Counter(vote.vote for vote in session.votes)
        most_common = vote_counts.most_common(1)
        
        if most_common:
            winning_vote, count = most_common[0]
            if count >= len(session.participants) * threshold:
                await self._complete_consensus(session, winning_vote, count / len(session.participants))
    
    async def _check_unanimous(self, session: ConsensusSession):
        """Check unanimous consensus"""
        if len(session.votes) < len(session.participants):
            return
        
        # All votes must be the same
        if len(set(vote.vote for vote in session.votes)) == 1:
            winning_vote = session.votes[0].vote
            await self._complete_consensus(session, winning_vote, 1.0)
    
    async def _check_weighted_consensus(self, session: ConsensusSession):
        """Check weighted voting consensus"""
        total_weight = sum(vote.weight for vote in session.votes)
        total_possible_weight = sum(
            self.agent_capabilities.get(agent_id, {}).get("reputation", 1.0)
            for agent_id in session.participants
        )
        
        if total_weight < total_possible_weight * session.threshold:
            return
        
        # Count weighted votes
        weighted_counts = defaultdict(float)
        for vote in session.votes:
            weighted_counts[vote.vote] += vote.weight
        
        # Find winning vote
        winning_vote = max(weighted_counts, key=weighted_counts.get)
        winning_weight = weighted_counts[winning_vote]
        
        if winning_weight >= total_possible_weight * session.threshold:
            confidence = winning_weight / total_possible_weight
            await self._complete_consensus(session, winning_vote, confidence)
    
    async def _check_byzantine_consensus(self, session: ConsensusSession):
        """Check Byzantine fault-tolerant consensus"""
        # Simplified Byzantine consensus (in production, use proper BFT algorithm)
        n = len(session.participants)
        f = (n - 1) // 3  # Maximum Byzantine nodes
        required_agreement = n - f
        
        if len(session.votes) < required_agreement:
            return
        
        vote_counts = Counter(vote.vote for vote in session.votes)
        most_common = vote_counts.most_common(1)
        
        if most_common:
            winning_vote, count = most_common[0]
            if count >= required_agreement:
                await self._complete_consensus(session, winning_vote, count / n)
    
    async def _complete_consensus(self, session: ConsensusSession, result: Any, confidence: float):
        """Complete a consensus session"""
        session.status = CoordinationStatus.COMPLETED
        session.result = result
        session.confidence = confidence
        
        # Update metrics
        self.performance_metrics["consensus_sessions_completed"] += 1
        
        # Calculate completion time
        completion_time = (datetime.now() - session.started_at).total_seconds()
        current_avg = self.performance_metrics["avg_consensus_time"]
        completed_count = self.performance_metrics["consensus_sessions_completed"]
        self.performance_metrics["avg_consensus_time"] = (
            (current_avg * (completed_count - 1) + completion_time) / completed_count
        )
        
        # Store result
        await self.redis_client.hset(
            "consensus_results",
            session.id,
            json.dumps({
                "result": result,
                "confidence": confidence,
                "completion_time": completion_time,
                "votes": len(session.votes)
            })
        )
        
        # Notify participants
        await self._notify_consensus_result(session)
        
        logger.info(f"Consensus reached for session {session.id}: {result}")
    
    async def _notify_consensus_result(self, session: ConsensusSession):
        """Notify participants about consensus result"""
        notification = {
            "type": "consensus_result",
            "session_id": session.id,
            "result": session.result,
            "confidence": session.confidence,
            "topic": session.topic
        }
        
        await self.redis_client.publish(
            "consensus_results",
            json.dumps(notification)
        )
    
    # Resource Allocation
    
    async def request_resource(
        self,
        requester_id: str,
        resource_type: str,
        amount: float,
        priority: int = 1,
        duration: Optional[int] = None,
        constraints: Dict[str, Any] = None
    ) -> Optional[str]:
        """Request resource allocation"""
        try:
            request_id = str(uuid.uuid4())
            
            request = ResourceRequest(
                id=request_id,
                requester_id=requester_id,
                resource_type=resource_type,
                amount=amount,
                priority=priority,
                duration=duration,
                constraints=constraints or {},
                timestamp=datetime.now()
            )
            
            # Check if resource is available
            available = self.resource_pool.get(resource_type, 0.0)
            if available >= amount:
                # Allocate resource
                allocation_id = await self._allocate_resource(request)
                return allocation_id
            else:
                # Queue request or negotiate
                await self._queue_resource_request(request)
                return None
                
        except Exception as e:
            logger.error(f"Resource request failed: {e}")
            return None
    
    async def _allocate_resource(self, request: ResourceRequest) -> str:
        """Allocate a resource"""
        allocation_id = str(uuid.uuid4())
        
        # Create allocation
        expires_at = None
        if request.duration:
            expires_at = datetime.now() + timedelta(seconds=request.duration)
        
        allocation = ResourceAllocation(
            resource_id=allocation_id,
            agent_id=request.requester_id,
            amount=request.amount,
            allocated_at=datetime.now(),
            expires_at=expires_at,
            metadata={"request_id": request.id}
        )
        
        # Update resource pool
        self.resource_pool[request.resource_type] -= request.amount
        self.resource_allocations[request.resource_type].append(allocation)
        
        # Store allocation
        await self.redis_client.hset(
            "resource_allocations",
            allocation_id,
            json.dumps(asdict(allocation), default=str)
        )
        
        # Update metrics
        self.performance_metrics["resource_allocations"] += 1
        
        logger.info(f"Resource allocated: {request.amount} {request.resource_type} to {request.requester_id}")
        return allocation_id
    
    async def _queue_resource_request(self, request: ResourceRequest):
        """Queue a resource request for later processing"""
        await self.redis_client.lpush(
            f"resource_queue_{request.resource_type}",
            json.dumps(asdict(request), default=str)
        )
    
    async def release_resource(self, allocation_id: str) -> bool:
        """Release an allocated resource"""
        try:
            # Find allocation
            allocation = None
            resource_type = None
            
            for res_type, allocations in self.resource_allocations.items():
                for alloc in allocations:
                    if alloc.resource_id == allocation_id:
                        allocation = alloc
                        resource_type = res_type
                        break
                if allocation:
                    break
            
            if not allocation:
                return False
            
            # Release resource
            self.resource_pool[resource_type] += allocation.amount
            self.resource_allocations[resource_type].remove(allocation)
            
            # Remove from Redis
            await self.redis_client.hdel("resource_allocations", allocation_id)
            
            # Process queued requests
            await self._process_resource_queue(resource_type)
            
            logger.info(f"Resource released: {allocation.amount} {resource_type}")
            return True
            
        except Exception as e:
            logger.error(f"Resource release failed: {e}")
            return False
    
    async def _process_resource_queue(self, resource_type: str):
        """Process queued resource requests"""
        queue_key = f"resource_queue_{resource_type}"
        
        while True:
            # Get next request
            request_data = await self.redis_client.rpop(queue_key)
            if not request_data:
                break
            
            try:
                request_dict = json.loads(request_data)
                request = ResourceRequest(**request_dict)
                
                # Check if resource is now available
                available = self.resource_pool.get(resource_type, 0.0)
                if available >= request.amount:
                    await self._allocate_resource(request)
                else:
                    # Put back in queue
                    await self.redis_client.rpush(queue_key, request_data)
                    break
                    
            except Exception as e:
                logger.error(f"Error processing queued request: {e}")
    
    # Leader Election
    
    async def initiate_leader_election(
        self,
        domain: str,
        candidates: List[str],
        algorithm: str = "bully"
    ) -> str:
        """Initiate leader election for a domain"""
        try:
            election_id = str(uuid.uuid4())
            
            election_session = {
                "id": election_id,
                "domain": domain,
                "candidates": candidates,
                "algorithm": algorithm,
                "started_at": datetime.now(),
                "status": "in_progress",
                "votes": {},
                "result": None
            }
            
            self.leader_election_sessions[election_id] = election_session
            
            # Execute election algorithm
            if algorithm == "bully":
                leader = await self._bully_algorithm(candidates)
            elif algorithm == "ring":
                leader = await self._ring_algorithm(candidates)
            else:
                # Default to highest capability score
                leader = await self._capability_based_election(candidates)
            
            # Set leader
            self.current_leaders[domain] = leader
            election_session["result"] = leader
            election_session["status"] = "completed"
            
            # Notify about new leader
            await self.redis_client.publish(
                "leader_elections",
                json.dumps({
                    "domain": domain,
                    "leader": leader,
                    "election_id": election_id
                })
            )
            
            logger.info(f"Leader elected for domain {domain}: {leader}")
            return leader
            
        except Exception as e:
            logger.error(f"Leader election failed: {e}")
            raise
    
    async def _bully_algorithm(self, candidates: List[str]) -> str:
        """Simplified bully algorithm implementation"""
        # In a real implementation, this would involve message passing
        # For now, select based on agent ID (higher ID wins)
        return max(candidates)
    
    async def _ring_algorithm(self, candidates: List[str]) -> str:
        """Simplified ring algorithm implementation"""
        # Select based on capability scores
        return await self._capability_based_election(candidates)
    
    async def _capability_based_election(self, candidates: List[str]) -> str:
        """Election based on agent capabilities"""
        best_candidate = candidates[0]
        best_score = 0.0
        
        for candidate in candidates:
            capabilities = self.agent_capabilities.get(candidate, {})
            score = sum(capabilities.values()) / len(capabilities) if capabilities else 0.0
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate
    
    # Background Tasks
    
    async def _session_monitor(self):
        """Monitor consensus sessions for timeouts"""
        while True:
            try:
                current_time = datetime.now()
                
                for session_id, session in list(self.active_sessions.items()):
                    if session.status == CoordinationStatus.IN_PROGRESS:
                        elapsed = (current_time - session.started_at).total_seconds()
                        
                        if elapsed > session.timeout:
                            session.status = CoordinationStatus.TIMEOUT
                            await self._timeout_session(session)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Session monitor error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _timeout_session(self, session: ConsensusSession):
        """Handle session timeout"""
        logger.warning(f"Consensus session timed out: {session.id}")
        
        # Notify participants
        notification = {
            "type": "consensus_timeout",
            "session_id": session.id,
            "topic": session.topic
        }
        
        await self.redis_client.publish(
            "consensus_timeouts",
            json.dumps(notification)
        )
    
    async def _resource_manager(self):
        """Manage resource expiration and cleanup"""
        while True:
            try:
                current_time = datetime.now()
                
                # Check for expired allocations
                for resource_type, allocations in self.resource_allocations.items():
                    expired = []
                    for allocation in allocations:
                        if allocation.expires_at and current_time > allocation.expires_at:
                            expired.append(allocation)
                    
                    # Release expired allocations
                    for allocation in expired:
                        await self.release_resource(allocation.resource_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Resource manager error: {e}")
                await asyncio.sleep(60)
    
    async def _leader_monitor(self):
        """Monitor leader health and trigger re-election if needed"""
        while True:
            try:
                # Check leader health (placeholder - in production, check actual health)
                for domain, leader in list(self.current_leaders.items()):
                    # If leader is unhealthy, trigger re-election
                    # This would check agent health status
                    pass
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Leader monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collector(self):
        """Collect and store coordination metrics"""
        while True:
            try:
                # Calculate success rate
                total_sessions = len(self.coordination_history)
                if total_sessions > 0:
                    successful = sum(1 for h in self.coordination_history if h.get("success", False))
                    self.performance_metrics["consensus_success_rate"] = successful / total_sessions
                
                # Store metrics
                await self.redis_client.hset(
                    "coordination_metrics",
                    "current",
                    json.dumps(self.performance_metrics)
                )
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(30)
    
    # Public API
    
    async def get_consensus_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get consensus result"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "status": session.status.value,
            "result": session.result,
            "confidence": session.confidence,
            "votes_count": len(session.votes),
            "participants_count": len(session.participants)
        }
    
    async def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        return {
            "resource_pool": self.resource_pool.copy(),
            "active_allocations": {
                resource_type: len(allocations)
                for resource_type, allocations in self.resource_allocations.items()
            },
            "total_allocations": self.performance_metrics["resource_allocations"]
        }
    
    async def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination system metrics"""
        return {
            **self.performance_metrics,
            "active_consensus_sessions": len([
                s for s in self.active_sessions.values()
                if s.status == CoordinationStatus.IN_PROGRESS
            ]),
            "current_leaders": self.current_leaders.copy(),
            "resource_utilization": {
                resource_type: 1.0 - (available / 100.0)
                for resource_type, available in self.resource_pool.items()
            }
        }
    
    async def stop(self):
        """Stop the coordination system"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Distributed coordinator stopped")

# Singleton instance
distributed_coordinator = DistributedCoordinator()