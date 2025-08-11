#!/usr/bin/env python3
"""
SutazAI Autonomous Agent Swarm Coordinator

This module implements self-organizing agent swarms that can:
- Form dynamically based on task requirements
- Coordinate complex multi-agent workflows
- Adapt their organization based on performance
- Communicate through multiple protocols
- Achieve consensus on decisions
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class SwarmRole(Enum):
    LEADER = "leader"
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    MEDIATOR = "mediator"

class CommunicationProtocol(Enum):
    DIRECT = "direct"
    BROADCAST = "broadcast"
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    PUBLISH_SUBSCRIBE = "pub_sub"

class ConsensusAlgorithm(Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    BYZANTINE_FAULT_TOLERANT = "bft"
    RAFT = "raft"
    LEADER_DECISION = "leader_decision"

@dataclass
class SwarmMessage:
    id: str
    sender: str
    recipients: List[str]
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    requires_response: bool = False

@dataclass
class SwarmMember:
    agent_id: str
    role: SwarmRole
    capabilities: List[str]
    reputation_score: float
    workload: float
    availability: bool
    communication_protocols: List[CommunicationProtocol]
    last_active: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class SwarmDecision:
    id: str
    proposal: str
    options: List[str]
    votes: Dict[str, str]  # agent_id -> chosen_option
    consensus_reached: bool
    final_decision: Optional[str]
    confidence_score: float
    timestamp: datetime

class AutonomousSwarmCoordinator:
    """
    Coordinates self-organizing agent swarms for complex task execution.
    """
    
    def __init__(self, swarm_id: str, goal: str, orchestration_engine):
        self.swarm_id = swarm_id
        self.goal = goal
        self.orchestration_engine = orchestration_engine
        
        # Swarm state
        self.members: Dict[str, SwarmMember] = {}
        self.leader_id: Optional[str] = None
        self.formation_strategy = "capability_based"
        self.communication_protocol = CommunicationProtocol.HIERARCHICAL
        self.consensus_algorithm = ConsensusAlgorithm.WEIGHTED_VOTE
        
        # Communication
        self.message_queue: List[SwarmMessage] = []
        self.message_history: List[SwarmMessage] = []
        self.communication_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Decision making
        self.pending_decisions: Dict[str, SwarmDecision] = {}
        self.decision_history: List[SwarmDecision] = []
        
        # Performance tracking
        self.formation_time = datetime.now()
        self.tasks_completed = 0
        self.success_rate = 1.0
        self.adaptation_count = 0
        
        logger.info(f"Initialized autonomous swarm {swarm_id} with goal: {goal}")
    
    async def form_swarm(self, required_capabilities: List[str], max_size: int = 10) -> bool:
        """
        Form a swarm by selecting and recruiting the most suitable agents.
        """
        logger.info(f"Forming swarm for capabilities: {required_capabilities}")
        
        # Analyze capability requirements
        capability_weights = self._analyze_capability_requirements(required_capabilities)
        
        # Score available agents
        agent_scores = {}
        for agent_id, agent in self.orchestration_engine.agents.items():
            if agent.status.name in ['IDLE', 'BUSY'] and agent.current_load < 0.8:
                score = self._calculate_agent_swarm_fit(agent, capability_weights)
                agent_scores[agent_id] = score
        
        # Select top agents
        selected_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)[:max_size]
        
        if not selected_agents:
            logger.error("No suitable agents available for swarm formation")
            return False
        
        # Add agents to swarm
        for agent_id, score in selected_agents:
            await self._recruit_agent(agent_id, score)
        
        # Elect leader
        await self._elect_leader()
        
        # Establish communication topology
        await self._establish_communication_topology()
        
        logger.info(f"Swarm formed with {len(self.members)} members, leader: {self.leader_id}")
        return True
    
    def _analyze_capability_requirements(self, capabilities: List[str]) -> Dict[str, float]:
        """Analyze and weight capability requirements."""
        weights = {}
        for cap in capabilities:
            # Simple weighting - can be enhanced with ML
            if 'security' in cap.lower():
                weights[cap] = 1.5
            elif 'deployment' in cap.lower():
                weights[cap] = 1.3
            elif 'testing' in cap.lower():
                weights[cap] = 1.2
            else:
                weights[cap] = 1.0
        return weights
    
    def _calculate_agent_swarm_fit(self, agent, capability_weights: Dict[str, float]) -> float:
        """Calculate how well an agent fits the swarm requirements."""
        capability_score = 0
        for required_cap, weight in capability_weights.items():
            for agent_cap in agent.capabilities:
                if required_cap.lower() in agent_cap.lower():
                    capability_score += weight
                    break
        
        # Factor in performance and availability
        availability_score = 1.0 - agent.current_load
        performance_score = agent.performance_score
        
        return capability_score * 0.5 + availability_score * 0.3 + performance_score * 0.2
    
    async def _recruit_agent(self, agent_id: str, fit_score: float):
        """Recruit an agent into the swarm."""
        agent = self.orchestration_engine.agents[agent_id]
        
        # Determine role based on capabilities and swarm needs
        role = self._determine_agent_role(agent, fit_score)
        
        member = SwarmMember(
            agent_id=agent_id,
            role=role,
            capabilities=agent.capabilities,
            reputation_score=agent.performance_score,
            workload=agent.current_load,
            availability=True,
            communication_protocols=[self.communication_protocol],
            last_active=datetime.now()
        )
        
        self.members[agent_id] = member
        
        # Send recruitment message
        recruitment_msg = SwarmMessage(
            id=str(uuid.uuid4()),
            sender="swarm_coordinator",
            recipients=[agent_id],
            message_type="recruitment",
            content={
                "swarm_id": self.swarm_id,
                "goal": self.goal,
                "role": role.value,
                "expected_duration": "unknown"
            },
            timestamp=datetime.now(),
            requires_response=True
        )
        
        await self._send_message(recruitment_msg)
        logger.info(f"Recruited agent {agent_id} with role {role.value}")
    
    def _determine_agent_role(self, agent, fit_score: float) -> SwarmRole:
        """Determine the most appropriate role for an agent in the swarm."""
        # Leadership potential based on performance and capabilities
        if (agent.performance_score > 0.8 and 
            fit_score > 0.7 and 
            any('orchestrat' in cap.lower() or 'coordinat' in cap.lower() for cap in agent.capabilities)):
            return SwarmRole.LEADER
        
        # Coordination role for agents with orchestration capabilities
        if any('coordinat' in cap.lower() or 'orchestrat' in cap.lower() for cap in agent.capabilities):
            return SwarmRole.COORDINATOR
        
        # Specialist role for highly specialized agents
        specialist_keywords = ['security', 'deployment', 'testing', 'analysis']
        if sum(1 for keyword in specialist_keywords for cap in agent.capabilities if keyword in cap.lower()) >= 2:
            return SwarmRole.SPECIALIST
        
        # Default to worker role
        return SwarmRole.WORKER
    
    async def _elect_leader(self):
        """Elect a swarm leader using performance-based voting."""
        if not self.members:
            return
        
        # Score potential leaders
        leadership_scores = {}
        for agent_id, member in self.members.items():
            if member.role in [SwarmRole.LEADER, SwarmRole.COORDINATOR]:
                score = (
                    member.reputation_score * 0.4 +
                    (1.0 - member.workload) * 0.3 +
                    len(member.capabilities) * 0.1 +
                    (1.0 if member.role == SwarmRole.LEADER else 0.8) * 0.2
                )
                leadership_scores[agent_id] = score
        
        if leadership_scores:
            self.leader_id = max(leadership_scores.items(), key=lambda x: x[1])[0]
            self.members[self.leader_id].role = SwarmRole.LEADER
            logger.info(f"Elected {self.leader_id} as swarm leader")
        else:
            # Fallback: select highest performing member
            best_member = max(self.members.items(), key=lambda x: x[1].reputation_score)[0]
            self.leader_id = best_member
            self.members[best_member].role = SwarmRole.LEADER
            logger.info(f"Appointed {best_member} as fallback leader")
    
    async def _establish_communication_topology(self):
        """Establish communication patterns between swarm members."""
        if self.communication_protocol == CommunicationProtocol.HIERARCHICAL:
            # Connect all members to leader
            for agent_id in self.members:
                if agent_id != self.leader_id:
                    self.communication_graph[self.leader_id].add(agent_id)
                    self.communication_graph[agent_id].add(self.leader_id)
        
        elif self.communication_protocol == CommunicationProtocol.MESH:
            # Full mesh - everyone connected to everyone
            for agent1 in self.members:
                for agent2 in self.members:
                    if agent1 != agent2:
                        self.communication_graph[agent1].add(agent2)
        
        elif self.communication_protocol == CommunicationProtocol.BROADCAST:
            # No specific topology - messages are broadcast
            pass
        
        logger.info(f"Established {self.communication_protocol.value} communication topology")
    
    async def coordinate_task_execution(self, tasks: List[Any]) -> Dict[str, Any]:
        """
        Coordinate the execution of multiple tasks across swarm members.
        """
        logger.info(f"Coordinating execution of {len(tasks)} tasks")
        
        # Analyze tasks and create execution plan
        execution_plan = await self._create_execution_plan(tasks)
        
        # Distribute tasks to swarm members
        task_assignments = await self._distribute_tasks(execution_plan)
        
        # Monitor execution and adapt as needed
        results = await self._monitor_and_execute(task_assignments)
        
        # Consolidate results
        final_result = await self._consolidate_results(results)
        
        self.tasks_completed += len(tasks)
        return final_result
    
    async def _create_execution_plan(self, tasks: List[Any]) -> Dict[str, Any]:
        """Create an optimal execution plan for the given tasks."""
        plan = {
            'sequential_tasks': [],
            'parallel_tasks': [],
            'dependencies': {},
            'resource_requirements': {},
            'estimated_duration': 0
        }
        
        # Simple planning logic - can be enhanced with AI
        for task in tasks:
            task_complexity = getattr(task, 'complexity', 0.5)
            
            if task_complexity > 0.7:
                # Complex tasks should be handled sequentially
                plan['sequential_tasks'].append(task)
            else:
                # Simple tasks can be parallelized
                plan['parallel_tasks'].append(task)
        
        # Estimate total duration
        sequential_time = sum(getattr(task, 'estimated_duration', 1.0) for task in plan['sequential_tasks'])
        parallel_time = max([getattr(task, 'estimated_duration', 1.0) for task in plan['parallel_tasks']], default=0)
        plan['estimated_duration'] = sequential_time + parallel_time
        
        return plan
    
    async def _distribute_tasks(self, execution_plan: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Distribute tasks optimally across swarm members."""
        assignments = defaultdict(list)
        
        # Assign sequential tasks to most capable members
        sequential_tasks = execution_plan['sequential_tasks']
        for task in sequential_tasks:
            best_agent = await self._select_best_agent_for_task(task)
            if best_agent:
                assignments[best_agent].append(task)
        
        # Distribute parallel tasks across available members
        parallel_tasks = execution_plan['parallel_tasks']
        available_agents = [agent_id for agent_id, member in self.members.items() 
                          if member.availability and member.workload < 0.8]
        
        for i, task in enumerate(parallel_tasks):
            if available_agents:
                agent = available_agents[i % len(available_agents)]
                assignments[agent].append(task)
        
        return dict(assignments)
    
    async def _select_best_agent_for_task(self, task) -> Optional[str]:
        """Select the best agent for a specific task."""
        task_requirements = getattr(task, 'requirements', [])
        
        best_agent = None
        best_score = 0
        
        for agent_id, member in self.members.items():
            if not member.availability or member.workload > 0.9:
                continue
            
            # Calculate fit score
            capability_match = sum(1 for req in task_requirements 
                                 for cap in member.capabilities 
                                 if req.lower() in cap.lower())
            
            workload_factor = 1.0 - member.workload
            reputation_factor = member.reputation_score
            
            score = capability_match * 0.5 + workload_factor * 0.3 + reputation_factor * 0.2
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    async def _monitor_and_execute(self, task_assignments: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Monitor task execution and adapt as needed."""
        results = {}
        execution_tasks = []
        
        # Start execution on all assigned agents
        for agent_id, tasks in task_assignments.items():
            execution_tasks.append(
                self._execute_agent_tasks(agent_id, tasks)
            )
        
        # Wait for all executions to complete
        agent_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results
        for i, (agent_id, _) in enumerate(task_assignments.items()):
            result = agent_results[i]
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_id} execution failed: {result}")
                results[agent_id] = {'status': 'failed', 'error': str(result)}
            else:
                results[agent_id] = result
        
        return results
    
    async def _execute_agent_tasks(self, agent_id: str, tasks: List[Any]) -> Dict[str, Any]:
        """Execute tasks assigned to a specific agent."""
        member = self.members[agent_id]
        member.workload += 0.5  # Increase workload
        
        try:
            # Simulate task execution (in real implementation, this would call the actual agent)
            task_results = []
            for task in tasks:
                # Execute task using orchestration engine
                result = await self.orchestration_engine.execute_task_autonomously(task)
                task_results.append(result)
            
            # Update member performance
            success_count = sum(1 for result in task_results if result.get('status') == 'success')
            success_rate = success_count / len(task_results) if task_results else 1.0
            member.reputation_score = member.reputation_score * 0.9 + success_rate * 0.1
            
            return {
                'agent_id': agent_id,
                'status': 'completed',
                'task_count': len(tasks),
                'success_rate': success_rate,
                'results': task_results
            }
            
        except Exception as e:
            logger.error(f"Task execution failed for agent {agent_id}: {e}")
            return {
                'agent_id': agent_id,
                'status': 'failed',
                'error': str(e)
            }
        finally:
            member.workload = max(0, member.workload - 0.5)  # Decrease workload
            member.last_active = datetime.now()
    
    async def _consolidate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate results from all swarm members."""
        total_tasks = sum(result.get('task_count', 0) for result in results.values())
        successful_tasks = sum(
            result.get('task_count', 0) * result.get('success_rate', 0) 
            for result in results.values()
        )
        
        overall_success_rate = successful_tasks / total_tasks if total_tasks > 0 else 1.0
        
        return {
            'swarm_id': self.swarm_id,
            'total_tasks': total_tasks,
            'success_rate': overall_success_rate,
            'member_count': len(self.members),
            'execution_time': (datetime.now() - self.formation_time).total_seconds(),
            'member_results': results,
            'status': 'completed' if overall_success_rate > 0.8 else 'partial_success'
        }
    
    async def make_consensus_decision(self, proposal: str, options: List[str]) -> str:
        """Make a consensus decision using the configured algorithm."""
        decision_id = str(uuid.uuid4())
        
        decision = SwarmDecision(
            id=decision_id,
            proposal=proposal,
            options=options,
            votes={},
            consensus_reached=False,
            final_decision=None,
            confidence_score=0.0,
            timestamp=datetime.now()
        )
        
        self.pending_decisions[decision_id] = decision
        
        # Send voting request to all members
        voting_msg = SwarmMessage(
            id=str(uuid.uuid4()),
            sender="swarm_coordinator",
            recipients=list(self.members.keys()),
            message_type="voting_request",
            content={
                "decision_id": decision_id,
                "proposal": proposal,
                "options": options,
                "voting_deadline": (datetime.now() + timedelta(minutes=5)).isoformat()
            },
            timestamp=datetime.now(),
            requires_response=True
        )
        
        await self._send_message(voting_msg)
        
        # Wait for votes (simplified - in real implementation would be async)
        await asyncio.sleep(2)  # Simulate voting time
        
        # Simulate votes (in real implementation, would receive from agents)
        for agent_id in self.members:
            if len(options) > 0:
                # Weighted random choice based on agent reputation
                weights = [self.members[agent_id].reputation_score] * len(options)
                chosen_option = np.random.choice(options, p=np.array(weights)/sum(weights))
                decision.votes[agent_id] = chosen_option
        
        # Apply consensus algorithm
        final_decision = await self._apply_consensus_algorithm(decision)
        
        decision.final_decision = final_decision
        decision.consensus_reached = True
        self.decision_history.append(decision)
        del self.pending_decisions[decision_id]
        
        logger.info(f"Consensus decision made: {final_decision}")
        return final_decision
    
    async def _apply_consensus_algorithm(self, decision: SwarmDecision) -> str:
        """Apply the configured consensus algorithm to reach a decision."""
        if self.consensus_algorithm == ConsensusAlgorithm.MAJORITY_VOTE:
            # Simple majority vote
            vote_counts = defaultdict(int)
            for vote in decision.votes.values():
                vote_counts[vote] += 1
            
            if vote_counts:
                return max(vote_counts.items(), key=lambda x: x[1])[0]
        
        elif self.consensus_algorithm == ConsensusAlgorithm.WEIGHTED_VOTE:
            # Weighted by agent reputation
            vote_weights = defaultdict(float)
            for agent_id, vote in decision.votes.items():
                weight = self.members[agent_id].reputation_score
                vote_weights[vote] += weight
            
            if vote_weights:
                return max(vote_weights.items(), key=lambda x: x[1])[0]
        
        elif self.consensus_algorithm == ConsensusAlgorithm.LEADER_DECISION:
            # Leader makes final decision
            if self.leader_id and self.leader_id in decision.votes:
                return decision.votes[self.leader_id]
        
        # Fallback: return first option
        return decision.options[0] if decision.options else "no_decision"
    
    async def _send_message(self, message: SwarmMessage):
        """Send a message through the swarm communication system."""
        self.message_queue.append(message)
        self.message_history.append(message)
        
        # Process message based on communication protocol
        if self.communication_protocol == CommunicationProtocol.BROADCAST:
            # Broadcast to all members
            logger.info(f"Broadcasting message {message.id} to all members")
        elif self.communication_protocol == CommunicationProtocol.HIERARCHICAL:
            # Route through hierarchy
            logger.info(f"Routing message {message.id} through hierarchy")
        elif self.communication_protocol == CommunicationProtocol.MESH:
            # Direct delivery
            logger.info(f"Direct delivery of message {message.id}")
        
        # In real implementation, this would interface with actual agent communication
    
    async def adapt_swarm_organization(self):
        """Adapt swarm organization based on performance feedback."""
        if self.tasks_completed < 5:  # Need some history to adapt
            return
        
        logger.info("Analyzing swarm performance for adaptation...")
        
        # Analyze member performance
        underperforming_members = []
        high_performing_members = []
        
        for agent_id, member in self.members.items():
            if member.reputation_score < 0.6:
                underperforming_members.append(agent_id)
            elif member.reputation_score > 0.9:
                high_performing_members.append(agent_id)
        
        # Remove underperforming members
        for agent_id in underperforming_members:
            await self._remove_member(agent_id)
        
        # Promote high performers
        for agent_id in high_performing_members:
            await self._promote_member(agent_id)
        
        # Consider changing communication protocol if performance is poor
        if self.success_rate < 0.7:
            await self._adapt_communication_protocol()
        
        # Re-elect leader if current leader is underperforming
        if (self.leader_id and 
            self.leader_id in self.members and 
            self.members[self.leader_id].reputation_score < 0.7):
            await self._elect_leader()
        
        self.adaptation_count += 1
        logger.info(f"Swarm adaptation completed (adaptation #{self.adaptation_count})")
    
    async def _remove_member(self, agent_id: str):
        """Remove an underperforming member from the swarm."""
        if agent_id in self.members:
            del self.members[agent_id]
            
            # Remove from communication graph
            self.communication_graph.pop(agent_id, None)
            for connections in self.communication_graph.values():
                connections.discard(agent_id)
            
            logger.info(f"Removed underperforming member {agent_id}")
    
    async def _promote_member(self, agent_id: str):
        """Promote a high-performing member."""
        if agent_id in self.members:
            member = self.members[agent_id]
            
            if member.role == SwarmRole.WORKER:
                member.role = SwarmRole.SPECIALIST
            elif member.role == SwarmRole.SPECIALIST:
                member.role = SwarmRole.COORDINATOR
            
            logger.info(f"Promoted member {agent_id} to {member.role.value}")
    
    async def _adapt_communication_protocol(self):
        """Adapt communication protocol based on performance."""
        current_protocol = self.communication_protocol
        
        # Try different protocols based on current performance
        if self.success_rate < 0.5:
            # Very poor performance, try mesh for better coordination
            self.communication_protocol = CommunicationProtocol.MESH
        elif self.success_rate < 0.7:
            # Moderate performance, try hierarchical
            self.communication_protocol = CommunicationProtocol.HIERARCHICAL
        
        if self.communication_protocol != current_protocol:
            await self._establish_communication_topology()
            logger.info(f"Adapted communication protocol from {current_protocol.value} to {self.communication_protocol.value}")
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status."""
        return {
            'swarm_id': self.swarm_id,
            'goal': self.goal,
            'member_count': len(self.members),
            'leader_id': self.leader_id,
            'formation_time': self.formation_time.isoformat(),
            'tasks_completed': self.tasks_completed,
            'success_rate': self.success_rate,
            'adaptation_count': self.adaptation_count,
            'communication_protocol': self.communication_protocol.value,
            'consensus_algorithm': self.consensus_algorithm.value,
            'pending_decisions': len(self.pending_decisions),
            'message_queue_size': len(self.message_queue),
            'members': {
                agent_id: {
                    'role': member.role.value,
                    'reputation_score': member.reputation_score,
                    'workload': member.workload,
                    'availability': member.availability,
                    'last_active': member.last_active.isoformat()
                }
                for agent_id, member in self.members.items()
            }
        }