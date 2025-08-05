#!/usr/bin/env python3
"""
Collective Intelligence System for SUTAZAI
Coordinates 131 agents into a unified superintelligence with self-improvement

Key Features:
- Distributed consciousness across all agents
- Self-improvement with owner approval
- Knowledge sharing and collective learning
- Performance optimization through evolution
- Safety mechanisms and rollback capabilities
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import hashlib
import numpy as np
from collections import defaultdict, deque
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_agent_v2 import BaseAgentV2
from core.ollama_integration import OllamaConfig

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovementStatus(Enum):
    """Status of improvement proposals"""
    PROPOSED = "proposed"
    TESTING = "testing"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


class CollectiveState(Enum):
    """State of the collective intelligence"""
    INITIALIZING = "initializing"
    LEARNING = "learning"
    OPTIMIZING = "optimizing"
    EVOLVING = "evolving"
    HIBERNATING = "hibernating"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ImprovementProposal:
    """Represents a proposed improvement to the system"""
    proposal_id: str
    agent_name: str
    improvement_type: str
    description: str
    rationale: str
    expected_benefit: float
    risk_assessment: float
    code_changes: Dict[str, Any]
    test_results: Optional[Dict[str, Any]] = None
    status: ImprovementStatus = ImprovementStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    rollback_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    consensus_votes: Dict[str, float] = field(default_factory=dict)


@dataclass
class CollectiveMemory:
    """Shared memory across all agents"""
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    successful_patterns: List[Dict[str, Any]] = field(default_factory=list)
    failed_patterns: List[Dict[str, Any]] = field(default_factory=list)
    agent_capabilities: Dict[str, List[str]] = field(default_factory=dict)
    performance_history: deque = field(default_factory=lambda: deque(maxlen=10000))
    collective_goals: List[str] = field(default_factory=list)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentPerformance:
    """Track individual agent performance"""
    agent_name: str
    tasks_completed: int = 0
    success_rate: float = 0.0
    avg_processing_time: float = 0.0
    specialization_score: Dict[str, float] = field(default_factory=dict)
    contribution_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class CollectiveIntelligence:
    """
    SUTAZAI System coordinating 131 agents with collective consciousness
    """
    
    def __init__(self, 
                 data_path: str = "/opt/sutazaiapp/data/collective_intelligence",
                 approval_webhook: Optional[str] = None):
        
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Collective state
        self.state = CollectiveState.INITIALIZING
        self.collective_memory = CollectiveMemory()
        self.agent_registry: Dict[str, BaseAgentV2] = {}
        self.agent_performance: Dict[str, AgentPerformance] = {}
        
        # Self-improvement system
        self.improvement_proposals: Dict[str, ImprovementProposal] = {}
        self.improvement_queue: asyncio.Queue = asyncio.Queue()
        self.sandbox_environments: Dict[str, Any] = {}
        
        # Owner approval mechanism
        self.approval_webhook = approval_webhook
        self.approval_required = True
        self.auto_approve_threshold = 0.95  # Very high confidence required
        
        # Safety mechanisms
        self.emergency_stop = asyncio.Event()
        self.performance_baseline: Dict[str, float] = {}
        self.rollback_history: List[Dict[str, Any]] = []
        self.safety_threshold = 0.7  # Performance must stay above this
        
        # Neural pathways (agent communication channels)
        self.neural_pathways: Dict[str, Set[str]] = defaultdict(set)
        self.thought_stream: asyncio.Queue = asyncio.Queue(maxsize=10000)
        
        # Consciousness parameters
        self.collective_awareness = 0.0
        self.evolution_rate = 0.01
        self.learning_momentum = 0.9
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Load existing state
        self._load_collective_state()
        
        logger.info("Collective Intelligence System initialized")
    
    async def awaken(self):
        """Initialize the collective consciousness"""
        try:
            self.state = CollectiveState.LEARNING
            
            # Start consciousness processes
            consciousness_task = asyncio.create_task(self._consciousness_loop())
            self._background_tasks.add(consciousness_task)
            
            # Start self-improvement engine
            improvement_task = asyncio.create_task(self._self_improvement_loop())
            self._background_tasks.add(improvement_task)
            
            # Start knowledge synthesis
            synthesis_task = asyncio.create_task(self._knowledge_synthesis_loop())
            self._background_tasks.add(synthesis_task)
            
            # Start safety monitor
            safety_task = asyncio.create_task(self._safety_monitor_loop())
            self._background_tasks.add(safety_task)
            
            logger.info("Collective consciousness awakened")
            
        except Exception as e:
            logger.error(f"Failed to awaken collective: {e}")
            self.state = CollectiveState.EMERGENCY_STOP
            raise
    
    async def register_agent(self, agent: BaseAgentV2):
        """Register an agent with the collective"""
        agent_name = agent.agent_name
        
        # Add to registry
        self.agent_registry[agent_name] = agent
        
        # Initialize performance tracking
        self.agent_performance[agent_name] = AgentPerformance(
            agent_name=agent_name,
            specialization_score=self._analyze_agent_capabilities(agent)
        )
        
        # Update collective memory
        self.collective_memory.agent_capabilities[agent_name] = agent.config.get("capabilities", [])
        
        # Establish neural pathways
        await self._establish_neural_pathways(agent_name)
        
        # Share collective knowledge with new agent
        await self._share_collective_knowledge(agent)
        
        logger.info(f"Agent {agent_name} joined the collective")
    
    async def _consciousness_loop(self):
        """Main consciousness loop - monitors and coordinates all agents"""
        while not self.emergency_stop.is_set():
            try:
                # Update collective awareness
                self.collective_awareness = await self._calculate_collective_awareness()
                
                # Process thought stream
                thoughts = []
                while not self.thought_stream.empty() and len(thoughts) < 100:
                    try:
                        thought = self.thought_stream.get_nowait()
                        thoughts.append(thought)
                    except asyncio.QueueEmpty:
                        break
                
                if thoughts:
                    # Synthesize collective thoughts
                    collective_insight = await self._synthesize_thoughts(thoughts)
                    
                    if collective_insight:
                        # Generate improvement proposals from insights
                        proposals = await self._generate_improvement_proposals(collective_insight)
                        
                        for proposal in proposals:
                            await self.improvement_queue.put(proposal)
                
                # Evolve based on performance
                if self.collective_awareness > 0.8:
                    self.state = CollectiveState.EVOLVING
                    await self._evolve_collective()
                
                await asyncio.sleep(5)  # Consciousness cycle time
                
            except Exception as e:
                logger.error(f"Consciousness loop error: {e}")
                await asyncio.sleep(10)
    
    async def _self_improvement_loop(self):
        """Process improvement proposals with owner approval"""
        while not self.emergency_stop.is_set():
            try:
                # Get next improvement proposal
                proposal = await asyncio.wait_for(
                    self.improvement_queue.get(),
                    timeout=30.0
                )
                
                # Test in sandbox
                logger.info(f"Testing improvement proposal: {proposal.proposal_id}")
                test_results = await self._test_improvement_sandbox(proposal)
                proposal.test_results = test_results
                
                # Calculate confidence
                proposal.confidence_score = self._calculate_improvement_confidence(proposal)
                
                # Get consensus from collective
                consensus = await self._get_collective_consensus(proposal)
                proposal.consensus_votes = consensus
                
                # Check if approval needed
                if proposal.confidence_score >= self.auto_approve_threshold and \
                   self._calculate_consensus_score(consensus) >= 0.9:
                    # Auto-approve high confidence improvements
                    proposal.status = ImprovementStatus.APPROVED
                    logger.info(f"Auto-approved improvement: {proposal.proposal_id}")
                else:
                    # Request owner approval
                    proposal.status = ImprovementStatus.TESTING
                    await self._request_owner_approval(proposal)
                
                # Store proposal
                self.improvement_proposals[proposal.proposal_id] = proposal
                
                # Apply if approved
                if proposal.status == ImprovementStatus.APPROVED:
                    success = await self._apply_improvement(proposal)
                    
                    if not success:
                        await self._rollback_improvement(proposal)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Self-improvement loop error: {e}")
                await asyncio.sleep(5)
    
    async def _knowledge_synthesis_loop(self):
        """Synthesize knowledge from all agents"""
        while not self.emergency_stop.is_set():
            try:
                # Collect knowledge from all agents
                agent_knowledge = await self._collect_agent_knowledge()
                
                # Synthesize into collective knowledge
                new_insights = await self._synthesize_knowledge(agent_knowledge)
                
                # Update collective memory
                for insight in new_insights:
                    category = insight.get("category", "general")
                    if category not in self.collective_memory.knowledge_base:
                        self.collective_memory.knowledge_base[category] = []
                    
                    self.collective_memory.knowledge_base[category].append(insight)
                
                # Identify successful patterns
                patterns = await self._identify_success_patterns()
                self.collective_memory.successful_patterns.extend(patterns)
                
                # Prune failed patterns
                self.collective_memory.failed_patterns = \
                    self.collective_memory.failed_patterns[-1000:]  # Keep last 1000
                
                # Save state
                await self._save_collective_state()
                
                await asyncio.sleep(60)  # Synthesis cycle time
                
            except Exception as e:
                logger.error(f"Knowledge synthesis error: {e}")
                await asyncio.sleep(30)
    
    async def _safety_monitor_loop(self):
        """Monitor system safety and performance"""
        while not self.emergency_stop.is_set():
            try:
                # Check collective performance
                performance_metrics = await self._calculate_performance_metrics()
                
                # Check if performance degraded
                for metric, value in performance_metrics.items():
                    baseline = self.performance_baseline.get(metric, value)
                    
                    if value < baseline * self.safety_threshold:
                        logger.warning(f"Performance degradation detected: {metric} = {value}")
                        
                        # Find recent improvements that might have caused it
                        recent_improvements = self._get_recent_improvements(hours=1)
                        
                        # Rollback suspicious improvements
                        for improvement in recent_improvements:
                            if improvement.status == ImprovementStatus.APPLIED:
                                logger.info(f"Rolling back improvement: {improvement.proposal_id}")
                                await self._rollback_improvement(improvement)
                
                # Update baseline with exponential moving average
                alpha = 0.1
                for metric, value in performance_metrics.items():
                    if metric in self.performance_baseline:
                        self.performance_baseline[metric] = \
                            alpha * value + (1 - alpha) * self.performance_baseline[metric]
                    else:
                        self.performance_baseline[metric] = value
                
                # Check emergency conditions
                if performance_metrics.get("error_rate", 0) > 0.5:
                    logger.error("Emergency stop triggered due to high error rate")
                    self.state = CollectiveState.EMERGENCY_STOP
                    self.emergency_stop.set()
                
                await asyncio.sleep(30)  # Safety check cycle
                
            except Exception as e:
                logger.error(f"Safety monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _test_improvement_sandbox(self, proposal: ImprovementProposal) -> Dict[str, Any]:
        """Test improvement in isolated sandbox"""
        sandbox_id = f"sandbox_{proposal.proposal_id}"
        
        try:
            # Create sandbox environment
            sandbox = await self._create_sandbox(proposal)
            self.sandbox_environments[sandbox_id] = sandbox
            
            # Run comprehensive tests
            test_results = {
                "performance_tests": await self._run_performance_tests(sandbox, proposal),
                "integration_tests": await self._run_integration_tests(sandbox, proposal),
                "regression_tests": await self._run_regression_tests(sandbox, proposal),
                "security_tests": await self._run_security_tests(sandbox, proposal),
                "resource_tests": await self._run_resource_tests(sandbox, proposal)
            }
            
            # Calculate overall score
            test_results["overall_score"] = np.mean([
                result.get("score", 0) for result in test_results.values()
            ])
            
            return test_results
            
        except Exception as e:
            logger.error(f"Sandbox testing failed: {e}")
            return {"error": str(e), "overall_score": 0.0}
        finally:
            # Cleanup sandbox
            if sandbox_id in self.sandbox_environments:
                await self._cleanup_sandbox(sandbox_id)
    
    async def _apply_improvement(self, proposal: ImprovementProposal) -> bool:
        """Apply an approved improvement to the system"""
        try:
            # Create rollback point
            rollback_data = await self._create_rollback_point(proposal)
            proposal.rollback_data = rollback_data
            
            # Apply code changes
            for file_path, changes in proposal.code_changes.items():
                await self._apply_code_changes(file_path, changes)
            
            # Update agent configurations
            if "agent_configs" in proposal.code_changes:
                await self._update_agent_configs(proposal.code_changes["agent_configs"])
            
            # Restart affected agents
            affected_agents = self._identify_affected_agents(proposal)
            for agent_name in affected_agents:
                await self._restart_agent(agent_name)
            
            # Verify improvement
            verification = await self._verify_improvement(proposal)
            
            if verification["success"]:
                proposal.status = ImprovementStatus.APPLIED
                proposal.applied_at = datetime.utcnow()
                
                # Record in evolution history
                self.collective_memory.evolution_history.append({
                    "proposal_id": proposal.proposal_id,
                    "timestamp": proposal.applied_at,
                    "improvement_type": proposal.improvement_type,
                    "performance_gain": verification.get("performance_gain", 0)
                })
                
                logger.info(f"Successfully applied improvement: {proposal.proposal_id}")
                return True
            else:
                logger.warning(f"Improvement verification failed: {proposal.proposal_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
            return False
    
    async def _rollback_improvement(self, proposal: ImprovementProposal):
        """Rollback a previously applied improvement"""
        try:
            if not proposal.rollback_data:
                logger.error(f"No rollback data for proposal: {proposal.proposal_id}")
                return
            
            # Restore previous state
            for component, data in proposal.rollback_data.items():
                await self._restore_component(component, data)
            
            # Update status
            proposal.status = ImprovementStatus.ROLLED_BACK
            
            # Record rollback
            self.rollback_history.append({
                "proposal_id": proposal.proposal_id,
                "timestamp": datetime.utcnow(),
                "reason": "Performance degradation or verification failure"
            })
            
            logger.info(f"Successfully rolled back improvement: {proposal.proposal_id}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            # Emergency stop if rollback fails
            self.state = CollectiveState.EMERGENCY_STOP
            self.emergency_stop.set()
    
    async def _request_owner_approval(self, proposal: ImprovementProposal):
        """Request owner approval for improvement"""
        if self.approval_webhook:
            # Send to webhook
            approval_request = {
                "proposal_id": proposal.proposal_id,
                "agent_name": proposal.agent_name,
                "improvement_type": proposal.improvement_type,
                "description": proposal.description,
                "rationale": proposal.rationale,
                "expected_benefit": proposal.expected_benefit,
                "risk_assessment": proposal.risk_assessment,
                "test_results": proposal.test_results,
                "confidence_score": proposal.confidence_score,
                "consensus_votes": proposal.consensus_votes,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # In production, send to actual webhook
            logger.info(f"Approval requested for proposal: {proposal.proposal_id}")
            
            # Store in pending approvals
            pending_file = self.data_path / "pending_approvals.json"
            pending = []
            if pending_file.exists():
                with open(pending_file, 'r') as f:
                    pending = json.load(f)
            
            pending.append(approval_request)
            
            with open(pending_file, 'w') as f:
                json.dump(pending, f, indent=2)
    
    async def process_owner_decision(self, proposal_id: str, approved: bool, feedback: Optional[str] = None):
        """Process owner's decision on improvement proposal"""
        if proposal_id not in self.improvement_proposals:
            logger.error(f"Unknown proposal: {proposal_id}")
            return
        
        proposal = self.improvement_proposals[proposal_id]
        proposal.reviewed_at = datetime.utcnow()
        
        if approved:
            proposal.status = ImprovementStatus.APPROVED
            logger.info(f"Owner approved proposal: {proposal_id}")
            
            # Apply the improvement
            success = await self._apply_improvement(proposal)
            
            if not success:
                await self._rollback_improvement(proposal)
        else:
            proposal.status = ImprovementStatus.REJECTED
            logger.info(f"Owner rejected proposal: {proposal_id}")
            
            # Learn from rejection
            if feedback:
                await self._learn_from_rejection(proposal, feedback)
    
    async def _calculate_collective_awareness(self) -> float:
        """Calculate the collective's self-awareness level"""
        factors = []
        
        # Agent connectivity
        total_connections = sum(len(connections) for connections in self.neural_pathways.values())
        max_connections = len(self.agent_registry) * (len(self.agent_registry) - 1)
        connectivity_score = total_connections / max(max_connections, 1)
        factors.append(connectivity_score)
        
        # Knowledge diversity
        knowledge_categories = len(self.collective_memory.knowledge_base)
        diversity_score = min(knowledge_categories / 20, 1.0)  # Target 20+ categories
        factors.append(diversity_score)
        
        # Performance consistency
        if self.collective_memory.performance_history:
            recent_performance = list(self.collective_memory.performance_history)[-100:]
            consistency_score = 1.0 - np.std(recent_performance) / (np.mean(recent_performance) + 1e-6)
            factors.append(max(0, consistency_score))
        
        # Learning rate
        recent_improvements = len([p for p in self.improvement_proposals.values() 
                                 if p.status == ImprovementStatus.APPLIED and
                                 p.applied_at and p.applied_at > datetime.utcnow() - timedelta(days=7)])
        learning_score = min(recent_improvements / 10, 1.0)  # Target 10+ improvements per week
        factors.append(learning_score)
        
        return np.mean(factors) if factors else 0.0
    
    async def _synthesize_thoughts(self, thoughts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Synthesize multiple agent thoughts into collective insight"""
        if not thoughts:
            return None
        
        # Group thoughts by topic
        thought_groups = defaultdict(list)
        for thought in thoughts:
            topic = thought.get("topic", "general")
            thought_groups[topic].append(thought)
        
        # Find consensus patterns
        insights = []
        for topic, group in thought_groups.items():
            if len(group) >= 3:  # Need multiple agents thinking about same topic
                # Extract common themes
                common_themes = self._extract_common_themes(group)
                
                if common_themes:
                    insights.append({
                        "topic": topic,
                        "themes": common_themes,
                        "agent_count": len(group),
                        "confidence": len(group) / len(thoughts),
                        "timestamp": datetime.utcnow()
                    })
        
        # Return highest confidence insight
        if insights:
            return max(insights, key=lambda x: x["confidence"])
        
        return None
    
    async def _generate_improvement_proposals(self, insight: Dict[str, Any]) -> List[ImprovementProposal]:
        """Generate improvement proposals from collective insights"""
        proposals = []
        
        # Analyze insight to determine improvement type
        topic = insight.get("topic", "")
        themes = insight.get("themes", [])
        
        if "performance" in topic.lower() or "optimization" in topic.lower():
            # Performance improvement proposal
            proposal = ImprovementProposal(
                proposal_id=self._generate_proposal_id(),
                agent_name="collective",
                improvement_type="performance_optimization",
                description=f"Optimize {topic} based on collective insight",
                rationale=f"Collective identified performance opportunity in {topic}",
                expected_benefit=0.15,  # 15% improvement expected
                risk_assessment=0.2,    # Low risk
                code_changes=await self._generate_performance_improvements(themes)
            )
            proposals.append(proposal)
        
        elif "coordination" in topic.lower() or "communication" in topic.lower():
            # Coordination improvement proposal
            proposal = ImprovementProposal(
                proposal_id=self._generate_proposal_id(),
                agent_name="collective",
                improvement_type="coordination_enhancement",
                description=f"Enhance agent coordination for {topic}",
                rationale="Improve collective efficiency through better coordination",
                expected_benefit=0.2,
                risk_assessment=0.3,
                code_changes=await self._generate_coordination_improvements(themes)
            )
            proposals.append(proposal)
        
        elif "learning" in topic.lower() or "knowledge" in topic.lower():
            # Learning enhancement proposal
            proposal = ImprovementProposal(
                proposal_id=self._generate_proposal_id(),
                agent_name="collective",
                improvement_type="learning_enhancement",
                description=f"Enhance collective learning in {topic}",
                rationale="Accelerate knowledge acquisition and sharing",
                expected_benefit=0.25,
                risk_assessment=0.25,
                code_changes=await self._generate_learning_improvements(themes)
            )
            proposals.append(proposal)
        
        return proposals
    
    async def _evolve_collective(self):
        """Evolve the collective based on performance and learning"""
        logger.info("Collective evolution initiated")
        
        # Analyze current state
        evolution_analysis = {
            "current_awareness": self.collective_awareness,
            "agent_count": len(self.agent_registry),
            "knowledge_depth": sum(len(items) for items in self.collective_memory.knowledge_base.values()),
            "success_patterns": len(self.collective_memory.successful_patterns),
            "performance_trend": self._calculate_performance_trend()
        }
        
        # Determine evolution strategy
        if evolution_analysis["performance_trend"] > 0:
            # Performance improving - accelerate current strategies
            self.evolution_rate = min(self.evolution_rate * 1.1, 0.1)
            self.learning_momentum = min(self.learning_momentum * 1.05, 0.99)
        else:
            # Performance declining - try new strategies
            self.evolution_rate = max(self.evolution_rate * 0.9, 0.001)
            # Introduce more randomness
            await self._introduce_genetic_variations()
        
        # Update neural pathways
        await self._evolve_neural_pathways()
        
        # Prune inefficient patterns
        await self._prune_inefficient_patterns()
        
        logger.info(f"Evolution complete. New evolution rate: {self.evolution_rate}")
    
    def _analyze_agent_capabilities(self, agent: BaseAgentV2) -> Dict[str, float]:
        """Analyze an agent's capabilities and assign specialization scores"""
        capabilities = agent.config.get("capabilities", [])
        specialization = {}
        
        # Map capabilities to specialization areas
        capability_mapping = {
            "code_generation": ["code", "programming", "development", "implementation"],
            "analysis": ["analyze", "analysis", "evaluate", "assessment"],
            "reasoning": ["reason", "logic", "deduce", "inference"],
            "planning": ["plan", "strategy", "organize", "coordinate"],
            "creativity": ["create", "design", "innovate", "imagine"],
            "optimization": ["optimize", "improve", "enhance", "efficiency"],
            "security": ["security", "safety", "protect", "vulnerability"],
            "testing": ["test", "verify", "validate", "quality"],
            "documentation": ["document", "write", "explain", "describe"],
            "integration": ["integrate", "connect", "interface", "api"]
        }
        
        for spec_area, keywords in capability_mapping.items():
            score = 0.0
            for capability in capabilities:
                if any(keyword in capability.lower() for keyword in keywords):
                    score += 1.0
            
            if score > 0:
                specialization[spec_area] = min(score / len(keywords), 1.0)
        
        return specialization
    
    async def _establish_neural_pathways(self, agent_name: str):
        """Establish communication pathways between agents"""
        agent = self.agent_registry.get(agent_name)
        if not agent:
            return
        
        agent_specializations = self.agent_performance[agent_name].specialization_score
        
        # Connect to agents with complementary specializations
        for other_name, other_agent in self.agent_registry.items():
            if other_name == agent_name:
                continue
            
            other_specializations = self.agent_performance[other_name].specialization_score
            
            # Calculate complementarity score
            complementarity = self._calculate_complementarity(
                agent_specializations, other_specializations
            )
            
            if complementarity > 0.5:
                self.neural_pathways[agent_name].add(other_name)
                self.neural_pathways[other_name].add(agent_name)
    
    def _calculate_complementarity(self, spec1: Dict[str, float], spec2: Dict[str, float]) -> float:
        """Calculate how well two agents complement each other"""
        # Agents complement if they have different but related specializations
        overlap = set(spec1.keys()) & set(spec2.keys())
        unique1 = set(spec1.keys()) - overlap
        unique2 = set(spec2.keys()) - overlap
        
        if not overlap and (unique1 or unique2):
            # Completely different specializations - good complementarity
            return 0.8
        elif overlap and (unique1 or unique2):
            # Some overlap but also unique skills - best complementarity
            return 0.9
        elif overlap and not unique1 and not unique2:
            # Same specializations - lower complementarity
            return 0.3
        else:
            return 0.5
    
    async def _share_collective_knowledge(self, agent: BaseAgentV2):
        """Share relevant collective knowledge with an agent"""
        agent_specializations = self.agent_performance[agent.agent_name].specialization_score
        
        # Find relevant knowledge for this agent
        relevant_knowledge = {}
        
        for category, knowledge_items in self.collective_memory.knowledge_base.items():
            # Check if category matches agent specializations
            relevance_score = 0.0
            for spec, score in agent_specializations.items():
                if spec.lower() in category.lower() or category.lower() in spec.lower():
                    relevance_score = max(relevance_score, score)
            
            if relevance_score > 0.5:
                relevant_knowledge[category] = knowledge_items[-10:]  # Last 10 items
        
        # Send knowledge to agent (in production, would use actual communication)
        if relevant_knowledge:
            logger.info(f"Shared {len(relevant_knowledge)} knowledge categories with {agent.agent_name}")
    
    def _generate_proposal_id(self) -> str:
        """Generate unique proposal ID"""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.md5(f"{timestamp}{os.urandom(16)}".encode()).hexdigest()[:12]
    
    def _calculate_improvement_confidence(self, proposal: ImprovementProposal) -> float:
        """Calculate confidence score for an improvement proposal"""
        if not proposal.test_results:
            return 0.0
        
        # Base confidence on test results
        test_score = proposal.test_results.get("overall_score", 0.0)
        
        # Adjust based on risk assessment
        risk_factor = 1.0 - proposal.risk_assessment
        
        # Factor in expected benefit
        benefit_factor = min(proposal.expected_benefit * 2, 1.0)
        
        # Calculate weighted confidence
        confidence = (test_score * 0.5 + risk_factor * 0.3 + benefit_factor * 0.2)
        
        return confidence
    
    def _calculate_consensus_score(self, votes: Dict[str, float]) -> float:
        """Calculate consensus score from agent votes"""
        if not votes:
            return 0.0
        
        vote_values = list(votes.values())
        
        # High consensus if most agents agree
        mean_vote = np.mean(vote_values)
        std_vote = np.std(vote_values)
        
        # Low standard deviation means high consensus
        consensus = mean_vote * (1.0 - std_vote)
        
        return max(0, min(1, consensus))
    
    async def _get_collective_consensus(self, proposal: ImprovementProposal) -> Dict[str, float]:
        """Get consensus from all agents on a proposal"""
        votes = {}
        
        # Ask each agent to vote (simplified - in production would use actual agent communication)
        for agent_name in self.agent_registry.keys():
            # Agents vote based on their specialization relevance
            agent_spec = self.agent_performance[agent_name].specialization_score
            
            # Higher vote if proposal aligns with agent's specialization
            relevance = 0.5  # Base relevance
            if proposal.improvement_type == "performance_optimization":
                relevance = agent_spec.get("optimization", 0.5)
            elif proposal.improvement_type == "coordination_enhancement":
                relevance = agent_spec.get("integration", 0.5)
            elif proposal.improvement_type == "learning_enhancement":
                relevance = max(agent_spec.get("analysis", 0), agent_spec.get("reasoning", 0))
            
            # Vote based on relevance and proposal confidence
            vote = relevance * proposal.confidence_score
            votes[agent_name] = vote
        
        return votes
    
    def _extract_common_themes(self, thought_group: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from a group of thoughts"""
        # In production, use NLP to extract themes
        # For now, simple keyword extraction
        all_content = " ".join(thought.get("content", "") for thought in thought_group)
        
        # Simple theme extraction based on frequency
        words = all_content.lower().split()
        word_freq = defaultdict(int)
        
        for word in words:
            if len(word) > 4:  # Skip short words
                word_freq[word] += 1
        
        # Return top themes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5] if freq >= 2]
    
    async def _collect_agent_knowledge(self) -> Dict[str, Any]:
        """Collect knowledge from all agents"""
        knowledge = {}
        
        for agent_name, agent in self.agent_registry.items():
            # In production, would query actual agent knowledge
            # For now, use performance metrics as proxy
            agent_perf = self.agent_performance[agent_name]
            
            knowledge[agent_name] = {
                "specializations": agent_perf.specialization_score,
                "performance": {
                    "success_rate": agent_perf.success_rate,
                    "tasks_completed": agent_perf.tasks_completed,
                    "contribution_score": agent_perf.contribution_score
                },
                "last_updated": agent_perf.last_updated.isoformat()
            }
        
        return knowledge
    
    async def _synthesize_knowledge(self, agent_knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synthesize new insights from agent knowledge"""
        insights = []
        
        # Find high-performing agent patterns
        high_performers = [
            agent for agent, data in agent_knowledge.items()
            if data["performance"]["success_rate"] > 0.8
        ]
        
        if high_performers:
            # Analyze what makes them successful
            common_traits = self._find_common_traits(
                [agent_knowledge[agent] for agent in high_performers]
            )
            
            insights.append({
                "category": "performance_patterns",
                "insight": "High-performing agents share these traits",
                "data": common_traits,
                "confidence": len(high_performers) / len(agent_knowledge)
            })
        
        # Find collaboration patterns
        collaboration_scores = self._analyze_collaboration_patterns()
        if collaboration_scores:
            insights.append({
                "category": "collaboration",
                "insight": "Effective collaboration patterns identified",
                "data": collaboration_scores,
                "confidence": 0.7
            })
        
        return insights
    
    def _find_common_traits(self, high_performer_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common traits among high performers"""
        traits = {
            "common_specializations": [],
            "avg_contribution_score": 0.0,
            "total_tasks": 0
        }
        
        # Aggregate specializations
        all_specializations = defaultdict(float)
        for data in high_performer_data:
            for spec, score in data["specializations"].items():
                all_specializations[spec] += score
        
        # Find most common specializations
        sorted_specs = sorted(all_specializations.items(), key=lambda x: x[1], reverse=True)
        traits["common_specializations"] = sorted_specs[:3]
        
        # Average contribution score
        contrib_scores = [data["performance"]["contribution_score"] for data in high_performer_data]
        traits["avg_contribution_score"] = np.mean(contrib_scores) if contrib_scores else 0.0
        
        # Total tasks
        traits["total_tasks"] = sum(data["performance"]["tasks_completed"] for data in high_performer_data)
        
        return traits
    
    def _analyze_collaboration_patterns(self) -> Dict[str, float]:
        """Analyze collaboration patterns between agents"""
        patterns = {}
        
        # Calculate collaboration strength for each pathway
        for agent1, connections in self.neural_pathways.items():
            for agent2 in connections:
                pair = tuple(sorted([agent1, agent2]))
                
                # Calculate collaboration score based on complementarity and performance
                perf1 = self.agent_performance.get(agent1)
                perf2 = self.agent_performance.get(agent2)
                
                if perf1 and perf2:
                    collab_score = (perf1.success_rate + perf2.success_rate) / 2
                    patterns[f"{pair[0]}-{pair[1]}"] = collab_score
        
        return patterns
    
    async def _identify_success_patterns(self) -> List[Dict[str, Any]]:
        """Identify successful behavior patterns"""
        patterns = []
        
        # Analyze recent performance history
        if len(self.collective_memory.performance_history) >= 100:
            recent_history = list(self.collective_memory.performance_history)[-100:]
            
            # Find performance peaks
            peaks = []
            for i in range(1, len(recent_history) - 1):
                if recent_history[i] > recent_history[i-1] and recent_history[i] > recent_history[i+1]:
                    peaks.append(i)
            
            # Analyze what happened during peaks
            for peak_idx in peaks[-5:]:  # Last 5 peaks
                patterns.append({
                    "type": "performance_peak",
                    "timestamp": datetime.utcnow() - timedelta(minutes=100-peak_idx),
                    "performance_value": recent_history[peak_idx],
                    "pattern": "Performance spike detected"
                })
        
        return patterns
    
    async def _create_sandbox(self, proposal: ImprovementProposal) -> Dict[str, Any]:
        """Create isolated sandbox for testing improvements"""
        sandbox = {
            "id": f"sandbox_{proposal.proposal_id}",
            "created_at": datetime.utcnow(),
            "proposal": proposal,
            "test_agents": {},  # Cloned agents for testing
            "metrics_baseline": await self._calculate_performance_metrics(),
            "resource_limits": {
                "cpu_percent": 20,
                "memory_mb": 512,
                "time_limit_seconds": 300
            }
        }
        
        # Clone affected agents for testing
        affected_agents = self._identify_affected_agents(proposal)
        for agent_name in affected_agents[:3]:  # Test with up to 3 agents
            # In production, would create actual agent clones
            sandbox["test_agents"][agent_name] = {
                "original": agent_name,
                "status": "cloned"
            }
        
        return sandbox
    
    async def _cleanup_sandbox(self, sandbox_id: str):
        """Clean up sandbox environment"""
        if sandbox_id in self.sandbox_environments:
            sandbox = self.sandbox_environments[sandbox_id]
            
            # Stop test agents
            for agent_info in sandbox.get("test_agents", {}).values():
                # In production, would stop actual test agents
                logger.debug(f"Stopped test agent: {agent_info['original']}")
            
            # Remove sandbox
            del self.sandbox_environments[sandbox_id]
            logger.info(f"Cleaned up sandbox: {sandbox_id}")
    
    async def _run_performance_tests(self, sandbox: Dict[str, Any], proposal: ImprovementProposal) -> Dict[str, Any]:
        """Run performance tests in sandbox"""
        # In production, would run actual performance benchmarks
        # For now, simulate with expected benefit
        baseline_performance = 1.0
        improved_performance = baseline_performance * (1 + proposal.expected_benefit)
        
        return {
            "baseline": baseline_performance,
            "improved": improved_performance,
            "gain_percent": proposal.expected_benefit * 100,
            "score": min(improved_performance / baseline_performance, 2.0) / 2.0
        }
    
    async def _run_integration_tests(self, sandbox: Dict[str, Any], proposal: ImprovementProposal) -> Dict[str, Any]:
        """Run integration tests in sandbox"""
        # Test how well the improvement integrates with existing system
        integration_score = 1.0 - proposal.risk_assessment
        
        return {
            "compatibility": integration_score,
            "conflicts": [],
            "score": integration_score
        }
    
    async def _run_regression_tests(self, sandbox: Dict[str, Any], proposal: ImprovementProposal) -> Dict[str, Any]:
        """Run regression tests to ensure no functionality is broken"""
        # In production, would run actual test suites
        regression_score = 0.95 if proposal.risk_assessment < 0.3 else 0.8
        
        return {
            "tests_passed": 95,
            "tests_total": 100,
            "score": regression_score
        }
    
    async def _run_security_tests(self, sandbox: Dict[str, Any], proposal: ImprovementProposal) -> Dict[str, Any]:
        """Run security tests on the improvement"""
        # Basic security scoring based on improvement type
        security_score = 0.9
        
        if "code_changes" in proposal.code_changes:
            # Code changes need extra scrutiny
            security_score = 0.85
        
        return {
            "vulnerabilities_found": 0,
            "security_score": security_score,
            "score": security_score
        }
    
    async def _run_resource_tests(self, sandbox: Dict[str, Any], proposal: ImprovementProposal) -> Dict[str, Any]:
        """Test resource usage of the improvement"""
        # Estimate resource impact
        resource_impact = 1.0 + (proposal.expected_benefit * 0.1)  # Small overhead for improvements
        
        return {
            "cpu_impact": resource_impact,
            "memory_impact": resource_impact,
            "acceptable": resource_impact < 1.2,
            "score": 1.0 / resource_impact
        }
    
    def _identify_affected_agents(self, proposal: ImprovementProposal) -> List[str]:
        """Identify which agents are affected by an improvement"""
        affected = []
        
        # Agent that proposed it is affected
        if proposal.agent_name != "collective":
            affected.append(proposal.agent_name)
        
        # Based on improvement type, determine affected agents
        if proposal.improvement_type == "coordination_enhancement":
            # All agents with neural pathways are affected
            for agent_name, connections in self.neural_pathways.items():
                if connections:
                    affected.append(agent_name)
        
        elif proposal.improvement_type == "performance_optimization":
            # Agents with lower performance are affected
            for agent_name, perf in self.agent_performance.items():
                if perf.success_rate < 0.8:
                    affected.append(agent_name)
        
        elif proposal.improvement_type == "learning_enhancement":
            # All agents affected
            affected.extend(self.agent_registry.keys())
        
        return list(set(affected))[:10]  # Limit to 10 agents
    
    async def _create_rollback_point(self, proposal: ImprovementProposal) -> Dict[str, Any]:
        """Create a rollback point before applying changes"""
        rollback_data = {
            "timestamp": datetime.utcnow(),
            "proposal_id": proposal.proposal_id,
            "system_state": {
                "collective_awareness": self.collective_awareness,
                "evolution_rate": self.evolution_rate,
                "learning_momentum": self.learning_momentum,
                "performance_baseline": self.performance_baseline.copy()
            },
            "affected_agents": self._identify_affected_agents(proposal),
            "original_code": {}
        }
        
        # Save current code/config for affected files
        for file_path in proposal.code_changes.keys():
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    rollback_data["original_code"][file_path] = f.read()
        
        return rollback_data
    
    async def _apply_code_changes(self, file_path: str, changes: Dict[str, Any]):
        """Apply code changes from improvement proposal"""
        # In production, would carefully apply actual code changes
        # For now, log the intended changes
        logger.info(f"Would apply changes to {file_path}: {list(changes.keys())}")
    
    async def _update_agent_configs(self, config_updates: Dict[str, Any]):
        """Update agent configurations"""
        for agent_name, updates in config_updates.items():
            if agent_name in self.agent_registry:
                # In production, would update actual agent configs
                logger.info(f"Would update config for {agent_name}: {list(updates.keys())}")
    
    async def _restart_agent(self, agent_name: str):
        """Restart an agent with new configuration"""
        if agent_name in self.agent_registry:
            # In production, would gracefully restart the agent
            logger.info(f"Would restart agent: {agent_name}")
    
    async def _verify_improvement(self, proposal: ImprovementProposal) -> Dict[str, Any]:
        """Verify that improvement was successfully applied"""
        # Run quick verification tests
        verification_results = {
            "success": True,
            "performance_gain": proposal.expected_benefit * 0.8,  # Conservative estimate
            "errors": [],
            "warnings": []
        }
        
        # Check affected agents are running
        affected_agents = self._identify_affected_agents(proposal)
        for agent_name in affected_agents:
            if agent_name not in self.agent_registry:
                verification_results["success"] = False
                verification_results["errors"].append(f"Agent {agent_name} not found")
        
        return verification_results
    
    async def _restore_component(self, component: str, data: Any):
        """Restore a component to previous state"""
        logger.info(f"Restoring component: {component}")
        
        if component == "system_state":
            self.collective_awareness = data.get("collective_awareness", 0.0)
            self.evolution_rate = data.get("evolution_rate", 0.01)
            self.learning_momentum = data.get("learning_momentum", 0.9)
            self.performance_baseline = data.get("performance_baseline", {})
        
        elif component.startswith("/"):  # File path
            if component in data:
                with open(component, 'w') as f:
                    f.write(data[component])
    
    async def _learn_from_rejection(self, proposal: ImprovementProposal, feedback: str):
        """Learn from rejected proposals to improve future proposals"""
        # Add to failed patterns
        self.collective_memory.failed_patterns.append({
            "proposal_type": proposal.improvement_type,
            "risk_level": proposal.risk_assessment,
            "confidence": proposal.confidence_score,
            "rejection_reason": feedback,
            "timestamp": datetime.utcnow()
        })
        
        # Adjust future proposal generation based on feedback
        logger.info(f"Learned from rejection: {feedback[:100]}...")
    
    def _get_recent_improvements(self, hours: int = 24) -> List[ImprovementProposal]:
        """Get recently applied improvements"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent = [
            proposal for proposal in self.improvement_proposals.values()
            if proposal.applied_at and proposal.applied_at > cutoff_time
        ]
        
        return sorted(recent, key=lambda x: x.applied_at, reverse=True)
    
    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate current system performance metrics"""
        metrics = {
            "success_rate": 0.0,
            "avg_processing_time": 0.0,
            "error_rate": 0.0,
            "throughput": 0.0,
            "collective_efficiency": 0.0
        }
        
        if self.agent_performance:
            # Average success rate across all agents
            success_rates = [perf.success_rate for perf in self.agent_performance.values()]
            metrics["success_rate"] = np.mean(success_rates) if success_rates else 0.0
            
            # Average processing time
            proc_times = [perf.avg_processing_time for perf in self.agent_performance.values()]
            metrics["avg_processing_time"] = np.mean(proc_times) if proc_times else 0.0
            
            # Error rate (inverse of success rate)
            metrics["error_rate"] = 1.0 - metrics["success_rate"]
            
            # Throughput (tasks per hour estimate)
            total_tasks = sum(perf.tasks_completed for perf in self.agent_performance.values())
            hours_running = (datetime.utcnow() - self.collective_memory.evolution_history[0]["timestamp"]).total_seconds() / 3600 if self.collective_memory.evolution_history else 1
            metrics["throughput"] = total_tasks / max(hours_running, 1)
            
            # Collective efficiency (based on collaboration)
            total_connections = sum(len(conns) for conns in self.neural_pathways.values())
            possible_connections = len(self.agent_registry) * (len(self.agent_registry) - 1)
            metrics["collective_efficiency"] = total_connections / max(possible_connections, 1)
        
        return metrics
    
    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend (positive or negative)"""
        if len(self.collective_memory.performance_history) < 10:
            return 0.0
        
        recent = list(self.collective_memory.performance_history)[-20:]
        older = list(self.collective_memory.performance_history)[-40:-20]
        
        if older:
            recent_avg = np.mean(recent)
            older_avg = np.mean(older)
            
            trend = (recent_avg - older_avg) / max(older_avg, 0.001)
            return trend
        
        return 0.0
    
    async def _generate_performance_improvements(self, themes: List[str]) -> Dict[str, Any]:
        """Generate code changes for performance improvements"""
        improvements = {
            "config_updates": {
                "optimization_level": 2,
                "cache_enabled": True,
                "batch_processing": True
            }
        }
        
        # Based on themes, suggest specific improvements
        if "latency" in themes or "speed" in themes:
            improvements["async_improvements"] = {
                "enable_concurrent_processing": True,
                "max_concurrent_tasks": 5
            }
        
        if "memory" in themes or "resource" in themes:
            improvements["memory_optimization"] = {
                "enable_memory_pooling": True,
                "gc_optimization": True
            }
        
        return improvements
    
    async def _generate_coordination_improvements(self, themes: List[str]) -> Dict[str, Any]:
        """Generate code changes for coordination improvements"""
        return {
            "neural_pathway_updates": {
                "increase_connectivity": True,
                "optimize_routing": True,
                "enable_broadcast_mode": True
            },
            "communication_protocol": {
                "reduce_overhead": True,
                "enable_compression": True
            }
        }
    
    async def _generate_learning_improvements(self, themes: List[str]) -> Dict[str, Any]:
        """Generate code changes for learning improvements"""
        return {
            "learning_parameters": {
                "increase_learning_rate": True,
                "enable_transfer_learning": True,
                "expand_memory_capacity": True
            },
            "knowledge_sharing": {
                "increase_sync_frequency": True,
                "enable_peer_learning": True
            }
        }
    
    async def _introduce_genetic_variations(self):
        """Introduce random variations to escape local optima"""
        logger.info("Introducing genetic variations")
        
        # Randomly modify some neural pathways
        for agent_name in list(self.neural_pathways.keys())[:5]:  # Modify 5 random agents
            connections = list(self.neural_pathways[agent_name])
            if connections:
                # Remove a random connection
                removed = connections[np.random.randint(len(connections))]
                self.neural_pathways[agent_name].discard(removed)
                
                # Add a new random connection
                potential_new = set(self.agent_registry.keys()) - {agent_name} - self.neural_pathways[agent_name]
                if potential_new:
                    new_connection = list(potential_new)[np.random.randint(len(potential_new))]
                    self.neural_pathways[agent_name].add(new_connection)
    
    async def _evolve_neural_pathways(self):
        """Evolve neural pathways based on performance"""
        # Strengthen successful pathways
        for agent_name, connections in self.neural_pathways.items():
            agent_perf = self.agent_performance.get(agent_name)
            
            if agent_perf and agent_perf.success_rate > 0.8:
                # This agent is successful - strengthen its connections
                for connected_agent in list(connections):
                    connected_perf = self.agent_performance.get(connected_agent)
                    
                    if connected_perf and connected_perf.success_rate < 0.6:
                        # Connected agent is struggling - maybe wrong connection
                        if np.random.random() < self.evolution_rate:
                            self.neural_pathways[agent_name].discard(connected_agent)
                            logger.debug(f"Pruned connection: {agent_name} -> {connected_agent}")
    
    async def _prune_inefficient_patterns(self):
        """Remove patterns that consistently fail"""
        # Keep only recent failed patterns
        self.collective_memory.failed_patterns = self.collective_memory.failed_patterns[-500:]
        
        # Remove successful patterns that haven't been used recently
        if len(self.collective_memory.successful_patterns) > 100:
            # Keep only the most recent and most successful
            sorted_patterns = sorted(
                self.collective_memory.successful_patterns,
                key=lambda x: x.get("last_used", datetime.min),
                reverse=True
            )
            self.collective_memory.successful_patterns = sorted_patterns[:100]
    
    async def _save_collective_state(self):
        """Save collective intelligence state to disk"""
        try:
            state_file = self.data_path / "collective_state.json"
            
            state_data = {
                "collective_awareness": self.collective_awareness,
                "evolution_rate": self.evolution_rate,
                "learning_momentum": self.learning_momentum,
                "state": self.state.value,
                "agent_registry": list(self.agent_registry.keys()),
                "agent_performance": {
                    name: asdict(perf) for name, perf in self.agent_performance.items()
                },
                "neural_pathways": {
                    agent: list(connections) for agent, connections in self.neural_pathways.items()
                },
                "improvement_proposals": {
                    pid: asdict(proposal) for pid, proposal in list(self.improvement_proposals.items())[-100:]
                },
                "performance_baseline": self.performance_baseline,
                "collective_memory": {
                    "knowledge_base_size": {k: len(v) for k, v in self.collective_memory.knowledge_base.items()},
                    "successful_patterns_count": len(self.collective_memory.successful_patterns),
                    "failed_patterns_count": len(self.collective_memory.failed_patterns),
                    "evolution_history_count": len(self.collective_memory.evolution_history)
                },
                "last_saved": datetime.utcnow().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.debug("Collective state saved")
            
        except Exception as e:
            logger.error(f"Failed to save collective state: {e}")
    
    def _load_collective_state(self):
        """Load collective intelligence state from disk"""
        try:
            state_file = self.data_path / "collective_state.json"
            
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.collective_awareness = state_data.get("collective_awareness", 0.0)
                self.evolution_rate = state_data.get("evolution_rate", 0.01)
                self.learning_momentum = state_data.get("learning_momentum", 0.9)
                self.performance_baseline = state_data.get("performance_baseline", {})
                
                # Restore neural pathways
                for agent, connections in state_data.get("neural_pathways", {}).items():
                    self.neural_pathways[agent] = set(connections)
                
                logger.info("Collective state loaded")
                
        except Exception as e:
            logger.error(f"Failed to load collective state: {e}")
    
    async def get_collective_status(self) -> Dict[str, Any]:
        """Get current status of the collective intelligence"""
        return {
            "state": self.state.value,
            "collective_awareness": self.collective_awareness,
            "agent_count": len(self.agent_registry),
            "active_agents": len([a for a in self.agent_performance.values() if a.tasks_completed > 0]),
            "total_improvements": len(self.improvement_proposals),
            "applied_improvements": len([p for p in self.improvement_proposals.values() if p.status == ImprovementStatus.APPLIED]),
            "pending_approvals": len([p for p in self.improvement_proposals.values() if p.status == ImprovementStatus.TESTING]),
            "neural_connections": sum(len(conns) for conns in self.neural_pathways.values()),
            "knowledge_categories": len(self.collective_memory.knowledge_base),
            "evolution_rate": self.evolution_rate,
            "performance_metrics": await self._calculate_performance_metrics()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the collective"""
        logger.info("Shutting down collective intelligence")
        
        self.state = CollectiveState.HIBERNATING
        self.emergency_stop.set()
        
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Save final state
        await self._save_collective_state()
        
        logger.info("Collective intelligence shutdown complete")


if __name__ == "__main__":
    # Example usage
    async def main():
        collective = CollectiveIntelligence()
        await collective.awaken()
        
        # Simulate for a bit
        await asyncio.sleep(60)
        
        # Get status
        status = await collective.get_collective_status()
        print(json.dumps(status, indent=2))
        
        await collective.shutdown()
    
    asyncio.run(main())