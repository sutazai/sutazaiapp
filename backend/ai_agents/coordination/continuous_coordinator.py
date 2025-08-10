#!/usr/bin/env python3
"""
Continuous Agent Coordinator
Ensures all AI agents are working together at all times
"""

import asyncio
import json
import logging
import time
import redis.asyncio as redis
import aiohttp
from dataclasses import dataclass, field
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentStatus:
    name: str
    status: str = "unknown"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    capabilities: List[str] = field(default_factory=list)
    collaborations: Set[str] = field(default_factory=set)

@dataclass
class CollaborationPattern:
    agents: List[str]
    pattern_type: str  # "sequential", "parallel", "hierarchical", "peer-to-peer"
    success_rate: float = 0.0
    usage_count: int = 0

class ContinuousCoordinator:
    def __init__(self):
        self.redis_client = None
        self.agents: Dict[str, AgentStatus] = {}
        self.collaboration_patterns: List[CollaborationPattern] = []
        self.active_collaborations: Dict[str, List[str]] = {}
        self.heartbeat_interval = 10  # seconds
        self.coordination_interval = 5  # seconds
        self.collaboration_history = defaultdict(list)
        
    async def initialize(self):
        """Initialize the continuous coordinator"""
        # Connect to Redis
        self.redis_client = await redis.Redis(
            host="redis",
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Initialize collaboration patterns
        self._init_collaboration_patterns()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._coordination_loop())
        asyncio.create_task(self._collaboration_optimizer())
        asyncio.create_task(self._agent_health_checker())
        asyncio.create_task(self._workload_balancer())
        asyncio.create_task(self._knowledge_synchronizer())
        
        logger.info("Continuous Coordinator initialized")
    
    def _init_collaboration_patterns(self):
        """Initialize known effective collaboration patterns"""
        self.collaboration_patterns = [
            # Development patterns
            CollaborationPattern(
                agents=["senior-ai-engineer", "senior-backend-developer", "senior-frontend-developer"],
                pattern_type="parallel",
                success_rate=0.95
            ),
            # Deployment patterns
            CollaborationPattern(
                agents=["testing-qa-validator", "security-pentesting-specialist", "deployment-automation-master"],
                pattern_type="sequential",
                success_rate=0.98
            ),
            # Optimization patterns
            CollaborationPattern(
                agents=["hardware-resource-optimizer", "context-optimization-engineer", "ai-agent-orchestrator"],
                pattern_type="hierarchical",
                success_rate=0.92
            ),
            # Architecture patterns
            CollaborationPattern(
                agents=["agi-system-architect", "autonomous-system-controller", "ai-agent-creator"],
                pattern_type="peer-to-peer",
                success_rate=0.90
            ),
            # Infrastructure patterns
            CollaborationPattern(
                agents=["infrastructure-devops-manager", "deployment-automation-master", "system-optimizer-reorganizer"],
                pattern_type="sequential",
                success_rate=0.96
            )
        ]
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and maintain status"""
        while True:
            try:
                # Get all registered agents
                agent_keys = []
                async for key in self.redis_client.scan_iter(match="agent:registry:*"):
                    agent_keys.append(key)
                
                for key in agent_keys:
                    agent_name = key.split(":")[-1]
                    agent_data = await self.redis_client.hgetall(key)
                    
                    # Update or create agent status
                    if agent_name not in self.agents:
                        self.agents[agent_name] = AgentStatus(
                            name=agent_name,
                            capabilities=json.loads(agent_data.get("capabilities", "[]"))
                        )
                    
                    # Update status
                    self.agents[agent_name].status = agent_data.get("status", "unknown")
                    
                    # Send heartbeat
                    await self._send_heartbeat(agent_name)
                
                # Check for stale agents
                now = datetime.now()
                for agent in self.agents.values():
                    if (now - agent.last_heartbeat).seconds > 60:
                        agent.status = "stale"
                        await self._revive_agent(agent.name)
                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self, agent_name: str):
        """Send heartbeat to agent"""
        await self.redis_client.publish(
            f"agent:heartbeat:{agent_name}",
            json.dumps({
                "timestamp": datetime.now().isoformat(),
                "coordinator": "continuous_coordinator"
            })
        )
    
    async def _revive_agent(self, agent_name: str):
        """Attempt to revive a stale agent"""
        logger.warning(f"Attempting to revive stale agent: {agent_name}")
        
        # Request MCP to check/restart agent
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://mcp-server:8100/manage_agent_workspace",
                    json={
                        "action": "restart",
                        "agent_name": agent_name
                    }
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully requested revival of {agent_name}")
        except Exception as e:
            logger.error(f"Failed to revive agent {agent_name}: {e}")
    
    async def _coordination_loop(self):
        """Main coordination loop - ensures agents work together"""
        while True:
            try:
                # Get pending tasks
                pending_tasks = await self._get_pending_tasks()
                
                # Analyze tasks and find collaboration opportunities
                collaborations = await self._identify_collaborations(pending_tasks)
                
                # Initiate collaborations
                for collab in collaborations:
                    await self._initiate_collaboration(collab)
                
                # Monitor active collaborations
                await self._monitor_collaborations()
                
                # Create proactive tasks if system is idle
                if len(pending_tasks) < 5:
                    await self._create_proactive_tasks()
                
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
            
            await asyncio.sleep(self.coordination_interval)
    
    async def _get_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks from the queue"""
        tasks = []
        task_count = await self.redis_client.llen("agent:tasks")
        
        if task_count > 0:
            # Peek at tasks without removing them
            task_data = await self.redis_client.lrange("agent:tasks", 0, task_count - 1)
            for data in task_data:
                try:
                    tasks.append(json.loads(data))
                except (IOError, OSError, FileNotFoundError) as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass
        
        return tasks
    
    async def _identify_collaborations(self, tasks: List[Dict]) -> List[Dict]:
        """Identify opportunities for agent collaboration"""
        collaborations = []
        
        # Group tasks by type
        task_groups = defaultdict(list)
        for task in tasks:
            task_groups[task.get("type", "general")].append(task)
        
        # Find matching collaboration patterns
        for task_type, grouped_tasks in task_groups.items():
            if len(grouped_tasks) >= 2:
                # Multiple related tasks - good for collaboration
                pattern = self._find_best_pattern(task_type)
                if pattern:
                    collaborations.append({
                        "pattern": pattern,
                        "tasks": grouped_tasks[:3],  # Limit to 3 tasks
                        "type": task_type
                    })
        
        return collaborations
    
    def _find_best_pattern(self, task_type: str) -> Optional[CollaborationPattern]:
        """Find the best collaboration pattern for a task type"""
        # Map task types to suitable patterns
        if "deploy" in task_type:
            return next((p for p in self.collaboration_patterns 
                        if "deployment-automation-master" in p.agents), None)
        elif "test" in task_type or "security" in task_type:
            return next((p for p in self.collaboration_patterns 
                        if "testing-qa-validator" in p.agents), None)
        elif "optimize" in task_type:
            return next((p for p in self.collaboration_patterns 
                        if "hardware-resource-optimizer" in p.agents), None)
        elif "ai" in task_type or "ml" in task_type:
            return next((p for p in self.collaboration_patterns 
                        if "senior-ai-engineer" in p.agents), None)
        
        # Default to a random pattern
        return random.choice(self.collaboration_patterns) if self.collaboration_patterns else None
    
    async def _initiate_collaboration(self, collab: Dict):
        """Initiate a collaboration between agents"""
        pattern = collab["pattern"]
        tasks = collab["tasks"]
        collab_id = f"collab_{int(time.time() * 1000)}"
        
        logger.info(f"Initiating collaboration {collab_id} with pattern {pattern.pattern_type}")
        
        # Record active collaboration
        self.active_collaborations[collab_id] = pattern.agents
        
        # Notify agents based on pattern type
        if pattern.pattern_type == "sequential":
            await self._start_sequential_collaboration(collab_id, pattern.agents, tasks)
        elif pattern.pattern_type == "parallel":
            await self._start_parallel_collaboration(collab_id, pattern.agents, tasks)
        elif pattern.pattern_type == "hierarchical":
            await self._start_hierarchical_collaboration(collab_id, pattern.agents, tasks)
        else:  # peer-to-peer
            await self._start_peer_collaboration(collab_id, pattern.agents, tasks)
        
        # Update collaboration history
        self.collaboration_history[pattern.pattern_type].append({
            "id": collab_id,
            "agents": pattern.agents,
            "started": datetime.now().isoformat(),
            "tasks": len(tasks)
        })
    
    async def _start_sequential_collaboration(self, collab_id: str, agents: List[str], tasks: List[Dict]):
        """Start a sequential collaboration where agents work one after another"""
        for i, agent in enumerate(agents):
            await self.redis_client.publish(
                f"agent:collaboration:{agent}",
                json.dumps({
                    "collaboration_id": collab_id,
                    "type": "sequential",
                    "position": i,
                    "total_agents": len(agents),
                    "next_agent": agents[i + 1] if i < len(agents) - 1 else None,
                    "tasks": tasks if i == 0 else []
                })
            )
    
    async def _start_parallel_collaboration(self, collab_id: str, agents: List[str], tasks: List[Dict]):
        """Start a parallel collaboration where agents work simultaneously"""
        # Distribute tasks among agents
        tasks_per_agent = len(tasks) // len(agents) or 1
        
        for i, agent in enumerate(agents):
            agent_tasks = tasks[i * tasks_per_agent:(i + 1) * tasks_per_agent]
            
            await self.redis_client.publish(
                f"agent:collaboration:{agent}",
                json.dumps({
                    "collaboration_id": collab_id,
                    "type": "parallel",
                    "tasks": agent_tasks,
                    "sync_channel": f"collab:{collab_id}:sync"
                })
            )
    
    async def _start_hierarchical_collaboration(self, collab_id: str, agents: List[str], tasks: List[Dict]):
        """Start a hierarchical collaboration with a lead agent"""
        lead_agent = agents[0]
        worker_agents = agents[1:]
        
        # Notify lead agent
        await self.redis_client.publish(
            f"agent:collaboration:{lead_agent}",
            json.dumps({
                "collaboration_id": collab_id,
                "type": "hierarchical",
                "role": "lead",
                "workers": worker_agents,
                "tasks": tasks
            })
        )
        
        # Notify worker agents
        for agent in worker_agents:
            await self.redis_client.publish(
                f"agent:collaboration:{agent}",
                json.dumps({
                    "collaboration_id": collab_id,
                    "type": "hierarchical",
                    "role": "worker",
                    "lead": lead_agent
                })
            )
    
    async def _start_peer_collaboration(self, collab_id: str, agents: List[str], tasks: List[Dict]):
        """Start a peer-to-peer collaboration where agents communicate directly"""
        for agent in agents:
            peers = [a for a in agents if a != agent]
            
            await self.redis_client.publish(
                f"agent:collaboration:{agent}",
                json.dumps({
                    "collaboration_id": collab_id,
                    "type": "peer-to-peer",
                    "peers": peers,
                    "tasks": tasks,
                    "communication_channel": f"collab:{collab_id}:p2p"
                })
            )
    
    async def _monitor_collaborations(self):
        """Monitor active collaborations and ensure progress"""
        completed_collabs = []
        
        for collab_id, agents in self.active_collaborations.items():
            # Check collaboration status
            status = await self._check_collaboration_status(collab_id, agents)
            
            if status == "completed":
                completed_collabs.append(collab_id)
                logger.info(f"Collaboration {collab_id} completed successfully")
            elif status == "stuck":
                # Intervention needed
                await self._unstick_collaboration(collab_id, agents)
        
        # Clean up completed collaborations
        for collab_id in completed_collabs:
            del self.active_collaborations[collab_id]
    
    async def _check_collaboration_status(self, collab_id: str, agents: List[str]) -> str:
        """Check the status of a collaboration"""
        # Check if all agents have reported completion
        completed_count = 0
        
        for agent in agents:
            status = await self.redis_client.hget(f"collab:{collab_id}:status", agent)
            if status == "completed":
                completed_count += 1
        
        if completed_count == len(agents):
            return "completed"
        elif completed_count > 0:
            return "in_progress"
        else:
            # Check if collaboration is stuck
            start_time = await self.redis_client.hget(f"collab:{collab_id}:meta", "start_time")
            if start_time:
                elapsed = (datetime.now() - datetime.fromisoformat(start_time)).seconds
                if elapsed > 300:  # 5 minutes
                    return "stuck"
        
        return "active"
    
    async def _unstick_collaboration(self, collab_id: str, agents: List[str]):
        """Attempt to unstick a collaboration"""
        logger.warning(f"Collaboration {collab_id} appears stuck, intervening...")
        
        # Send nudge to all agents
        for agent in agents:
            await self.redis_client.publish(
                f"agent:nudge:{agent}",
                json.dumps({
                    "collaboration_id": collab_id,
                    "message": "Please check collaboration progress",
                    "action": "status_update_required"
                })
            )
    
    async def _create_proactive_tasks(self):
        """Create proactive tasks to keep agents busy and working together"""
        proactive_tasks = [
            {
                "type": "optimization",
                "description": "Analyze and optimize system performance metrics",
                "collaborative": True
            },
            {
                "type": "security",
                "description": "Perform routine security audit and vulnerability assessment",
                "collaborative": True
            },
            {
                "type": "testing",
                "description": "Run automated test suite and generate coverage report",
                "collaborative": True
            },
            {
                "type": "maintenance",
                "description": "Clean up unused resources and optimize storage",
                "collaborative": True
            },
            {
                "type": "documentation",
                "description": "Update system documentation and API references",
                "collaborative": True
            }
        ]
        
        # Select a random task
        task = random.choice(proactive_tasks)
        
        # Submit via task queue
        await self.redis_client.lpush("agent:tasks", json.dumps(task))
        
        logger.info(f"Created proactive task: {task['description']}")
    
    async def _collaboration_optimizer(self):
        """Optimize collaboration patterns based on success metrics"""
        while True:
            try:
                # Analyze collaboration history
                for pattern in self.collaboration_patterns:
                    # Calculate success rate from recent collaborations
                    success_count = 0
                    total_count = 0
                    
                    # Update pattern usage and success metrics
                    pattern.usage_count = len(self.collaboration_history[pattern.pattern_type])
                    
                    # Adjust patterns based on performance
                    if pattern.success_rate < 0.8 and pattern.usage_count > 10:
                        logger.warning(f"Pattern {pattern.pattern_type} underperforming, adjusting...")
                        # Could swap agent positions or change pattern type
                
            except Exception as e:
                logger.error(f"Collaboration optimizer error: {e}")
            
            await asyncio.sleep(300)  # Run every 5 minutes
    
    async def _agent_health_checker(self):
        """Continuously check agent health and take corrective actions"""
        while True:
            try:
                unhealthy_agents = []
                
                for agent_name, agent_status in self.agents.items():
                    if agent_status.status in ["unhealthy", "stale", "error"]:
                        unhealthy_agents.append(agent_name)
                
                # Attempt to heal unhealthy agents
                for agent in unhealthy_agents:
                    await self._heal_agent(agent)
                
            except Exception as e:
                logger.error(f"Health checker error: {e}")
            
            await asyncio.sleep(30)
    
    async def _heal_agent(self, agent_name: str):
        """Attempt to heal an unhealthy agent"""
        logger.info(f"Attempting to heal agent: {agent_name}")
        
        # First, try a soft restart via MCP
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://mcp-server:8100/manage_agent_workspace",
                    json={
                        "action": "restart",
                        "agent_name": agent_name,
                        "workspace_data": {"soft_restart": True}
                    }
                ) as response:
                    if response.status == 200:
                        logger.info(f"Soft restart initiated for {agent_name}")
        except Exception as e:
            logger.error(f"Failed to heal agent {agent_name}: {e}")
    
    async def _workload_balancer(self):
        """Balance workload across agents"""
        while True:
            try:
                # Calculate workload for each agent
                agent_workloads = {}
                
                for agent_name, agent_status in self.agents.items():
                    workload = len(agent_status.current_tasks)
                    agent_workloads[agent_name] = workload
                
                # Find overloaded and underutilized agents
                avg_workload = sum(agent_workloads.values()) / len(agent_workloads) if agent_workloads else 0
                
                overloaded = [a for a, w in agent_workloads.items() if w > avg_workload * 1.5]
                underutilized = [a for a, w in agent_workloads.items() if w < avg_workload * 0.5]
                
                # Rebalance if needed
                if overloaded and underutilized:
                    await self._rebalance_workload(overloaded, underutilized)
                
            except Exception as e:
                logger.error(f"Workload balancer error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _rebalance_workload(self, overloaded: List[str], underutilized: List[str]):
        """Rebalance workload between agents"""
        logger.info(f"Rebalancing workload: {len(overloaded)} overloaded, {len(underutilized)} underutilized")
        
        # For each overloaded agent, try to redistribute some tasks
        for overloaded_agent in overloaded:
            # Find suitable underutilized agent based on capabilities
            for underutilized_agent in underutilized:
                # Check if capabilities match
                overloaded_caps = set(self.agents[overloaded_agent].capabilities)
                underutilized_caps = set(self.agents[underutilized_agent].capabilities)
                
                if overloaded_caps.intersection(underutilized_caps):
                    # Can transfer tasks
                    await self.redis_client.publish(
                        f"agent:rebalance:{overloaded_agent}",
                        json.dumps({
                            "action": "transfer_tasks",
                            "to_agent": underutilized_agent,
                            "max_tasks": 2
                        })
                    )
                    break
    
    async def _knowledge_synchronizer(self):
        """Synchronize knowledge between agents"""
        while True:
            try:
                # Create knowledge sharing sessions
                knowledge_topics = [
                    "recent_deployments",
                    "optimization_strategies",
                    "security_findings",
                    "performance_metrics",
                    "best_practices"
                ]
                
                topic = random.choice(knowledge_topics)
                
                # Select agents for knowledge sharing
                participating_agents = random.sample(
                    list(self.agents.keys()),
                    min(5, len(self.agents))
                )
                
                # Initiate knowledge sharing session
                session_id = f"knowledge_{int(time.time())}"
                
                for agent in participating_agents:
                    await self.redis_client.publish(
                        f"agent:knowledge:{agent}",
                        json.dumps({
                            "session_id": session_id,
                            "topic": topic,
                            "participants": participating_agents,
                            "action": "share_knowledge"
                        })
                    )
                
                logger.info(f"Initiated knowledge sharing session {session_id} on {topic}")
                
            except Exception as e:
                logger.error(f"Knowledge synchronizer error: {e}")
            
            await asyncio.sleep(600)  # Every 10 minutes
    
    async def get_coordination_status(self) -> Dict:
        """Get current coordination status"""
        return {
            "active_agents": len([a for a in self.agents.values() if a.status == "active"]),
            "total_agents": len(self.agents),
            "active_collaborations": len(self.active_collaborations),
            "collaboration_history": {
                pattern_type: len(history)
                for pattern_type, history in self.collaboration_history.items()
            },
            "agent_statuses": {
                agent.name: {
                    "status": agent.status,
                    "current_tasks": len(agent.current_tasks),
                    "completed_tasks": agent.completed_tasks,
                    "collaborations": list(agent.collaborations)
                }
                for agent in self.agents.values()
            }
        }

async def main():
    """Main function to run continuous coordinator"""
    coordinator = ContinuousCoordinator()
    await coordinator.initialize()
    
    logger.info("Continuous Coordinator is running")
    logger.info("All agents will work together continuously")
    
    # Keep running
    try:
        while True:
            # Print status every minute
            status = await coordinator.get_coordination_status()
            logger.info(f"Coordination Status: {json.dumps(status, indent=2)}")
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Shutting down continuous coordinator...")

if __name__ == "__main__":
    asyncio.run(main())