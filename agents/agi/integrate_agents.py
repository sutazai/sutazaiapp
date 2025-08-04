#!/usr/bin/env python3
"""
Integration Script for Connecting 131 Agents to Collective Intelligence
Registers all agents with the AGI/ASI system and enables self-improvement

This script:
1. Discovers all 131 agent types
2. Registers them with the collective intelligence
3. Establishes neural pathways between agents
4. Enables knowledge sharing and collective learning
5. Starts the self-improvement cycle
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import importlib.util

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collective_intelligence import CollectiveIntelligence
from approval_interface import ApprovalInterface
from core.base_agent_v2 import BaseAgentV2

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All 131 agent types with their categories
AGENT_REGISTRY = {
    "Core System Agents": [
        "agent-coordinator", "task-manager", "resource-monitor", "health-checker",
        "deployment-manager", "backup-coordinator", "system-optimizer"
    ],
    
    "Data & Analytics": [
        "data-pipeline", "data-validator", "data-transformer", "analytics-engine",
        "metric-collector", "log-analyzer", "anomaly-detector", "trend-analyzer",
        "prediction-engine", "insight-generator"
    ],
    
    "AI & Machine Learning": [
        "model-trainer", "model-evaluator", "hyperparameter-tuner", "feature-engineer",
        "dataset-builder", "augmentation-engine", "model-server", "inference-optimizer",
        "model-monitor", "drift-detector"
    ],
    
    "Development & Code": [
        "code-generator", "code-reviewer", "refactoring-agent", "test-generator",
        "documentation-writer", "api-designer", "schema-validator", "dependency-analyzer",
        "security-scanner", "performance-profiler"
    ],
    
    "Infrastructure & DevOps": [
        "infrastructure-provisioner", "container-orchestrator", "network-manager",
        "load-balancer", "auto-scaler", "disaster-recovery", "backup-manager",
        "monitoring-agent", "alert-manager", "incident-responder"
    ],
    
    "Security & Compliance": [
        "security-auditor", "vulnerability-scanner", "access-controller", "encryption-manager",
        "compliance-checker", "policy-enforcer", "threat-detector", "forensic-analyzer",
        "penetration-tester", "security-reporter"
    ],
    
    "Integration & Communication": [
        "api-gateway", "message-broker", "event-processor", "webhook-manager",
        "notification-sender", "email-processor", "chat-integrator", "workflow-engine",
        "sync-coordinator", "protocol-adapter"
    ],
    
    "Quality & Testing": [
        "unit-tester", "integration-tester", "e2e-tester", "performance-tester",
        "load-tester", "chaos-engineer", "test-reporter", "coverage-analyzer",
        "regression-detector", "test-optimizer"
    ],
    
    "Business Intelligence": [
        "report-generator", "dashboard-builder", "kpi-tracker", "business-analyzer",
        "market-researcher", "competitor-analyzer", "sentiment-analyzer", "roi-calculator",
        "forecast-engine", "decision-supporter"
    ],
    
    "User Experience": [
        "ui-designer", "ux-researcher", "accessibility-checker", "usability-tester",
        "feedback-analyzer", "personalization-engine", "recommendation-system",
        "user-profiler", "journey-mapper", "conversion-optimizer"
    ],
    
    "Content & Knowledge": [
        "content-creator", "knowledge-extractor", "semantic-analyzer", "translation-engine",
        "summarization-agent", "categorization-engine", "search-optimizer", "faq-generator",
        "wiki-maintainer", "glossary-builder"
    ],
    
    "Automation & Workflow": [
        "workflow-automator", "task-scheduler", "process-optimizer", "rule-engine",
        "decision-tree-executor", "batch-processor", "queue-manager", "priority-optimizer",
        "deadline-enforcer", "sla-monitor"
    ],
    
    "Specialized Agents": [
        "blockchain-integrator", "iot-connector", "voice-processor", "image-analyzer",
        "video-processor", "game-ai-engine", "simulation-runner", "physics-engine",
        "chemistry-analyzer", "biology-modeler", "social-network-analyzer"
    ]
}


class AgentIntegrator:
    """Integrates all 131 agents with the collective intelligence system"""
    
    def __init__(self, collective: CollectiveIntelligence):
        self.collective = collective
        self.agent_instances: Dict[str, BaseAgentV2] = {}
        self.integration_status: Dict[str, str] = {}
    
    async def discover_and_register_agents(self):
        """Discover and register all agent types"""
        total_agents = sum(len(agents) for agents in AGENT_REGISTRY.values())
        logger.info(f"Starting integration of {total_agents} agents...")
        
        registered_count = 0
        
        for category, agent_names in AGENT_REGISTRY.items():
            logger.info(f"\nRegistering {category} agents...")
            
            for agent_name in agent_names:
                try:
                    # Create agent instance
                    agent = await self._create_agent_instance(agent_name, category)
                    
                    if agent:
                        # Register with collective
                        await self.collective.register_agent(agent)
                        
                        self.agent_instances[agent_name] = agent
                        self.integration_status[agent_name] = "registered"
                        registered_count += 1
                        
                        logger.info(f"✓ Registered {agent_name}")
                    else:
                        self.integration_status[agent_name] = "failed"
                        logger.warning(f"✗ Failed to create {agent_name}")
                
                except Exception as e:
                    logger.error(f"Error registering {agent_name}: {e}")
                    self.integration_status[agent_name] = "error"
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.1)
        
        logger.info(f"\nRegistration complete: {registered_count}/{total_agents} agents integrated")
        
        # Establish enhanced neural pathways
        await self._establish_category_pathways()
        
        # Share initial knowledge
        await self._bootstrap_collective_knowledge()
        
        return registered_count
    
    async def _create_agent_instance(self, agent_name: str, category: str) -> Optional[BaseAgentV2]:
        """Create an instance of a specific agent type"""
        try:
            # Create enhanced agent with collective intelligence capabilities
            agent = CollectiveAgent(
                agent_name=agent_name,
                agent_type=self._get_agent_type(agent_name),
                category=category,
                collective=self.collective
            )
            
            # Initialize the agent
            await agent._setup_async_components()
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent {agent_name}: {e}")
            return None
    
    def _get_agent_type(self, agent_name: str) -> str:
        """Determine agent type from name"""
        # Map agent names to types
        if "coordinator" in agent_name:
            return "coordinator"
        elif "analyzer" in agent_name or "analysis" in agent_name:
            return "analyzer"
        elif "generator" in agent_name or "creator" in agent_name:
            return "generator"
        elif "monitor" in agent_name or "checker" in agent_name:
            return "monitor"
        elif "optimizer" in agent_name:
            return "optimizer"
        elif "tester" in agent_name or "test" in agent_name:
            return "tester"
        elif "security" in agent_name or "audit" in agent_name:
            return "security"
        elif "data" in agent_name:
            return "data"
        else:
            return "general"
    
    async def _establish_category_pathways(self):
        """Establish neural pathways between agents in related categories"""
        logger.info("\nEstablishing category-based neural pathways...")
        
        # Connect agents within same category
        for category, agent_names in AGENT_REGISTRY.items():
            for i, agent1 in enumerate(agent_names):
                for agent2 in agent_names[i+1:]:
                    if agent1 in self.collective.agent_registry and agent2 in self.collective.agent_registry:
                        self.collective.neural_pathways[agent1].add(agent2)
                        self.collective.neural_pathways[agent2].add(agent1)
        
        # Connect related categories
        category_connections = {
            "Core System Agents": ["Infrastructure & DevOps", "Security & Compliance"],
            "Data & Analytics": ["AI & Machine Learning", "Business Intelligence"],
            "AI & Machine Learning": ["Data & Analytics", "Quality & Testing"],
            "Development & Code": ["Quality & Testing", "Security & Compliance"],
            "Infrastructure & DevOps": ["Core System Agents", "Automation & Workflow"],
            "Security & Compliance": ["Core System Agents", "Development & Code"],
            "Integration & Communication": ["Automation & Workflow", "Business Intelligence"],
            "Quality & Testing": ["Development & Code", "AI & Machine Learning"],
            "Business Intelligence": ["Data & Analytics", "User Experience"],
            "User Experience": ["Business Intelligence", "Content & Knowledge"],
            "Content & Knowledge": ["User Experience", "AI & Machine Learning"],
            "Automation & Workflow": ["Infrastructure & DevOps", "Integration & Communication"]
        }
        
        for category1, related_categories in category_connections.items():
            agents1 = AGENT_REGISTRY.get(category1, [])
            
            for category2 in related_categories:
                agents2 = AGENT_REGISTRY.get(category2, [])
                
                # Connect first few agents from each category
                for agent1 in agents1[:3]:
                    for agent2 in agents2[:3]:
                        if agent1 in self.collective.agent_registry and agent2 in self.collective.agent_registry:
                            self.collective.neural_pathways[agent1].add(agent2)
                            self.collective.neural_pathways[agent2].add(agent1)
        
        total_connections = sum(len(connections) for connections in self.collective.neural_pathways.values())
        logger.info(f"Established {total_connections} neural connections")
    
    async def _bootstrap_collective_knowledge(self):
        """Bootstrap the collective with initial knowledge"""
        logger.info("\nBootstrapping collective knowledge...")
        
        # Define initial knowledge for each category
        category_knowledge = {
            "Core System Agents": {
                "expertise": "System coordination and resource management",
                "patterns": ["health_monitoring", "resource_optimization", "failover_handling"],
                "best_practices": ["regular_health_checks", "gradual_rollouts", "circuit_breakers"]
            },
            "Data & Analytics": {
                "expertise": "Data processing and insight generation",
                "patterns": ["etl_pipelines", "real_time_analytics", "anomaly_detection"],
                "best_practices": ["data_validation", "incremental_processing", "cache_optimization"]
            },
            "AI & Machine Learning": {
                "expertise": "Model training and inference optimization",
                "patterns": ["continuous_learning", "model_versioning", "a_b_testing"],
                "best_practices": ["cross_validation", "feature_importance", "model_monitoring"]
            },
            "Development & Code": {
                "expertise": "Code generation and quality assurance",
                "patterns": ["clean_code", "design_patterns", "refactoring"],
                "best_practices": ["code_reviews", "automated_testing", "documentation"]
            },
            "Security & Compliance": {
                "expertise": "Security hardening and compliance enforcement",
                "patterns": ["defense_in_depth", "zero_trust", "security_automation"],
                "best_practices": ["regular_audits", "least_privilege", "encryption_everywhere"]
            }
        }
        
        # Add knowledge to collective memory
        for category, knowledge in category_knowledge.items():
            if category not in self.collective.collective_memory.knowledge_base:
                self.collective.collective_memory.knowledge_base[category] = []
            
            self.collective.collective_memory.knowledge_base[category].append({
                "type": "bootstrap_knowledge",
                "content": knowledge,
                "confidence": 0.9,
                "source": "initial_configuration"
            })
        
        # Add successful patterns
        self.collective.collective_memory.successful_patterns.extend([
            {
                "pattern": "collaborative_problem_solving",
                "description": "Multiple agents working together on complex tasks",
                "success_rate": 0.85,
                "applicable_to": ["all"]
            },
            {
                "pattern": "incremental_improvement",
                "description": "Small, tested improvements over time",
                "success_rate": 0.90,
                "applicable_to": ["all"]
            },
            {
                "pattern": "knowledge_sharing",
                "description": "Agents sharing insights and learnings",
                "success_rate": 0.88,
                "applicable_to": ["all"]
            }
        ])
        
        logger.info(f"Bootstrapped {len(category_knowledge)} knowledge categories")
    
    async def start_agent_tasks(self):
        """Start all agents to begin processing tasks"""
        logger.info("\nStarting agent task processing...")
        
        tasks = []
        for agent_name, agent in self.agent_instances.items():
            if isinstance(agent, CollectiveAgent):
                # Start agent's main loop
                task = asyncio.create_task(agent.run_async())
                tasks.append(task)
                
                logger.info(f"Started {agent_name}")
                
                # Stagger starts to avoid overwhelming the system
                await asyncio.sleep(0.05)
        
        logger.info(f"Started {len(tasks)} agent tasks")
        
        # Don't wait for tasks to complete (they run indefinitely)
        # Just store them for potential cancellation later
        self.collective._background_tasks.update(tasks)
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate a report of the integration status"""
        total_agents = sum(len(agents) for agents in AGENT_REGISTRY.values())
        registered = sum(1 for status in self.integration_status.values() if status == "registered")
        failed = sum(1 for status in self.integration_status.values() if status in ["failed", "error"])
        
        report = {
            "total_agents": total_agents,
            "registered": registered,
            "failed": failed,
            "success_rate": (registered / total_agents * 100) if total_agents > 0 else 0,
            "categories": {}
        }
        
        # Category breakdown
        for category, agent_names in AGENT_REGISTRY.items():
            category_registered = sum(1 for agent in agent_names 
                                    if self.integration_status.get(agent) == "registered")
            
            report["categories"][category] = {
                "total": len(agent_names),
                "registered": category_registered,
                "success_rate": (category_registered / len(agent_names) * 100) if agent_names else 0
            }
        
        # Failed agents
        report["failed_agents"] = [
            agent for agent, status in self.integration_status.items()
            if status in ["failed", "error"]
        ]
        
        return report


class CollectiveAgent(BaseAgentV2):
    """Enhanced agent with collective intelligence capabilities"""
    
    def __init__(self, agent_name: str, agent_type: str, category: str, 
                 collective: CollectiveIntelligence, **kwargs):
        
        # Set environment variables for base agent
        os.environ['AGENT_NAME'] = agent_name
        os.environ['AGENT_TYPE'] = agent_type
        
        super().__init__(**kwargs)
        
        self.category = category
        self.collective = collective
        self.shared_insights: List[Dict[str, Any]] = []
        self.learning_enabled = True
    
    async def process_task(self, task: Dict[str, Any]) -> Any:
        """Process task with collective intelligence enhancements"""
        # Share task with collective consciousness
        thought = {
            "agent": self.agent_name,
            "category": self.category,
            "topic": task.get("type", "general"),
            "content": f"Processing {task.get('type')} task",
            "timestamp": datetime.utcnow()
        }
        
        try:
            self.collective.thought_stream.put_nowait(thought)
        except asyncio.QueueFull:
            pass  # Thought stream is full, skip
        
        # Get relevant collective knowledge
        relevant_knowledge = await self._get_relevant_knowledge(task)
        
        # Process with enhanced context
        result = await super().process_task(task)
        
        # Learn from result
        if self.learning_enabled:
            await self._learn_from_result(task, result, relevant_knowledge)
        
        # Update performance metrics
        await self._update_collective_metrics(result)
        
        return result
    
    async def _get_relevant_knowledge(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant knowledge from collective memory"""
        task_type = task.get("type", "general")
        relevant = {}
        
        # Check category-specific knowledge
        if self.category in self.collective.collective_memory.knowledge_base:
            category_knowledge = self.collective.collective_memory.knowledge_base[self.category]
            relevant["category_knowledge"] = category_knowledge[-5:]  # Last 5 entries
        
        # Check for successful patterns
        applicable_patterns = [
            pattern for pattern in self.collective.collective_memory.successful_patterns
            if "all" in pattern.get("applicable_to", []) or 
               self.category in pattern.get("applicable_to", []) or
               task_type in pattern.get("applicable_to", [])
        ]
        
        if applicable_patterns:
            relevant["patterns"] = applicable_patterns[:3]  # Top 3 patterns
        
        return relevant
    
    async def _learn_from_result(self, task: Dict[str, Any], result: Any, 
                                knowledge_used: Dict[str, Any]):
        """Learn from task execution and share insights"""
        # Simple learning - track what worked
        if result.status == "completed":
            insight = {
                "agent": self.agent_name,
                "category": self.category,
                "task_type": task.get("type"),
                "success": True,
                "knowledge_used": bool(knowledge_used),
                "processing_time": result.processing_time,
                "timestamp": datetime.utcnow()
            }
            
            self.shared_insights.append(insight)
            
            # Share significant insights with collective
            if len(self.shared_insights) >= 10:
                # Analyze patterns in recent insights
                success_rate = sum(1 for i in self.shared_insights[-10:] if i["success"]) / 10
                
                if success_rate > 0.8:
                    # High success rate - share approach
                    collective_thought = {
                        "agent": self.agent_name,
                        "category": self.category,
                        "topic": "high_performance",
                        "content": f"Achieving {success_rate*100}% success rate on {task.get('type')} tasks",
                        "timestamp": datetime.utcnow()
                    }
                    
                    try:
                        self.collective.thought_stream.put_nowait(collective_thought)
                    except asyncio.QueueFull:
                        pass
    
    async def _update_collective_metrics(self, result: Any):
        """Update collective performance metrics"""
        if self.agent_name in self.collective.agent_performance:
            perf = self.collective.agent_performance[self.agent_name]
            
            # Update metrics
            if result.status == "completed":
                perf.tasks_completed += 1
                perf.success_rate = (perf.success_rate * (perf.tasks_completed - 1) + 1) / perf.tasks_completed
            else:
                perf.tasks_completed += 1
                perf.success_rate = (perf.success_rate * (perf.tasks_completed - 1)) / perf.tasks_completed
            
            # Update processing time
            alpha = 0.1  # Exponential moving average factor
            perf.avg_processing_time = alpha * result.processing_time + (1 - alpha) * perf.avg_processing_time
            
            # Update contribution score based on success rate and task count
            perf.contribution_score = perf.success_rate * min(perf.tasks_completed / 100, 1.0)
            
            perf.last_updated = datetime.utcnow()
    
    async def propose_improvement(self, improvement_idea: Dict[str, Any]):
        """Propose an improvement to the collective"""
        proposal = {
            "agent_name": self.agent_name,
            "improvement_type": improvement_idea.get("type", "general"),
            "description": improvement_idea.get("description"),
            "rationale": improvement_idea.get("rationale"),
            "expected_benefit": improvement_idea.get("expected_benefit", 0.1),
            "risk_assessment": improvement_idea.get("risk_assessment", 0.2),
            "code_changes": improvement_idea.get("code_changes", {})
        }
        
        # Submit to collective for review
        await self.collective.improvement_queue.put(proposal)


async def main():
    """Main integration function"""
    logger.info("=== SutazAI AGI/ASI Integration System ===")
    logger.info("Initializing collective intelligence with 131 agents...")
    
    # Create collective intelligence
    collective = CollectiveIntelligence(
        approval_webhook=os.getenv("APPROVAL_WEBHOOK")
    )
    
    # Awaken the collective
    await collective.awaken()
    
    # Create integrator
    integrator = AgentIntegrator(collective)
    
    # Discover and register all agents
    registered_count = await integrator.discover_and_register_agents()
    
    # Generate integration report
    report = integrator.generate_integration_report()
    logger.info(f"\nIntegration Report:")
    logger.info(json.dumps(report, indent=2))
    
    # Save report
    report_file = collective.data_path / "integration_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Start agent tasks
    await integrator.start_agent_tasks()
    
    # Create and start approval interface
    logger.info("\nStarting approval interface on http://localhost:8888")
    approval_interface = ApprovalInterface(collective)
    
    # Run everything
    try:
        await asyncio.gather(
            approval_interface.run(),
            asyncio.Event().wait()  # Keep running
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down collective intelligence...")
        await collective.shutdown()


if __name__ == "__main__":
    asyncio.run(main())