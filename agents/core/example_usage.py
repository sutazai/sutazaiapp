#!/usr/bin/env python3
"""
Example Usage of Enhanced SutazAI Agent System
Demonstrates how to use the new BaseAgentV2 with all 131 agents

This file shows practical examples for:
- Creating new agents with BaseAgentV2
- Migrating existing agents  
- Using legacy wrapper for backward compatibility
- Advanced features like circuit breakers and request queues
"""

import asyncio
import logging
from typing import Dict, Any
import sys
import os

# Add agents to path
sys.path.append('/opt/sutazaiapp/agents')

from core.base_agent_v2 import BaseAgentV2, TaskResult, AgentStatus
from core.ollama_pool import OllamaConnectionPool
from core.circuit_breaker import CircuitBreaker
from core.request_queue import RequestQueue, RequestPriority
from core.migration_helper import LegacyAgentWrapper, create_agent_factory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example 1: New Agent with BaseAgentV2
class AISystemArchitectAgent(BaseAgentV2):
    """
    Example of a new agent using BaseAgentV2 directly
    This represents one of the 131 agents in the system
    """
    
    def __init__(self):
        super().__init__(
            max_concurrent_tasks=3,
            max_ollama_connections=2,
            heartbeat_interval=30
        )
        self.architecture_templates = []
    
    async def process_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process architecture-related tasks
        """
        start_time = asyncio.get_event_loop().time()
        task_id = task.get("id", "unknown")
        task_type = task.get("type", "design")
        
        try:
            if task_type == "system_design":
                result = await self._design_system(task)
            elif task_type == "architecture_review":
                result = await self._review_architecture(task)
            elif task_type == "performance_analysis":
                result = await self._analyze_performance(task)
            else:
                result = await self._generic_architecture_task(task)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return TaskResult(
                task_id=task_id,
                status="completed",
                result=result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Architecture task failed: {e}")
            
            return TaskResult(
                task_id=task_id,
                status="failed",
                result={"error": str(e)},
                processing_time=processing_time,
                error=str(e)
            )
    
    async def _design_system(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Design a system architecture"""
        requirements = task.get("requirements", "")
        constraints = task.get("constraints", "")
        
        # Use Ollama for AI-assisted design
        prompt = f"""
        As a senior system architect, design a system with these requirements:
        {requirements}
        
        Constraints: {constraints}
        
        Provide a detailed architecture including:
        1. System components
        2. Data flow
        3. Technology stack
        4. Scalability considerations
        5. Security measures
        """
        
        ai_response = await self.query_ollama(
            prompt,
            system="You are an expert system architect with 15+ years of experience."
        )
        
        return {
            "type": "system_design",
            "architecture": ai_response,
            "requirements": requirements,
            "constraints": constraints,
            "status": "completed"
        }
    
    async def _review_architecture(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Review existing architecture"""
        architecture_doc = task.get("architecture", "")
        
        prompt = f"""
        Review this system architecture and provide detailed feedback:
        {architecture_doc}
        
        Focus on:
        1. Scalability issues
        2. Security vulnerabilities  
        3. Performance bottlenecks
        4. Maintainability concerns
        5. Recommended improvements
        """
        
        review = await self.query_ollama(prompt)
        
        return {
            "type": "architecture_review",
            "review": review,
            "original_architecture": architecture_doc,
            "status": "completed"
        }
    
    async def _analyze_performance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system performance"""
        metrics = task.get("metrics", {})
        
        prompt = f"""
        Analyze these system performance metrics and provide recommendations:
        {metrics}
        
        Provide:
        1. Performance assessment
        2. Bottleneck identification
        3. Optimization recommendations
        4. Scaling strategies
        """
        
        analysis = await self.query_ollama(prompt)
        
        return {
            "type": "performance_analysis",
            "analysis": analysis,
            "metrics": metrics,
            "status": "completed"
        }
    
    async def _generic_architecture_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic architecture tasks"""
        description = task.get("description", "")
        
        prompt = f"""
        As a system architect, help with this task:
        {description}
        
        Provide practical, actionable guidance.
        """
        
        response = await self.query_ollama(prompt)
        
        return {
            "type": "generic_architecture",
            "response": response,
            "task_description": description,
            "status": "completed"
        }


# Example 2: Migrated Legacy Agent
class DataAnalysisEngineerAgent(BaseAgentV2):
    """
    Example of a migrated agent from BaseAgent to BaseAgentV2
    Shows how to update existing agent code
    """
    
    def __init__(self):
        super().__init__(
            max_concurrent_tasks=2,  # Conservative for data processing
            max_ollama_connections=1
        )
        self.analysis_cache = {}
    
    async def process_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process data analysis tasks
        """
        task_id = task.get("id", "unknown")
        
        try:
            # Old sync method would be:
            # result = self.analyze_data(task["data"])
            
            # New async method:
            result = await self._analyze_data_async(task.get("data", {}))
            
            return TaskResult(
                task_id=task_id,
                status="completed",
                result=result,
                processing_time=1.0
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task_id,
                status="failed",
                result={"error": str(e)},
                processing_time=0.0,
                error=str(e)
            )
    
    async def _analyze_data_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Async data analysis"""
        # Simulate data processing
        await asyncio.sleep(0.5)
        
        # Use AI for insights
        data_summary = str(data)[:500]  # Truncate for prompt
        
        insights = await self.query_ollama(
            f"Analyze this data and provide key insights: {data_summary}",
            model="qwen2.5-coder:7b"  # Use coding model for data analysis
        )
        
        return {
            "analysis": insights,
            "data_points": len(data),
            "timestamp": asyncio.get_event_loop().time()
        }


# Example 3: Using Legacy Wrapper
async def example_legacy_wrapper():
    """
    Show how to use legacy wrapper for existing agents
    """
    logger.info("=== Legacy Wrapper Example ===")
    
    # Import an existing legacy agent
    from agent_base import BaseAgent
    
    class OldStyleAgent(BaseAgent):
        def process_task(self, task):
            # Old synchronous processing
            return {
                "status": "success",
                "message": f"Old style processing for {task.get('id')}",
                "data": task
            }
    
    # Wrap with new system
    wrapped_agent = LegacyAgentWrapper(OldStyleAgent)
    
    # Use as normal BaseAgentV2 agent
    await wrapped_agent._setup_async_components()
    
    test_task = {"id": "legacy_test", "type": "old_task"}
    result = await wrapped_agent.process_task(test_task)
    
    logger.info(f"Legacy wrapper result: {result.status}")
    
    await wrapped_agent._cleanup_async_components()


# Example 4: Advanced Features Demo
async def example_advanced_features():
    """
    Demonstrate advanced features like circuit breakers and request queues
    """
    logger.info("=== Advanced Features Example ===")
    
    # Circuit breaker for resilience
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=10.0,
        name="ollama_breaker"
    )
    
    # Request queue for concurrency control
    queue = RequestQueue(
        max_queue_size=20,
        max_concurrent=3,
        name="agent_queue"
    )
    
    async def risky_ollama_call(prompt):
        """Simulate risky operation that might fail"""
        async with OllamaConnectionPool(max_connections=1) as pool:
            return await pool.generate(prompt)
    
    # Submit requests with different priorities
    high_priority_id = await queue.submit(
        risky_ollama_call,
        "High priority prompt",
        priority=RequestPriority.HIGH
    )
    
    normal_priority_id = await queue.submit(
        risky_ollama_call, 
        "Normal priority prompt",
        priority=RequestPriority.NORMAL
    )
    
    # Get results
    try:
        high_result = await queue.get_result(high_priority_id, timeout=30)
        normal_result = await queue.get_result(normal_priority_id, timeout=30)
        
        logger.info("Queue processing completed successfully")
        
    except Exception as e:
        logger.error(f"Queue processing failed: {e}")
    
    # Get statistics
    queue_stats = queue.get_stats()
    breaker_stats = breaker.get_stats()
    
    logger.info(f"Queue completed: {queue_stats['completed_requests']}")
    logger.info(f"Breaker state: {breaker_stats['state']}")
    
    await queue.close()


# Example 5: Production Agent Runner
class ProductionAgentRunner:
    """
    Production-ready agent runner for any of the 131 agents
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent = None
        
    async def initialize_agent(self):
        """Initialize agent based on name"""
        try:
            # Use factory to create appropriate agent
            self.agent = create_agent_factory(self.agent_name)
            logger.info(f"Initialized agent: {self.agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.agent_name}: {e}")
            
            # Fallback to generic BaseAgentV2
            self.agent = BaseAgentV2()
            logger.info(f"Using generic agent for {self.agent_name}")
    
    async def run_agent(self):
        """Run the agent in production mode"""
        if not self.agent:
            await self.initialize_agent()
        
        try:
            # Run the agent
            await self.agent.run_async()
            
        except KeyboardInterrupt:
            logger.info("Agent stopped by user")
        except Exception as e:
            logger.error(f"Agent runtime error: {e}")
            raise
        finally:
            logger.info(f"Agent {self.agent_name} shutdown complete")


# Example 6: Batch Agent Manager
class BatchAgentManager:
    """
    Manage multiple agents for system-wide operations
    """
    
    def __init__(self, agent_names: list):
        self.agent_names = agent_names
        self.agents = {}
        self.tasks = []
    
    async def initialize_all_agents(self):
        """Initialize all agents"""
        for name in self.agent_names:
            try:
                runner = ProductionAgentRunner(name)
                await runner.initialize_agent()
                self.agents[name] = runner.agent
                logger.info(f"Initialized {name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")
    
    async def distribute_tasks(self, tasks: list):
        """Distribute tasks across agents"""
        for i, task in enumerate(tasks):
            agent_name = self.agent_names[i % len(self.agent_names)]
            agent = self.agents.get(agent_name)
            
            if agent:
                task_coro = agent.process_task(task)
                self.tasks.append(asyncio.create_task(task_coro))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info(f"Completed {len(results)} tasks across {len(self.agents)} agents")
        return results
    
    async def get_system_health(self):
        """Get health status of all agents"""
        health_reports = {}
        
        for name, agent in self.agents.items():
            try:
                health = await agent.health_check()
                health_reports[name] = health
            except Exception as e:
                health_reports[name] = {"healthy": False, "error": str(e)}
        
        return health_reports


# Main demonstration
async def main():
    """
    Main demonstration of the enhanced agent system
    """
    logger.info("=== SutazAI Enhanced Agent System Demo ===")
    
    # Example 1: New architecture agent
    logger.info("\n1. Testing AI System Architect Agent...")
    arch_agent = AISystemArchitectAgent()
    await arch_agent._setup_async_components()
    
    test_task = {
        "id": "arch_001",
        "type": "system_design",
        "requirements": "High-performance chat system for 10k concurrent users",
        "constraints": "Limited hardware, CPU-only environment"
    }
    
    try:
        result = await arch_agent.process_task(test_task)
        logger.info(f"Architecture task completed: {result.status}")
    except Exception as e:
        logger.error(f"Architecture task failed: {e}")
    
    await arch_agent._cleanup_async_components()
    
    # Example 2: Data analysis agent
    logger.info("\n2. Testing Data Analysis Engineer Agent...")
    data_agent = DataAnalysisEngineerAgent()
    await data_agent._setup_async_components()
    
    data_task = {
        "id": "data_001",
        "data": {"users": 1000, "transactions": 50000, "errors": 12}
    }
    
    try:
        result = await data_agent.process_task(data_task)
        logger.info(f"Data analysis completed: {result.status}")
    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
    
    await data_agent._cleanup_async_components()
    
    # Example 3: Legacy wrapper
    await example_legacy_wrapper()
    
    # Example 4: Advanced features (skip if no Ollama)
    try:
        await example_advanced_features()
    except Exception as e:
        logger.info(f"Skipping advanced features (Ollama not available): {e}")
    
    logger.info("\n=== Demo Complete ===")


# Utility functions for production use
def create_production_agent(agent_name: str, config_overrides: dict = None):
    """
    Create production-ready agent with custom configuration
    """
    try:
        agent = create_agent_factory(agent_name)
        
        # Apply configuration overrides
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(agent, key, value)
        
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create production agent {agent_name}: {e}")
        return None


async def run_agent_with_monitoring(agent_name: str, duration: int = 3600):
    """
    Run agent with comprehensive monitoring for specified duration
    """
    agent = create_production_agent(agent_name)
    if not agent:
        return False
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Start monitoring task
        async def monitor():
            while True:
                health = await agent.health_check()
                logger.info(f"Health: {health.get('healthy', False)}, "
                          f"Tasks: {health.get('tasks_processed', 0)}")
                await asyncio.sleep(60)  # Monitor every minute
        
        monitor_task = asyncio.create_task(monitor())
        
        # Run agent with timeout
        await asyncio.wait_for(agent.run_async(), timeout=duration)
        
    except asyncio.TimeoutError:
        logger.info(f"Agent {agent_name} completed scheduled run ({duration}s)")
    except Exception as e:
        logger.error(f"Agent {agent_name} runtime error: {e}")
        return False
    finally:
        monitor_task.cancel()
    
    return True


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())