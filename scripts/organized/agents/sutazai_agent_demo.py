"""
SutazAI Agent System Demo - Complete Usage Example
================================================

This comprehensive demo shows how to use all available AI agents in the SutazAI system,
including agent communication through Redis, task orchestration, and common workflows.

Features demonstrated:
- Agent instantiation for all agent types
- Redis-based messaging and communication
- Task execution and result handling
- Agent discovery and capability querying
- Error handling and best practices
- Real-world workflow examples

Usage:
    python sutazai_agent_demo.py

Requirements:
    - Redis server running on localhost:6379
    - Ollama server running on localhost:11434
    - Python packages: redis, aioredis, httpx, asyncio
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import redis
import aioredis
import httpx

# Import SutazAI components
from backend.ai_agents.core.base_agent import (
    BaseAgent, AgentConfig, AgentCapability, AgentStatus, AgentMessage
)
from backend.ai_agents.specialized.code_generator import CodeGeneratorAgent
from backend.ai_agents.communication.agent_bus import (
    AgentCommunicationBus, MessageType, MessagePriority
)
from backend.app.agents.registry import AgentRegistry


class SutazAIAgentDemo:
    """
    Comprehensive demonstration of SutazAI's AI agent system
    
    This class shows how to:
    1. Initialize and configure all available agents
    2. Establish Redis communication
    3. Execute common workflows
    4. Handle errors gracefully
    5. Monitor system performance
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize the demo system
        
        Args:
            redis_url: Redis connection URL for messaging
            ollama_url: Ollama server URL for local AI models
        """
        self.redis_url = redis_url
        self.ollama_url = ollama_url
        
        # Core components
        self.communication_bus: Optional[AgentCommunicationBus] = None
        self.agent_registry: Optional[AgentRegistry] = None
        self.active_agents: Dict[str, BaseAgent] = {}
        
        # Redis connections
        self.redis_client: Optional[redis.Redis] = None
        self.async_redis: Optional[aioredis.Redis] = None
        
        # Demo state
        self.demo_session_id = str(uuid.uuid4())
        self.task_results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("sutazai.demo")
        
        # Available agent configurations
        self.agent_configs = self._get_agent_configurations()
    
    def _get_agent_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all available agents"""
        return {
            "code_generator": {
                "agent_type": "CodeGeneratorAgent",
                "name": "Code Generator",
                "description": "Specialized agent for code generation, completion, and refactoring",
                "capabilities": [
                    AgentCapability.CODE_GENERATION,
                    AgentCapability.CODE_ANALYSIS,
                    AgentCapability.REASONING
                ],
                "model": "codellama"
            },
            "security_analyzer": {
                "agent_type": "SecurityAnalyzerAgent", 
                "name": "Security Analyzer",
                "description": "Semgrep-based security vulnerability scanner",
                "capabilities": [
                    AgentCapability.SECURITY_ANALYSIS,
                    AgentCapability.CODE_ANALYSIS,
                    AgentCapability.TESTING
                ],
                "model": "llama2"
            },
            "opendevin_generator": {
                "agent_type": "OpenDevinAgent",
                "name": "OpenDevin Code Generator", 
                "description": "Autonomous software engineering agent",
                "capabilities": [
                    AgentCapability.CODE_GENERATION,
                    AgentCapability.AUTONOMOUS_EXECUTION,
                    AgentCapability.TESTING,
                    AgentCapability.DEPLOYMENT
                ],
                "model": "codellama"
            },
            "dify_automation": {
                "agent_type": "DifyAutomationAgent",
                "name": "Dify Automation Specialist",
                "description": "AI-powered workflow automation agent",
                "capabilities": [
                    AgentCapability.AUTOMATION,
                    AgentCapability.API_INTEGRATION,
                    AgentCapability.DATA_PROCESSING
                ],
                "model": "llama2"
            },
            "agentgpt_executor": {
                "agent_type": "AgentGPTExecutor",
                "name": "AgentGPT Autonomous Executor",
                "description": "Goal-driven autonomous task executor",
                "capabilities": [
                    AgentCapability.AUTONOMOUS_EXECUTION,
                    AgentCapability.REASONING,
                    AgentCapability.ORCHESTRATION
                ],
                "model": "llama2"
            },
            "langflow_designer": {
                "agent_type": "LangflowDesigner",
                "name": "Langflow Workflow Designer",
                "description": "Visual AI workflow creation agent",
                "capabilities": [
                    AgentCapability.AUTOMATION,
                    AgentCapability.API_INTEGRATION,
                    AgentCapability.CODE_GENERATION
                ],
                "model": "llama2"
            },
            "localagi_orchestrator": {
                "agent_type": "LocalAGIOrchestrator",
                "name": "LocalAGI Orchestration Manager",
                "description": "Multi-agent orchestration and coordination",
                "capabilities": [
                    AgentCapability.ORCHESTRATION,
                    AgentCapability.AUTONOMOUS_EXECUTION,
                    AgentCapability.COMMUNICATION
                ],
                "model": "llama2"
            },
            "bigagi_manager": {
                "agent_type": "BigAGIManager",
                "name": "BigAGI System Manager",
                "description": "Advanced conversational AI interface manager",
                "capabilities": [
                    AgentCapability.COMMUNICATION,
                    AgentCapability.API_INTEGRATION,
                    AgentCapability.REASONING
                ],
                "model": "llama2"
            },
            "flowiseai_manager": {
                "agent_type": "FlowiseAIManager",
                "name": "FlowiseAI Flow Manager",
                "description": "Visual LangChain application builder",
                "capabilities": [
                    AgentCapability.AUTOMATION,
                    AgentCapability.CODE_GENERATION,
                    AgentCapability.API_INTEGRATION
                ],
                "model": "llama2"
            },
            "agentzero_coordinator": {
                "agent_type": "AgentZeroCoordinator",
                "name": "AgentZero Coordinator",
                "description": "General-purpose adaptive agent coordinator",
                "capabilities": [
                    AgentCapability.AUTONOMOUS_EXECUTION,
                    AgentCapability.LEARNING,
                    AgentCapability.REASONING
                ],
                "model": "llama2"
            }
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the complete demo system
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing SutazAI Agent Demo System")
            
            # Initialize Redis connections
            await self._initialize_redis()
            
            # Initialize communication bus
            await self._initialize_communication_bus()
            
            # Initialize agent registry
            await self._initialize_agent_registry()
            
            # Verify system dependencies
            await self._verify_dependencies()
            
            self.logger.info("SutazAI Agent Demo System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize demo system: {e}")
            return False
    
    async def _initialize_redis(self):
        """Initialize Redis connections"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.async_redis = aioredis.from_url(self.redis_url, decode_responses=True)
        
        # Test connections
        await self.async_redis.ping()
        self.redis_client.ping()
        
        self.logger.info("Redis connections established")
    
    async def _initialize_communication_bus(self):
        """Initialize the agent communication bus"""
        self.communication_bus = AgentCommunicationBus(self.redis_url)
        await self.communication_bus.initialize()
        self.logger.info("Agent communication bus initialized")
    
    async def _initialize_agent_registry(self):
        """Initialize the agent registry"""
        self.agent_registry = AgentRegistry()
        self.logger.info("Agent registry initialized")
    
    async def _verify_dependencies(self):
        """Verify system dependencies are available"""
        # Check Ollama server
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    models = [model["name"] for model in response.json().get("models", [])]
                    self.logger.info(f"Ollama server available with models: {models}")
                else:
                    self.logger.warning("Ollama server not responding properly")
        except Exception as e:
            self.logger.warning(f"Could not connect to Ollama server: {e}")
    
    async def create_all_agents(self) -> Dict[str, BaseAgent]:
        """
        Create and initialize all available agents
        
        Returns:
            Dictionary of agent instances by agent ID
        """
        self.logger.info("Creating all available agents")
        created_agents = {}
        
        for agent_id, config_data in self.agent_configs.items():
            try:
                agent = await self._create_agent(agent_id, config_data)
                if agent:
                    created_agents[agent_id] = agent
                    self.active_agents[agent_id] = agent
                    self.logger.info(f"Created and initialized agent: {agent_id}")
                else:
                    self.logger.error(f"Failed to create agent: {agent_id}")
                    
            except Exception as e:
                self.logger.error(f"Error creating agent {agent_id}: {e}")
        
        self.logger.info(f"Successfully created {len(created_agents)} agents")
        return created_agents
    
    async def _create_agent(self, agent_id: str, config_data: Dict[str, Any]) -> Optional[BaseAgent]:
        """Create a single agent instance"""
        try:
            # Create agent configuration
            agent_config = AgentConfig(
                agent_id=f"{agent_id}_{self.demo_session_id}",
                agent_type=config_data["agent_type"],
                name=config_data["name"],
                description=config_data["description"],
                capabilities=config_data["capabilities"],
                model_config={
                    "ollama_url": self.ollama_url,
                    "model": config_data["model"]
                },
                redis_config={
                    "url": self.redis_url
                },
                max_concurrent_tasks=3,
                heartbeat_interval=30,
                message_timeout=300
            )
            
            # Create agent instance based on type
            if config_data["agent_type"] == "CodeGeneratorAgent":
                agent = CodeGeneratorAgent(agent_config)
            else:
                # For demo purposes, create a generic agent
                # In production, you'd have specific classes for each agent type
                agent = DemoGenericAgent(agent_config, config_data)
            
            # Initialize the agent
            success = await agent.initialize()
            if success:
                # Register with communication bus and registry
                await self.communication_bus.register_agent(
                    agent.agent_id,
                    {cap.value for cap in agent.capabilities},
                    current_load=0.0
                )
                
                self.agent_registry.register_agent(
                    agent.agent_id,
                    agent.name,
                    list(agent.capabilities),
                    metadata={"demo_session": self.demo_session_id}
                )
                
                return agent
            else:
                self.logger.error(f"Failed to initialize agent: {agent_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating agent {agent_id}: {e}")
            return None
    
    async def demonstrate_agent_communication(self):
        """Demonstrate inter-agent communication patterns"""
        self.logger.info("=== Demonstrating Agent Communication ===")
        
        if len(self.active_agents) < 2:
            self.logger.warning("Need at least 2 agents for communication demo")
            return
        
        agent_ids = list(self.active_agents.keys())
        sender_id = agent_ids[0]
        receiver_id = agent_ids[1]
        
        sender = self.active_agents[sender_id]
        
        # 1. Direct messaging
        self.logger.info(f"Sending direct message from {sender_id} to {receiver_id}")
        message_id = await sender.send_message(
            receiver_id,
            "greeting",
            {
                "message": "Hello from the demo system!",
                "timestamp": datetime.utcnow().isoformat(),
                "demo_session": self.demo_session_id
            }
        )
        
        # 2. Broadcast messaging
        self.logger.info("Sending broadcast message to all agents")
        await sender.send_message(
            "broadcast",
            "system_announcement",
            {
                "announcement": "Demo system is running!",
                "sender": sender_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # 3. Capability query
        self.logger.info("Querying agent capabilities")
        await sender.send_message(
            receiver_id,
            "capabilities",
            {"requested_by": sender_id}
        )
        
        # 4. Status check
        self.logger.info("Checking agent status")
        await sender.send_message(
            receiver_id,
            "status",
            {"requested_by": sender_id}
        )
        
        # Wait for message processing
        await asyncio.sleep(2)
        
        self.logger.info("Agent communication demonstration completed")
    
    async def demonstrate_task_execution(self):
        """Demonstrate task execution workflows"""
        self.logger.info("=== Demonstrating Task Execution ===")
        
        # Find agents with specific capabilities
        code_agents = [
            agent_id for agent_id, agent in self.active_agents.items()
            if AgentCapability.CODE_GENERATION in agent.capabilities
        ]
        
        if not code_agents:
            self.logger.warning("No code generation agents available")
            return
        
        code_agent = self.active_agents[code_agents[0]]
        
        # 1. Simple code generation task
        self.logger.info("Executing code generation task")
        task_data = {
            "task_type": "generate_code",
            "specification": "Create a Python function that calculates the factorial of a number",
            "language": "python",
            "code_type": "function"
        }
        
        task_id = f"demo_task_{uuid.uuid4()}"
        result = await code_agent.execute_task(task_id, task_data)
        self.task_results[task_id] = result
        
        if result.get("success"):
            self.logger.info("Code generation task completed successfully")
            generated_code = result.get("result", {}).get("generated_code", "")
            self.logger.info(f"Generated code preview: {generated_code[:200]}...")
        else:
            self.logger.error(f"Code generation task failed: {result.get('error')}")
        
        # 2. Code explanation task
        if result.get("success"):
            self.logger.info("Executing code explanation task")
            explanation_task_data = {
                "task_type": "explain_code",
                "code": generated_code,
                "language": "python",
                "explanation_level": "intermediate"
            }
            
            explanation_task_id = f"demo_explanation_{uuid.uuid4()}"
            explanation_result = await code_agent.execute_task(
                explanation_task_id, 
                explanation_task_data
            )
            self.task_results[explanation_task_id] = explanation_result
            
            if explanation_result.get("success"):
                self.logger.info("Code explanation task completed successfully")
            else:
                self.logger.error(f"Code explanation failed: {explanation_result.get('error')}")
    
    async def demonstrate_collaborative_workflow(self):
        """Demonstrate multi-agent collaborative workflows"""
        self.logger.info("=== Demonstrating Collaborative Workflow ===")
        
        # Define a complex task requiring multiple agent types
        complex_task = {
            "project_name": "Demo Web API",
            "requirements": {
                "generate_api": {
                    "description": "Create a REST API for user management",
                    "language": "python",
                    "framework": "FastAPI"
                },
                "security_scan": {
                    "description": "Scan the generated code for security vulnerabilities",
                    "focus": "authentication and authorization"
                },
                "documentation": {
                    "description": "Generate API documentation",
                    "format": "OpenAPI/Swagger"
                }
            }
        }
        
        # Find suitable agents for each task
        agent_assignments = {}
        
        # Code generation
        code_agents = [
            agent_id for agent_id, agent in self.active_agents.items()
            if AgentCapability.CODE_GENERATION in agent.capabilities
        ]
        if code_agents:
            agent_assignments["code_generation"] = code_agents[0]
        
        # Security analysis
        security_agents = [
            agent_id for agent_id, agent in self.active_agents.items()
            if AgentCapability.SECURITY_ANALYSIS in agent.capabilities
        ]
        if security_agents:
            agent_assignments["security_analysis"] = security_agents[0]
        
        # If we have suitable agents, execute the workflow
        if len(agent_assignments) >= 1:
            self.logger.info(f"Executing collaborative workflow with agents: {agent_assignments}")
            
            workflow_results = {}
            
            # Step 1: Generate code
            if "code_generation" in agent_assignments:
                code_agent = self.active_agents[agent_assignments["code_generation"]]
                task_data = {
                    "task_type": "generate_code",
                    "specification": complex_task["requirements"]["generate_api"]["description"],
                    "language": "python",
                    "code_type": "api"
                }
                
                result = await code_agent.execute_task("collaborative_code_gen", task_data)
                workflow_results["code_generation"] = result
                
                # Step 2: Security analysis (if we have a security agent and code was generated)
                if "security_analysis" in agent_assignments and result.get("success"):
                    security_agent = self.active_agents[agent_assignments["security_analysis"]]
                    generated_code = result.get("result", {}).get("generated_code", "")
                    
                    security_task = {
                        "task_type": "security_scan",
                        "code": generated_code,
                        "language": "python",
                        "focus": "authentication"
                    }
                    
                    security_result = await security_agent.execute_task(
                        "collaborative_security_scan", 
                        security_task
                    )
                    workflow_results["security_analysis"] = security_result
            
            # Store workflow results
            self.task_results["collaborative_workflow"] = workflow_results
            
            self.logger.info("Collaborative workflow demonstration completed")
        else:
            self.logger.warning("Insufficient agents for collaborative workflow demonstration")
    
    async def demonstrate_system_monitoring(self):
        """Demonstrate system monitoring and metrics collection"""
        self.logger.info("=== Demonstrating System Monitoring ===")
        
        # Collect agent registry statistics
        registry_stats = self.agent_registry.get_registry_stats()
        self.logger.info(f"Agent Registry Stats: {registry_stats}")
        
        # Collect communication bus metrics
        bus_metrics = await self.communication_bus.get_system_metrics()
        self.logger.info(f"Communication Bus Metrics: {bus_metrics}")
        
        # Collect individual agent status
        agent_statuses = {}
        for agent_id, agent in self.active_agents.items():
            try:
                agent_info = await agent.get_agent_info()
                agent_statuses[agent_id] = {
                    "status": agent_info["status"],
                    "active_tasks": agent_info["active_tasks"],
                    "uptime": agent_info["uptime"],
                    "task_count": agent_info["task_count"],
                    "error_count": agent_info["error_count"]
                }
            except Exception as e:
                self.logger.error(f"Error getting status for agent {agent_id}: {e}")
                agent_statuses[agent_id] = {"error": str(e)}
        
        # Store performance metrics
        self.performance_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.demo_session_id,
            "registry_stats": registry_stats,
            "bus_metrics": bus_metrics,
            "agent_statuses": agent_statuses,
            "total_tasks_executed": len(self.task_results),
            "successful_tasks": len([r for r in self.task_results.values() 
                                   if isinstance(r, dict) and r.get("success")])
        }
        
        self.logger.info("System monitoring demonstration completed")
    
    async def demonstrate_error_handling(self):
        """Demonstrate error handling and recovery patterns"""
        self.logger.info("=== Demonstrating Error Handling ===")
        
        if not self.active_agents:
            self.logger.warning("No active agents for error handling demo")
            return
        
        agent_id = list(self.active_agents.keys())[0]
        agent = self.active_agents[agent_id]
        
        # 1. Invalid task type
        self.logger.info("Testing invalid task type handling")
        try:
            result = await agent.execute_task("error_test_1", {
                "task_type": "invalid_task_type",
                "data": "test"
            })
            self.logger.info(f"Invalid task result: {result}")
        except Exception as e:
            self.logger.info(f"Caught expected error: {e}")
        
        # 2. Malformed task data
        self.logger.info("Testing malformed task data handling")
        try:
            result = await agent.execute_task("error_test_2", {
                "task_type": "generate_code",
                # Missing required fields
            })
            self.logger.info(f"Malformed task result: {result}")
        except Exception as e:
            self.logger.info(f"Caught expected error: {e}")
        
        # 3. Agent capacity overflow
        self.logger.info("Testing agent capacity limits")
        tasks = []
        for i in range(agent.config.max_concurrent_tasks + 2):
            task = asyncio.create_task(
                agent.execute_task(f"capacity_test_{i}", {
                    "task_type": "generate_code",
                    "specification": f"Simple task {i}",
                    "language": "python"
                })
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - successful
        
        self.logger.info(f"Capacity test results: {successful} successful, {failed} failed")
        
        self.logger.info("Error handling demonstration completed")
    
    async def generate_demo_report(self) -> Dict[str, Any]:
        """Generate a comprehensive demo report"""
        report = {
            "demo_session_id": self.demo_session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": {
                "redis_url": self.redis_url,
                "ollama_url": self.ollama_url,
                "total_agents_configured": len(self.agent_configs),
                "total_agents_created": len(self.active_agents)
            },
            "agent_summary": {},
            "task_results": self.task_results,
            "performance_metrics": self.performance_metrics,
            "recommendations": []
        }
        
        # Add agent summary
        for agent_id, agent in self.active_agents.items():
            try:
                agent_info = await agent.get_agent_info()
                report["agent_summary"][agent_id] = {
                    "name": agent_info["name"],
                    "type": agent_info["agent_type"],
                    "status": agent_info["status"],
                    "capabilities": agent_info["capabilities"],
                    "uptime": agent_info["uptime"],
                    "tasks_completed": agent_info["task_count"],
                    "errors": agent_info["error_count"]
                }
            except Exception as e:
                report["agent_summary"][agent_id] = {"error": str(e)}
        
        # Add recommendations
        if len(self.active_agents) < len(self.agent_configs):
            report["recommendations"].append(
                "Some agents failed to initialize. Check Ollama server and model availability."
            )
        
        if self.performance_metrics.get("successful_tasks", 0) < len(self.task_results):
            report["recommendations"].append(
                "Some tasks failed. Review task data and agent capabilities."
            )
        
        return report
    
    async def cleanup(self):
        """Clean up resources and shutdown agents"""
        self.logger.info("Cleaning up demo system")
        
        # Shutdown all agents
        for agent_id, agent in self.active_agents.items():
            try:
                await agent.shutdown()
                self.logger.info(f"Shutdown agent: {agent_id}")
            except Exception as e:
                self.logger.error(f"Error shutting down agent {agent_id}: {e}")
        
        # Shutdown communication bus
        if self.communication_bus:
            await self.communication_bus.shutdown()
        
        # Close Redis connections
        if self.async_redis:
            await self.async_redis.close()
        if self.redis_client:
            self.redis_client.close()
        
        self.logger.info("Demo system cleanup completed")
    
    async def run_complete_demo(self):
        """Run the complete demonstration workflow"""
        try:
            self.logger.info("🚀 Starting SutazAI Agent System Demo")
            
            # Initialize system
            if not await self.initialize():
                self.logger.error("Failed to initialize demo system")
                return
            
            # Create all agents
            agents = await self.create_all_agents()
            if not agents:
                self.logger.error("No agents were created successfully")
                return
            
            self.logger.info(f"✅ Created {len(agents)} agents successfully")
            
            # Run demonstrations
            await self.demonstrate_agent_communication()
            await self.demonstrate_task_execution()
            await self.demonstrate_collaborative_workflow()
            await self.demonstrate_system_monitoring()
            await self.demonstrate_error_handling()
            
            # Generate and save report
            report = await self.generate_demo_report()
            
            # Save report to file
            report_filename = f"sutazai_demo_report_{self.demo_session_id[:8]}.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"📊 Demo report saved to: {report_filename}")
            self.logger.info("🎉 SutazAI Agent System Demo completed successfully!")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Demo failed with error: {e}")
            raise
        finally:
            await self.cleanup()


class DemoGenericAgent(BaseAgent):
    """
    Generic agent implementation for demo purposes
    
    This represents agents that don't have specific implementations yet
    but can still demonstrate basic functionality.
    """
    
    def __init__(self, config: AgentConfig, agent_data: Dict[str, Any]):
        super().__init__(config)
        self.agent_data = agent_data
    
    async def on_initialize(self):
        """Initialize the generic demo agent"""
        self.logger.info(f"Initializing generic demo agent: {self.name}")
        
        # Register basic message handlers
        self.register_message_handler("demo_task", self._handle_demo_task)
        self.register_message_handler("greeting", self._handle_greeting)
        
        self.logger.info(f"Generic demo agent {self.name} initialized")
    
    async def on_task_execute(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a generic demo task"""
        task_type = task_data.get("task_type", "generic")
        
        # Simulate task processing
        await asyncio.sleep(0.5)  # Simulate work
        
        return {
            "success": True,
            "result": {
                "task_type": task_type,
                "agent_type": self.agent_type,
                "agent_capabilities": [cap.value for cap in self.capabilities],
                "processed_at": datetime.utcnow().isoformat(),
                "message": f"Task {task_type} processed by {self.name}"
            },
            "task_id": task_id
        }
    
    async def on_message_received(self, message: AgentMessage):
        """Handle unknown message types"""
        self.logger.info(f"Received message type: {message.message_type} from {message.sender_id}")
        
        # Echo back a response
        await self.send_message(
            message.sender_id,
            "message_received",
            {
                "original_type": message.message_type,
                "received_by": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def on_shutdown(self):
        """Cleanup when shutting down"""
        self.logger.info(f"Generic demo agent {self.name} shutting down")
    
    async def _handle_demo_task(self, message: AgentMessage):
        """Handle demo task message"""
        content = message.content
        
        # Process the demo task
        result = await self.execute_task(message.id, {
            "task_type": "demo_task",
            **content
        })
        
        # Send response
        await self.send_message(
            message.sender_id,
            "demo_task_result",
            result
        )
    
    async def _handle_greeting(self, message: AgentMessage):
        """Handle greeting message"""
        await self.send_message(
            message.sender_id,
            "greeting_response",
            {
                "message": f"Hello from {self.name}!",
                "agent_type": self.agent_type,
                "capabilities": [cap.value for cap in self.capabilities],
                "timestamp": datetime.utcnow().isoformat()
            }
        )


async def main():
    """Main function to run the demo"""
    demo = SutazAIAgentDemo()
    
    try:
        report = await demo.run_complete_demo()
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Session ID: {demo.demo_session_id}")
        print(f"Agents Created: {len(demo.active_agents)}")
        print(f"Tasks Executed: {len(demo.task_results)}")
        print(f"Report File: sutazai_demo_report_{demo.demo_session_id[:8]}.json")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        await demo.cleanup()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        await demo.cleanup()
        raise


if __name__ == "__main__":
    asyncio.run(main())