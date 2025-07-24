#!/usr/bin/env python3
"""
SutazAI Central AI Orchestrator
Cross-references all AI components and provides unified communication
Implements self-improving code generation with safety checks
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import requests
from datetime import datetime, timezone
import subprocess
import os
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

# Import existing components
from .ai_agents.agent_framework import AgentFramework
from .ai_agents.model_manager import ModelManager
from .vector_db import VectorDatabase
from .neuromorphic.enhanced_engine import EnhancedNeuromorphicEngine
from .self_improvement.autonomous_code_generator import AutonomousCodeGenerator

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class AIService:
    name: str
    url: str
    port: int
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    response_time: Optional[float] = None
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskRequest:
    task_id: str
    task_type: str
    content: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    requires_approval: bool = False
    safety_level: str = "normal"

@dataclass
class TaskResult:
    task_id: str
    status: str
    result: Any
    execution_time: float
    service_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class SutazAIOrchestrator:
    """
    Central orchestrator that manages all AI services and provides
    cross-component communication and self-improvement capabilities
    """
    
    def __init__(self):
        self.services: Dict[str, AIService] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.health_check_interval = 30
        self.running = False
        self.task_queue = asyncio.Queue()
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.task_history: List[TaskResult] = []
        
        # Cross-referenced components
        self.agent_framework = None
        self.model_manager = None
        self.vector_db = None
        self.neuromorphic_engine = None
        self.code_generator = None
        
        # Self-improvement tracking
        self.improvement_suggestions = []
        self.code_improvements = {}
        self.performance_metrics = {}
        
        self._initialize_services()
        
    def _initialize_services(self):
        """Initialize all AI services with cross-references"""
        
        # Core Infrastructure
        self.services.update({
            "ollama": AIService(
                name="Ollama Model Server",
                url="http://ollama:11434",
                port=11434,
                capabilities=["text_generation", "code_generation", "chat"],
                metadata={"models": ["deepseek-r1:8b", "qwen3:8b", "deepseek-coder:33b", "llama3.2:1b"]}
            ),
            "enhanced_model_manager": AIService(
                name="Enhanced Model Manager",
                url="http://enhanced-model-manager:8000",
                port=8000,
                capabilities=["model_management", "quantization", "optimization"],
                dependencies=["ollama"]
            ),
            
            # Vector Databases
            "qdrant": AIService(
                name="Qdrant Vector Database",
                url="http://qdrant:6333",
                port=6333,
                capabilities=["vector_search", "similarity", "clustering"]
            ),
            "chromadb": AIService(
                name="ChromaDB Vector Store", 
                url="http://chromadb:8000",
                port=8000,
                capabilities=["embeddings", "vector_storage", "semantic_search"]
            ),
            "faiss": AIService(
                name="FAISS Vector Search",
                url="http://faiss:8000", 
                port=8000,
                capabilities=["fast_similarity", "indexing", "retrieval"]
            ),
            
            # AI Agents
            "autogpt": AIService(
                name="AutoGPT Agent",
                url="http://autogpt:8000",
                port=8000,
                capabilities=["task_automation", "goal_decomposition", "autonomous_execution"],
                dependencies=["ollama", "qdrant"]
            ),
            "localagi": AIService(
                name="LocalAGI Orchestrator",
                url="http://localagi:8080",
                port=8080,
                capabilities=["agi_coordination", "multi_agent", "local_inference"]
            ),
            "tabbyml": AIService(
                name="TabbyML Code Assistant",
                url="http://tabbyml:8080", 
                port=8080,
                capabilities=["code_completion", "code_analysis", "suggestions"],
                dependencies=["ollama"]
            ),
            "semgrep": AIService(
                name="Semgrep Security Scanner",
                url="http://semgrep:8000",
                port=8000,
                capabilities=["security_analysis", "vulnerability_detection", "code_audit"]
            ),
            
            # Agent Frameworks
            "langchain_agents": AIService(
                name="LangChain Agents",
                url="http://langchain-agents:8000",
                port=8000,
                capabilities=["agent_orchestration", "chain_building", "workflow"],
                dependencies=["ollama"]
            ),
            "autogen_agents": AIService(
                name="AutoGen Multi-Agent System",
                url="http://autogen-agents:8000",
                port=8000,
                capabilities=["multi_agent_chat", "collaborative_problem_solving", "agent_coordination"],
                dependencies=["ollama"]
            ),
            "agentzero": AIService(
                name="AgentZero Framework",
                url="http://agentzero:8000",
                port=8000,
                capabilities=["autonomous_agent", "task_execution", "zero_shot_learning"],
                dependencies=["ollama", "qdrant"]
            ),
            "bigagi": AIService(
                name="BigAGI Interface",
                url="http://bigagi:3000",
                port=3000,
                capabilities=["advanced_ui", "conversation", "multi_modal"],
                dependencies=["ollama"]
            ),
            
            # Web Automation
            "browser_use": AIService(
                name="Browser Use Automation",
                url="http://browser-use:8000",
                port=8000,
                capabilities=["web_automation", "browser_control", "data_extraction"],
                dependencies=["ollama"]
            ),
            "skyvern": AIService(
                name="Skyvern Web Agent",
                url="http://skyvern:8000",
                port=8000,
                capabilities=["web_scraping", "form_filling", "automation"],
                dependencies=["postgres", "redis", "ollama"]
            ),
            
            # Document & Financial Processing
            "documind": AIService(
                name="Documind Document AI",
                url="http://documind:8000",
                port=8000,
                capabilities=["document_processing", "pdf_analysis", "text_extraction"],
                dependencies=["ollama"]
            ),
            "finrobot": AIService(
                name="FinRobot Financial AI",
                url="http://finrobot:8000",
                port=8000,
                capabilities=["financial_analysis", "trading", "market_data"],
                dependencies=["postgres", "redis", "ollama"]
            ),
            
            # Code Generation & Development
            "gpt_engineer": AIService(
                name="GPT Engineer Code Generator", 
                url="http://gpt-engineer:8000",
                port=8000,
                capabilities=["code_generation", "project_creation", "architecture"],
                dependencies=["ollama"]
            ),
            "aider": AIService(
                name="Aider AI Code Editor",
                url="http://aider:8000",
                port=8000,
                capabilities=["code_editing", "refactoring", "git_integration"],
                dependencies=["ollama"]
            ),
            
            # UI & Workflow Platforms
            "open_webui": AIService(
                name="OpenWebUI Chat Interface",
                url="http://open-webui:8080",
                port=8080,
                capabilities=["chat_interface", "conversation", "multi_model"],
                dependencies=["ollama"]
            ),
            "langflow": AIService(
                name="LangFlow Visual Builder",
                url="http://langflow:7860",
                port=7860,
                capabilities=["visual_workflow", "flow_builder", "no_code"],
                dependencies=["ollama"]
            ),
            "dify": AIService(
                name="Dify LLM Platform",
                url="http://dify:5001",
                port=5001,
                capabilities=["llm_apps", "workflow", "api_creation"],
                dependencies=["postgres", "redis", "ollama"]
            ),
            
            # ML Frameworks
            "pytorch": AIService(
                name="PyTorch ML Framework",
                url="http://pytorch:8000",
                port=8000,
                capabilities=["deep_learning", "neural_networks", "training"]
            ),
            "tensorflow": AIService(
                name="TensorFlow ML Platform",
                url="http://tensorflow:8000",
                port=8000,
                capabilities=["machine_learning", "production_ml", "deployment"]
            ),
            "jax": AIService(
                name="JAX Research Framework",
                url="http://jax:8000",
                port=8000,
                capabilities=["research", "high_performance", "scientific_computing"]
            ),
            
            # Advanced AI Systems
            "awesome_code_ai": AIService(
                name="Awesome Code AI Tools",
                url="http://awesome-code-ai:8000",
                port=8000,
                capabilities=["code_analysis", "best_practices", "tooling"],
                dependencies=["ollama"]
            ),
            "neuromorphic_engine": AIService(
                name="Neuromorphic Computing Engine",
                url="http://neuromorphic-engine:8000",
                port=8000,
                capabilities=["spiking_networks", "biological_modeling", "stdp_learning"]
            ),
            "self_improvement_engine": AIService(
                name="Self-Improvement Engine",
                url="http://self-improvement-engine:8000",
                port=8000,
                capabilities=["code_generation", "self_modification", "improvement_suggestions"],
                dependencies=["ollama", "gpt_engineer", "aider"]
            ),
            "web_learning_engine": AIService(
                name="Web Learning Engine",
                url="http://web-learning-engine:8000",
                port=8000,
                capabilities=["web_scraping", "knowledge_extraction", "continuous_learning"],
                dependencies=["browser_use", "ollama"]
            )
        })
        
    async def initialize(self):
        """Initialize the orchestrator and all components"""
        try:
            logger.info("Initializing SutazAI Orchestrator...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Initialize cross-referenced components
            await self._initialize_components()
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self._health_monitoring_loop())
            asyncio.create_task(self._task_processing_loop())
            asyncio.create_task(self._self_improvement_loop())
            
            logger.info("SutazAI Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
            
    async def _initialize_components(self):
        """Initialize and cross-reference all AI components"""
        try:
            # Initialize model manager with cross-references
            self.model_manager = ModelManager()
            await self.model_manager.initialize()
            
            # Initialize agent framework with model manager reference
            self.agent_framework = AgentFramework(model_manager=self.model_manager)
            await self.agent_framework.initialize()
            
            # Initialize vector database with agent framework reference
            self.vector_db = VectorDatabase(agent_framework=self.agent_framework)
            await self.vector_db.initialize()
            
            # Initialize neuromorphic engine with vector DB reference
            self.neuromorphic_engine = EnhancedNeuromorphicEngine(
                vector_db=self.vector_db,
                model_manager=self.model_manager
            )
            await self.neuromorphic_engine.initialize()
            
            # Initialize code generator with all component references
            self.code_generator = AutonomousCodeGenerator(
                model_manager=self.model_manager,
                agent_framework=self.agent_framework,
                vector_db=self.vector_db,
                orchestrator=self
            )
            await self.code_generator.initialize()
            
            logger.info("All components cross-referenced and initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
            
    async def check_service_health(self, service_name: str) -> ServiceStatus:
        """Check health of a specific service"""
        if service_name not in self.services:
            return ServiceStatus.UNKNOWN
            
        service = self.services[service_name]
        start_time = time.time()
        
        try:
            # Check service dependencies first
            for dep in service.dependencies:
                dep_status = await self.check_service_health(dep)
                if dep_status == ServiceStatus.FAILED:
                    service.status = ServiceStatus.DEGRADED
                    return ServiceStatus.DEGRADED
            
            # Health check endpoint
            health_url = f"{service.url}/health"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    service.status = ServiceStatus.HEALTHY
                    service.response_time = time.time() - start_time
                    service.last_health_check = datetime.now(timezone.utc)
                    return ServiceStatus.HEALTHY
                else:
                    service.status = ServiceStatus.DEGRADED
                    return ServiceStatus.DEGRADED
                    
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            service.status = ServiceStatus.FAILED
            return ServiceStatus.FAILED
            
    async def _health_monitoring_loop(self):
        """Continuous health monitoring of all services"""
        while self.running:
            try:
                logger.debug("Performing health checks...")
                
                # Check all services in parallel
                health_tasks = [
                    self.check_service_health(name) 
                    for name in self.services.keys()
                ]
                
                results = await asyncio.gather(*health_tasks, return_exceptions=True)
                
                # Update performance metrics
                healthy_count = sum(1 for r in results if r == ServiceStatus.HEALTHY)
                total_count = len(self.services)
                
                self.performance_metrics.update({
                    "health_check_timestamp": datetime.now(timezone.utc),
                    "healthy_services": healthy_count,
                    "total_services": total_count,
                    "health_percentage": (healthy_count / total_count) * 100,
                    "failed_services": [
                        name for name, status in zip(self.services.keys(), results)
                        if status == ServiceStatus.FAILED
                    ]
                })
                
                logger.debug(f"Health check completed: {healthy_count}/{total_count} services healthy")
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
            await asyncio.sleep(self.health_check_interval)
            
    async def execute_task(self, task: TaskRequest) -> TaskResult:
        """Execute a task using the most appropriate AI service"""
        start_time = time.time()
        
        try:
            logger.info(f"Executing task {task.task_id}: {task.task_type}")
            
            # Determine best service for task
            best_service = await self._select_best_service(task)
            if not best_service:
                raise ValueError("No suitable service found for task")
            
            # Execute task
            result = await self._execute_on_service(task, best_service)
            
            # Create result
            task_result = TaskResult(
                task_id=task.task_id,
                status="completed",
                result=result,
                execution_time=time.time() - start_time,
                service_used=best_service,
                metadata={
                    "task_type": task.task_type,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Store in history
            self.task_history.append(task_result)
            
            # Check for improvement opportunities
            await self._analyze_for_improvements(task, task_result)
            
            return task_result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                result=str(e),
                execution_time=time.time() - start_time,
                service_used="none"
            )
            
    async def _select_best_service(self, task: TaskRequest) -> Optional[str]:
        """Select the best service for a given task based on capabilities and health"""
        
        # Define task type to service mappings
        task_mappings = {
            "text_generation": ["ollama", "enhanced_model_manager"],
            "code_generation": ["gpt_engineer", "aider", "tabbyml", "ollama"],
            "code_analysis": ["semgrep", "tabbyml", "awesome_code_ai"],
            "web_automation": ["browser_use", "skyvern"],
            "document_processing": ["documind"],
            "financial_analysis": ["finrobot"],
            "vector_search": ["qdrant", "chromadb", "faiss"],
            "agent_orchestration": ["langchain_agents", "autogen_agents", "autogpt"],
            "self_improvement": ["self_improvement_engine", "gpt_engineer"],
            "web_learning": ["web_learning_engine", "browser_use"],
            "neuromorphic": ["neuromorphic_engine"]
        }
        
        # Get candidate services
        candidates = task_mappings.get(task.task_type, [])
        if not candidates:
            # Fallback to general-purpose services
            candidates = ["ollama", "enhanced_model_manager"]
        
        # Filter by health status and availability
        healthy_candidates = [
            name for name in candidates
            if name in self.services and 
            self.services[name].status == ServiceStatus.HEALTHY
        ]
        
        if not healthy_candidates:
            # Try degraded services as fallback
            healthy_candidates = [
                name for name in candidates
                if name in self.services and 
                self.services[name].status == ServiceStatus.DEGRADED
            ]
        
        if healthy_candidates:
            # Select service with best response time
            best_service = min(
                healthy_candidates,
                key=lambda x: self.services[x].response_time or float('inf')
            )
            return best_service
            
        return None
        
    async def _execute_on_service(self, task: TaskRequest, service_name: str) -> Any:
        """Execute task on a specific service"""
        service = self.services[service_name]
        
        # Prepare request based on service type
        if service_name == "ollama":
            return await self._execute_ollama_task(task, service)
        elif service_name == "gpt_engineer":
            return await self._execute_code_generation_task(task, service)
        elif service_name in ["qdrant", "chromadb", "faiss"]:
            return await self._execute_vector_task(task, service)
        elif service_name in ["autogpt", "langchain_agents", "autogen_agents"]:
            return await self._execute_agent_task(task, service)
        else:
            return await self._execute_generic_task(task, service)
            
    async def _execute_ollama_task(self, task: TaskRequest, service: AIService) -> Any:
        """Execute task on Ollama service"""
        payload = {
            "model": task.parameters.get("model", "deepseek-r1:8b"),
            "prompt": task.content,
            "stream": False,
            "options": task.parameters.get("options", {})
        }
        
        async with self.session.post(f"{service.url}/api/generate", json=payload) as response:
            result = await response.json()
            return result.get("response", "")
            
    async def _execute_code_generation_task(self, task: TaskRequest, service: AIService) -> Any:
        """Execute code generation task"""
        payload = {
            "prompt": task.content,
            "language": task.parameters.get("language", "python"),
            "project_type": task.parameters.get("project_type", "script"),
            "requirements": task.parameters.get("requirements", [])
        }
        
        async with self.session.post(f"{service.url}/generate", json=payload) as response:
            return await response.json()
            
    async def _execute_vector_task(self, task: TaskRequest, service: AIService) -> Any:
        """Execute vector database task"""
        if task.task_type == "vector_search":
            payload = {
                "query": task.content,
                "limit": task.parameters.get("limit", 10),
                "threshold": task.parameters.get("threshold", 0.7)
            }
            
            async with self.session.post(f"{service.url}/search", json=payload) as response:
                return await response.json()
        else:
            return {"error": "Unsupported vector task type"}
            
    async def _execute_agent_task(self, task: TaskRequest, service: AIService) -> Any:
        """Execute agent orchestration task"""
        payload = {
            "task": task.content,
            "parameters": task.parameters,
            "agent_type": task.parameters.get("agent_type", "general")
        }
        
        async with self.session.post(f"{service.url}/execute", json=payload) as response:
            return await response.json()
            
    async def _execute_generic_task(self, task: TaskRequest, service: AIService) -> Any:
        """Execute generic task on service"""
        payload = {
            "input": task.content,
            "parameters": task.parameters
        }
        
        async with self.session.post(f"{service.url}/process", json=payload) as response:
            return await response.json()
            
    async def _task_processing_loop(self):
        """Process tasks from the queue"""
        while self.running:
            try:
                # Wait for task or timeout
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Add to active tasks
                self.active_tasks[task.task_id] = task
                
                # Execute task
                result = await self.execute_task(task)
                
                # Remove from active tasks
                self.active_tasks.pop(task.task_id, None)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Task processing error: {e}")
                
    async def _self_improvement_loop(self):
        """Continuous self-improvement monitoring"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                if self.code_generator:
                    # Analyze recent tasks for improvement opportunities
                    recent_tasks = self.task_history[-100:]  # Last 100 tasks
                    
                    # Look for patterns that could be optimized
                    improvements = await self.code_generator.analyze_performance_patterns(recent_tasks)
                    
                    for improvement in improvements:
                        if improvement.get("requires_approval", True):
                            logger.info(f"Self-improvement suggestion: {improvement['description']}")
                            self.improvement_suggestions.append(improvement)
                        else:
                            # Auto-apply safe improvements
                            await self._apply_improvement(improvement)
                            
            except Exception as e:
                logger.error(f"Self-improvement loop error: {e}")
                
    async def _analyze_for_improvements(self, task: TaskRequest, result: TaskResult):
        """Analyze task execution for potential improvements"""
        try:
            # Performance analysis
            if result.execution_time > 10.0:  # Slow execution
                improvement = {
                    "type": "performance",
                    "description": f"Task {task.task_type} took {result.execution_time:.2f}s",
                    "suggestion": "Consider optimizing or using different service",
                    "task_type": task.task_type,
                    "service_used": result.service_used,
                    "requires_approval": False
                }
                self.improvement_suggestions.append(improvement)
            
            # Error pattern analysis
            if result.status == "failed":
                improvement = {
                    "type": "error_handling",
                    "description": f"Task {task.task_type} failed: {result.result}",
                    "suggestion": "Implement better error handling or fallback",
                    "task_type": task.task_type,
                    "requires_approval": True
                }
                self.improvement_suggestions.append(improvement)
                
        except Exception as e:
            logger.error(f"Improvement analysis error: {e}")
            
    async def _apply_improvement(self, improvement: Dict[str, Any]):
        """Apply an improvement suggestion"""
        try:
            logger.info(f"Applying improvement: {improvement['description']}")
            
            if improvement["type"] == "performance":
                # Optimize service selection
                await self._optimize_service_selection(improvement)
            elif improvement["type"] == "error_handling":
                # Improve error handling
                await self._improve_error_handling(improvement)
            elif improvement["type"] == "code_optimization":
                # Generate code improvements
                await self.code_generator.generate_code_improvement(improvement)
                
            logger.info("Improvement applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
            
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                name: {
                    "status": service.status.value,
                    "response_time": service.response_time,
                    "last_health_check": service.last_health_check.isoformat() if service.last_health_check else None,
                    "capabilities": service.capabilities,
                    "dependencies": service.dependencies
                }
                for name, service in self.services.items()
            },
            "performance_metrics": self.performance_metrics,
            "active_tasks": len(self.active_tasks),
            "task_history_count": len(self.task_history),
            "improvement_suggestions": len(self.improvement_suggestions),
            "components": {
                "model_manager": self.model_manager is not None,
                "agent_framework": self.agent_framework is not None,
                "vector_db": self.vector_db is not None,
                "neuromorphic_engine": self.neuromorphic_engine is not None,
                "code_generator": self.code_generator is not None
            }
        }
        
    async def submit_task(self, task: TaskRequest) -> str:
        """Submit a task for execution"""
        await self.task_queue.put(task)
        return task.task_id
        
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed task"""
        for result in self.task_history:
            if result.task_id == task_id:
                return result
        return None
        
    async def shutdown(self):
        """Shutdown the orchestrator"""
        self.running = False
        
        if self.session:
            await self.session.close()
            
        logger.info("SutazAI Orchestrator shutdown complete")

# Global orchestrator instance
sutazai_orchestrator = SutazAIOrchestrator()