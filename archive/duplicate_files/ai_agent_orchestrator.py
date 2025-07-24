#!/usr/bin/env python3
"""
AI Agent Orchestrator for SutazAI AGI/ASI System
High-performance multi-agent coordination and communication
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

# Enhanced logging
from enhanced_logging_system import info, debug, warning, error, critical, log_exception, log_context

@dataclass
class AgentConfig:
    name: str
    url: str
    port: int
    capabilities: List[str]
    priority: int = 1
    max_concurrent: int = 3
    timeout: int = 30
    health_endpoint: str = "/health"
    enabled: bool = True

@dataclass
class AgentTask:
    task_id: str
    agent_name: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 1
    timeout: int = 30
    retry_count: int = 0
    max_retries: int = 2
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class AgentResponse:
    agent_name: str
    task_id: str
    success: bool
    result: Any = None
    error: str = None
    response_time: float = 0.0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class AgentOrchestrator:
    """High-performance AI agent orchestration system"""
    
    def __init__(self):
        self.agents = self._initialize_agents()
        self.task_queue = asyncio.Queue()
        self.result_store = {}
        self.active_tasks = {}
        self.agent_sessions = {}
        self.performance_metrics = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        info("AI Agent Orchestrator initialized", category="system")
    
    def _initialize_agents(self) -> Dict[str, AgentConfig]:
        """Initialize AI agent configurations"""
        
        agents = {
            "browser_use": AgentConfig(
                name="browser_use",
                url="http://localhost",
                port=8088,
                capabilities=["web_browsing", "automation", "data_extraction"],
                priority=2,
                max_concurrent=2,
                timeout=45
            ),
            "langchain_agents": AgentConfig(
                name="langchain_agents", 
                url="http://localhost",
                port=8084,
                capabilities=["language_processing", "chain_reasoning", "tool_usage"],
                priority=1,
                max_concurrent=4,
                timeout=30
            ),
            "autogen": AgentConfig(
                name="autogen",
                url="http://localhost", 
                port=8085,
                capabilities=["multi_agent_chat", "code_generation", "collaboration"],
                priority=1,
                max_concurrent=3,
                timeout=60
            ),
            "ollama": AgentConfig(
                name="ollama",
                url="http://localhost",
                port=11434,
                capabilities=["language_models", "inference", "text_generation"],
                priority=3,
                max_concurrent=5,
                timeout=90,
                health_endpoint="/api/version"
            ),
            "open_webui": AgentConfig(
                name="open_webui",
                url="http://localhost",
                port=8089,
                capabilities=["ui_interface", "model_management", "chat_interface"],
                priority=2,
                max_concurrent=2,
                timeout=20
            )
        }
        
        debug(f"Initialized {len(agents)} AI agents", category="system")
        return agents
    
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        info("Starting AI Agent Orchestrator", category="system")
        
        # Create persistent sessions for each agent
        await self._create_agent_sessions()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._task_processor()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._performance_optimizer())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            log_exception(e, context="Orchestrator startup", category="error")
            await self.stop()
    
    async def stop(self):
        """Stop the orchestrator gracefully"""
        info("Stopping AI Agent Orchestrator", category="system")
        self.running = False
        
        # Close all agent sessions
        for session in self.agent_sessions.values():
            if session and not session.closed:
                await session.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        info("AI Agent Orchestrator stopped", category="system")
    
    async def _create_agent_sessions(self):
        """Create persistent HTTP sessions for agents"""
        
        for agent_name, config in self.agents.items():
            if not config.enabled:
                continue
                
            connector = aiohttp.TCPConnector(
                limit=config.max_concurrent,
                limit_per_host=config.max_concurrent,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=60
            )
            
            timeout = aiohttp.ClientTimeout(
                total=config.timeout,
                connect=10,
                sock_read=config.timeout - 10
            )
            
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "SutazAI-Orchestrator/1.0"}
            )
            
            self.agent_sessions[agent_name] = session
            debug(f"Created session for {agent_name}", category="system")
    
    async def _task_processor(self):
        """Process tasks from the queue"""
        
        while self.running:
            try:
                # Get task with timeout
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Process task asynchronously
                asyncio.create_task(self._execute_task(task))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log_exception(e, context="Task processor", category="error")
    
    async def _execute_task(self, task: AgentTask):
        """Execute a single task"""
        
        with log_context(f"Executing task {task.task_id} on {task.agent_name}", category="system"):
            
            start_time = time.time()
            self.active_tasks[task.task_id] = task
            
            try:
                # Get agent config
                agent_config = self.agents.get(task.agent_name)
                if not agent_config or not agent_config.enabled:
                    raise ValueError(f"Agent {task.agent_name} not available")
                
                # Get session
                session = self.agent_sessions.get(task.agent_name)
                if not session:
                    raise ValueError(f"No session for {task.agent_name}")
                
                # Execute task
                result = await self._call_agent(session, agent_config, task)
                response_time = time.time() - start_time
                
                # Create response
                response = AgentResponse(
                    agent_name=task.agent_name,
                    task_id=task.task_id,
                    success=True,
                    result=result,
                    response_time=response_time
                )
                
                info(
                    f"Task {task.task_id} completed successfully in {response_time:.3f}s",
                    category="system",
                    task_id=task.task_id,
                    agent=task.agent_name,
                    response_time=response_time
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                
                # Handle retries
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    warning(
                        f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})",
                        category="system"
                    )
                    await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                    await self.task_queue.put(task)
                    return
                
                response = AgentResponse(
                    agent_name=task.agent_name,
                    task_id=task.task_id,
                    success=False,
                    error=str(e),
                    response_time=response_time
                )
                
                error(
                    f"Task {task.task_id} failed permanently: {str(e)}",
                    category="error",
                    task_id=task.task_id,
                    agent=task.agent_name
                )
            
            finally:
                # Store result and cleanup
                self.result_store[task.task_id] = response
                self.active_tasks.pop(task.task_id, None)
                self._update_metrics(task.agent_name, response)
    
    async def _call_agent(self, session: aiohttp.ClientSession, config: AgentConfig, task: AgentTask):
        """Make API call to agent"""
        
        url = f"{config.url}:{config.port}/execute"
        
        payload = {
            "task_type": task.task_type,
            "task_id": task.task_id,
            "data": task.payload,
            "timeout": task.timeout
        }
        
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Agent {config.name} returned {response.status}: {error_text}")
    
    async def _health_monitor(self):
        """Monitor agent health"""
        
        while self.running:
            try:
                for agent_name, config in self.agents.items():
                    if not config.enabled:
                        continue
                    
                    await self._check_agent_health(agent_name, config)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                log_exception(e, context="Health monitor", category="error")
                await asyncio.sleep(5)
    
    async def _check_agent_health(self, agent_name: str, config: AgentConfig):
        """Check individual agent health"""
        
        try:
            session = self.agent_sessions.get(agent_name)
            if not session:
                return
            
            url = f"{config.url}:{config.port}{config.health_endpoint}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    debug(f"Agent {agent_name} healthy", category="system")
                    config.enabled = True
                else:
                    warning(f"Agent {agent_name} unhealthy: {response.status}", category="system")
                    config.enabled = False
                    
        except Exception as e:
            warning(f"Agent {agent_name} unreachable: {str(e)}", category="system")
            config.enabled = False
    
    async def _metrics_collector(self):
        """Collect performance metrics"""
        
        while self.running:
            try:
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "active_tasks": len(self.active_tasks),
                    "queue_size": self.task_queue.qsize(),
                    "agent_status": {
                        name: config.enabled 
                        for name, config in self.agents.items()
                    },
                    "performance": self.performance_metrics.copy()
                }
                
                debug(f"Metrics: {json.dumps(metrics, indent=2)}", category="system")
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                log_exception(e, context="Metrics collector", category="error")
                await asyncio.sleep(5)
    
    async def _performance_optimizer(self):
        """Optimize performance based on metrics"""
        
        while self.running:
            try:
                # Analyze metrics and adjust agent priorities
                for agent_name, metrics in self.performance_metrics.items():
                    if metrics.get("avg_response_time", 0) > 30:
                        warning(f"Agent {agent_name} slow, reducing priority", category="system")
                        if agent_name in self.agents:
                            self.agents[agent_name].priority = max(1, self.agents[agent_name].priority - 1)
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                log_exception(e, context="Performance optimizer", category="error")
                await asyncio.sleep(60)
    
    def _update_metrics(self, agent_name: str, response: AgentResponse):
        """Update performance metrics"""
        
        if agent_name not in self.performance_metrics:
            self.performance_metrics[agent_name] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "total_response_time": 0.0,
                "avg_response_time": 0.0
            }
        
        metrics = self.performance_metrics[agent_name]
        metrics["total_tasks"] += 1
        metrics["total_response_time"] += response.response_time
        metrics["avg_response_time"] = metrics["total_response_time"] / metrics["total_tasks"]
        
        if response.success:
            metrics["successful_tasks"] += 1
        else:
            metrics["failed_tasks"] += 1
    
    async def submit_task(self, agent_name: str, task_type: str, payload: Dict[str, Any], 
                         priority: int = 1, timeout: int = 30) -> str:
        """Submit a task to an agent"""
        
        task_id = f"{agent_name}_{int(time.time() * 1000)}"
        
        task = AgentTask(
            task_id=task_id,
            agent_name=agent_name,
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout=timeout
        )
        
        await self.task_queue.put(task)
        
        info(f"Task {task_id} submitted to {agent_name}", category="system")
        return task_id
    
    async def get_result(self, task_id: str, timeout: int = 60) -> Optional[AgentResponse]:
        """Get task result"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.result_store:
                return self.result_store[task_id]
            
            await asyncio.sleep(0.1)
        
        return None
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        
        return {
            "agents": {
                name: {
                    "enabled": config.enabled,
                    "capabilities": config.capabilities,
                    "priority": config.priority,
                    "performance": self.performance_metrics.get(name, {})
                }
                for name, config in self.agents.items()
            },
            "system": {
                "active_tasks": len(self.active_tasks),
                "queue_size": self.task_queue.qsize(),
                "running": self.running
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        info(f"Received signal {signum}, shutting down", category="system")
        self.running = False

# Global orchestrator instance
orchestrator = None

async def initialize_orchestrator():
    """Initialize the global orchestrator"""
    global orchestrator
    if orchestrator is None:
        orchestrator = AgentOrchestrator()
        # Start in background
        asyncio.create_task(orchestrator.start())
    return orchestrator

async def submit_agent_task(agent_name: str, task_type: str, payload: Dict[str, Any], 
                           priority: int = 1, timeout: int = 30) -> str:
    """Submit task to agent (convenience function)"""
    orch = await initialize_orchestrator()
    return await orch.submit_task(agent_name, task_type, payload, priority, timeout)

async def get_agent_result(task_id: str, timeout: int = 60) -> Optional[AgentResponse]:
    """Get task result (convenience function)"""
    orch = await initialize_orchestrator()
    return await orch.get_result(task_id, timeout)

def get_system_status() -> Dict[str, Any]:
    """Get system status (convenience function)"""
    global orchestrator
    if orchestrator:
        return orchestrator.get_agent_status()
    return {"error": "Orchestrator not initialized"}

if __name__ == "__main__":
    async def main():
        orch = AgentOrchestrator()
        try:
            await orch.start()
        except KeyboardInterrupt:
            await orch.stop()
    
    asyncio.run(main())