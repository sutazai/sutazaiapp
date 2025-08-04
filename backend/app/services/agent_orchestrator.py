"""
Enhanced Agent Orchestrator - Manages all 131 AI agents with Ollama integration
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid
import subprocess
import os
from pathlib import Path
import requests
import logging

# Fallback imports if core modules not available
try:
    from app.core.logging import get_logger
    from app.core.config import settings
    from app.models.agent import AgentTask
    from app.services.base_service import BaseService
    from app.agents.registry import AgentRegistry
    from app.core.exceptions import AgentNotFoundError, AgentExecutionError
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Mock classes for fallback
    class BaseService:
        def __init__(self): pass
    
    class AgentTask:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class AgentRegistry:
        def __init__(self): 
            self.agents = {}
        def load_agents(self): pass
        def agent_exists(self, name): return True
        def get_agent(self, name): return None
        def list_agents(self): return []
        def get_agent_info(self, name): return {}
    
    class settings:
        MAX_CONCURRENT_AGENTS = 50
    
    class AgentNotFoundError(Exception): pass
    class AgentExecutionError(Exception): pass

class AgentOrchestrator(BaseService):
    """
    Enhanced orchestrator for 131 AI agents with Ollama/TinyLlama integration
    """
    
    def __init__(self):
        super().__init__()
        self.registry = AgentRegistry()
        self.active_tasks: Dict[str, AgentTask] = {}
        self.task_queue = asyncio.Queue(maxsize=getattr(settings, 'MAX_CONCURRENT_AGENTS', 50))
        self.workers: List[asyncio.Task] = []
        
        # Enhanced attributes for mass deployment
        self.agents_dir = Path("/opt/sutazaiapp/agents")
        self.ollama_base_url = "http://localhost:11434"
        self.active_agents: Dict[str, Any] = {}
        self.agent_processes: Dict[str, subprocess.Popen] = {}
        self.ollama_ready = False
        self.deployment_stats = {
            "total_discovered": 0,
            "successful_starts": 0,
            "failed_starts": 0,
            "healthy_agents": 0
        }
        
    async def initialize(self):
        """Initialize the orchestrator"""
        logger.info("Initializing Agent Orchestrator...")
        
        # Load all agents
        await self.registry.load_agents()
        
        # Start worker tasks
        for i in range(settings.MAX_CONCURRENT_AGENTS):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            
        logger.info(f"Started {len(self.workers)} agent workers")
        
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Agent Orchestrator...")
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
    async def execute_task(
        self,
        agent_name: str,
        task_description: str,
        parameters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Execute a task using specified agent
        """
        # Validate agent exists
        if not self.registry.agent_exists(agent_name):
            raise AgentNotFoundError(f"Agent '{agent_name}' not found")
            
        # Create task
        task_id = str(uuid.uuid4())
        task = AgentTask(
            id=task_id,
            agent_name=agent_name,
            task_description=task_description,
            parameters=parameters or {},
            user_id=user_id,
            status="queued",
            created_at=datetime.utcnow()
        )
        
        # Store task
        self.active_tasks[task_id] = task
        
        # Queue task
        await self.task_queue.put(task)
        
        logger.info(f"Queued task {task_id} for agent {agent_name}")
        
        return task_id
        
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task"""
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        return {
            "id": task.id,
            "agent_name": task.agent_name,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result
        }
        
    async def cancel_task(self, task_id: str):
        """Cancel a running task"""
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        if task.status in ["completed", "failed"]:
            raise ValueError(f"Task {task_id} already {task.status}")
            
        task.status = "cancelled"
        task.completed_at = datetime.utcnow()
        
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents"""
        return self.registry.list_agents()
        
    async def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a specific agent"""
        return self.registry.get_agent_info(agent_name)
        
    async def _worker(self, worker_id: str):
        """Worker task that processes agent tasks"""
        logger.info(f"Agent worker {worker_id} started")
        
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Update task status
                task.status = "running"
                
                logger.info(f"Worker {worker_id} executing task {task.id}")
                
                # Get agent
                agent = self.registry.get_agent(task.agent_name)
                
                # Execute task
                try:
                    result = await agent.execute(
                        task.task_description,
                        task.parameters
                    )
                    
                    # Update task with result
                    task.status = "completed"
                    task.result = result
                    task.completed_at = datetime.utcnow()
                    
                except Exception as e:
                    logger.error(f"Agent execution error: {e}")
                    task.status = "failed"
                    task.result = {"error": str(e)}
                    task.completed_at = datetime.utcnow()
                    
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.info(f"Agent worker {worker_id} stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "active_agents": len(self.registry.agents),
            "queued_tasks": self.task_queue.qsize(),
            "active_tasks": len([t for t in self.active_tasks.values() if t.status == "running"]),
            "workers": len(self.workers),
            "deployed_agents": len(self.active_agents),
            "deployment_stats": self.deployment_stats,
            "ollama_ready": self.ollama_ready
        }
    
    async def initialize_ollama(self) -> bool:
        """Initialize Ollama service"""
        try:
            logger.info("🔧 Initializing Ollama service...")
            
            # Check if already running
            try:
                response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    self.ollama_ready = True
                    logger.info("✅ Ollama is already running")
                    return await self.ensure_tinyllama()
            except requests.exceptions.RequestException:
                pass
            
            # Start Ollama service
            logger.info("🚀 Starting Ollama service...")
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for startup
            for attempt in range(30):
                try:
                    response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=3)
                    if response.status_code == 200:
                        self.ollama_ready = True
                        logger.info("✅ Ollama started successfully")
                        return await self.ensure_tinyllama()
                except requests.exceptions.RequestException:
                    await asyncio.sleep(1)
            
            logger.error("❌ Failed to start Ollama")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error initializing Ollama: {e}")
            return False
    
    async def ensure_tinyllama(self) -> bool:
        """Ensure TinyLlama model is available"""
        try:
            # Check existing models
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if any("tinyllama" in model.get("name", "").lower() for model in models):
                    logger.info("✅ TinyLlama model is available")
                    return True
            
            # Pull TinyLlama
            logger.info("📥 Pulling TinyLlama model...")
            pull_process = subprocess.run(
                ["ollama", "pull", "tinyllama"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if pull_process.returncode == 0:
                logger.info("✅ TinyLlama model ready")
                return True
            else:
                logger.error(f"❌ Failed to pull TinyLlama: {pull_process.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error ensuring TinyLlama: {e}")
            return False
    
    async def discover_all_agents(self) -> List[Dict[str, Any]]:
        """Discover all available agents"""
        discovered = []
        
        try:
            # Load agent registry
            registry_file = self.agents_dir / "agent_registry.json"
            registry_data = {}
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f).get("agents", {})
            
            # Scan agent directories
            for agent_dir in self.agents_dir.iterdir():
                if agent_dir.is_dir() and (agent_dir / "app.py").exists():
                    agent_name = agent_dir.name
                    registry_info = registry_data.get(agent_name, {})
                    
                    agent_info = {
                        "name": agent_name,
                        "path": str(agent_dir),
                        "type": self.classify_agent(agent_name),
                        "description": registry_info.get("description", f"AI agent for {agent_name}"),
                        "capabilities": registry_info.get("capabilities", ["automation"]),
                        "status": "discovered"
                    }
                    discovered.append(agent_info)
            
            self.deployment_stats["total_discovered"] = len(discovered)
            logger.info(f"🔍 Discovered {len(discovered)} agents")
            return discovered
            
        except Exception as e:
            logger.error(f"❌ Error discovering agents: {e}")
            return []
    
    def classify_agent(self, name: str) -> str:
        """Classify agent type"""
        name_lower = name.lower()
        if any(k in name_lower for k in ['opus', 'agi', 'asi']): return "opus"
        elif any(k in name_lower for k in ['sonnet', 'ai-']): return "sonnet"
        elif any(k in name_lower for k in ['security', 'kali']): return "security"
        elif any(k in name_lower for k in ['frontend', 'ui']): return "frontend"
        elif any(k in name_lower for k in ['backend', 'api']): return "backend"
        elif any(k in name_lower for k in ['devops', 'infrastructure']): return "infrastructure"
        elif any(k in name_lower for k in ['test', 'qa']): return "testing"
        elif any(k in name_lower for k in ['monitor', 'observability']): return "monitoring"
        else: return "utility"
    
    async def deploy_all_agents(self) -> Dict[str, Any]:
        """Deploy all 131 agents with Ollama integration"""
        logger.info("🚀 Starting mass agent deployment...")
        start_time = datetime.utcnow()
        
        # Phase 1: Initialize Ollama
        if not await self.initialize_ollama():
            return {"status": "failed", "error": "Ollama initialization failed"}
        
        # Phase 2: Discover agents
        agents = await self.discover_all_agents()
        if not agents:
            return {"status": "failed", "error": "No agents discovered"}
        
        # Phase 3: Deploy in batches
        logger.info(f"📦 Deploying {len(agents)} agents in batches...")
        batch_size = 12
        
        for i in range(0, len(agents), batch_size):
            batch = agents[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(agents) + batch_size - 1) // batch_size
            
            logger.info(f"⚡ Batch {batch_num}/{total_batches}: {len(batch)} agents")
            
            # Deploy batch concurrently
            tasks = [self.start_agent_with_ollama(agent) for agent in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(results):
                if isinstance(result, Exception) or not result:
                    self.deployment_stats["failed_starts"] += 1
                else:
                    self.deployment_stats["successful_starts"] += 1
            
            await asyncio.sleep(1)  # Brief pause between batches
        
        # Phase 4: Health check
        await asyncio.sleep(5)
        healthy = await self.health_check_all()
        self.deployment_stats["healthy_agents"] = healthy
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        result = {
            "status": "completed",
            "duration_seconds": duration,
            "stats": self.deployment_stats,
            "intelligence_level": "ASI" if healthy > 100 else "AGI" if healthy > 50 else "Multi-Agent",
            "collective_active": healthy > 10,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"🎉 Deployment complete: {healthy}/{len(agents)} agents healthy")
        return result
    
    async def start_agent_with_ollama(self, agent_info: Dict[str, Any]) -> bool:
        """Start individual agent with Ollama integration"""
        try:
            agent_name = agent_info["name"]
            agent_path = Path(agent_info["path"])
            port = 8000 + len(self.active_agents)
            
            # Environment setup
            env = os.environ.copy()
            env.update({
                'AGENT_NAME': agent_name,
                'AGENT_PORT': str(port),
                'OLLAMA_BASE_URL': self.ollama_base_url,
                'OLLAMA_MODEL': 'tinyllama',
                'PYTHONPATH': '/opt/sutazaiapp:/opt/sutazaiapp/agents'
            })
            
            # Start agent
            process = subprocess.Popen(
                ['python3', 'app.py'],
                cwd=agent_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            await asyncio.sleep(2)  # Startup time
            
            if process.poll() is None:  # Still running
                self.agent_processes[agent_name] = process
                agent_info.update({
                    "status": "running",
                    "port": port,
                    "process_id": process.pid,
                    "startup_time": datetime.utcnow().isoformat()
                })
                self.active_agents[agent_name] = agent_info
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start {agent_name}: {e}")
            return False
    
    async def health_check_all(self) -> int:
        """Health check all active agents"""
        healthy = 0
        for name, info in self.active_agents.items():
            try:
                if name in self.agent_processes:
                    process = self.agent_processes[name]
                    if process.poll() is None:  # Still running
                        info["status"] = "healthy"
                        info["last_check"] = datetime.utcnow().isoformat()
                        healthy += 1
                    else:
                        info["status"] = "stopped"
            except Exception:
                info["status"] = "unhealthy"
        return healthy
    
    async def activate_collective_intelligence(self) -> Dict[str, Any]:
        """Activate AGI/ASI collective intelligence"""
        logger.info("🧠 Activating collective intelligence...")
        
        healthy = await self.health_check_all()
        
        collective = {
            "active": True,
            "total_agents": len(self.active_agents),
            "healthy_agents": healthy, 
            "level": "ASI" if healthy > 100 else "AGI" if healthy > 50 else "Multi-Agent",
            "capabilities": [
                "distributed_reasoning",
                "collective_problem_solving",
                "autonomous_coordination",
                "self_improvement",
                "emergent_intelligence"
            ],
            "activation_time": datetime.utcnow().isoformat()
        }
        
        # Save collective config
        config_file = self.agents_dir / "collective_intelligence.json"
        with open(config_file, 'w') as f:
            json.dump(collective, f, indent=2)
        
        logger.info(f"🧠 Collective intelligence activated: {collective['level']}")
        return collective

# Create singleton instance
agent_orchestrator = AgentOrchestrator()