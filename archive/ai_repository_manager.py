#!/usr/bin/env python3
"""
SutazAI v9 AI Repository Integration Manager
Dynamically manages and integrates all 48+ AI repositories
"""

import asyncio
import json
import logging
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    UNKNOWN = "unknown"

class ServiceType(Enum):
    AI_MODEL = "ai_model"
    CODE_GENERATION = "code_generation"
    DOCUMENT_PROCESSING = "document_processing"
    WEB_AUTOMATION = "web_automation"
    ML_FRAMEWORK = "ml_framework"
    KNOWLEDGE_MANAGEMENT = "knowledge_management"
    AGENT_FRAMEWORK = "agent_framework"
    SPECIALIZED_AI = "specialized_ai"

@dataclass
class AIRepository:
    """Represents an AI repository/service"""
    name: str
    path: str
    service_type: ServiceType
    status: ServiceStatus
    port: int
    dockerfile_path: str
    config: Dict[str, Any]
    dependencies: List[str]
    capabilities: List[str]
    health_endpoint: str = "/health"
    api_endpoint: str = "/api"
    container_id: Optional[str] = None
    last_health_check: float = 0
    startup_time: float = 0
    error_message: str = ""

class AIRepositoryManager:
    """Manages all AI repositories and services"""
    
    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        self.base_path = Path(base_path)
        self.docker_path = self.base_path / "docker"
        self.repositories: Dict[str, AIRepository] = {}
        self.service_registry = {}
        self.port_manager = PortManager(start_port=8100)
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Load service configurations
        self._discover_repositories()
        self._load_service_configs()
    
    def _discover_repositories(self):
        """Discover all AI repositories in the docker directory"""
        logger.info("Discovering AI repositories...")
        
        if not self.docker_path.exists():
            logger.error(f"Docker path {self.docker_path} does not exist")
            return
        
        # Repository configurations with types and capabilities
        repo_configs = {
            "agentgpt": {
                "type": ServiceType.AGENT_FRAMEWORK,
                "capabilities": ["autonomous_tasks", "goal_planning", "web_search"],
                "dependencies": ["redis", "postgresql"]
            },
            "agentzero": {
                "type": ServiceType.AGENT_FRAMEWORK,
                "capabilities": ["zero_shot_learning", "task_automation"],
                "dependencies": []
            },
            "aider": {
                "type": ServiceType.CODE_GENERATION,
                "capabilities": ["code_editing", "git_integration", "refactoring"],
                "dependencies": ["git"]
            },
            "autogen": {
                "type": ServiceType.AGENT_FRAMEWORK,
                "capabilities": ["multi_agent_chat", "code_execution", "planning"],
                "dependencies": []
            },
            "autogpt": {
                "type": ServiceType.AGENT_FRAMEWORK,
                "capabilities": ["autonomous_tasks", "file_operations", "web_browsing"],
                "dependencies": ["redis"]
            },
            "awesome-code-ai": {
                "type": ServiceType.CODE_GENERATION,
                "capabilities": ["code_generation", "code_analysis", "optimization"],
                "dependencies": []
            },
            "bigagi": {
                "type": ServiceType.AI_MODEL,
                "capabilities": ["large_language_model", "conversation", "reasoning"],
                "dependencies": []
            },
            "browser-use": {
                "type": ServiceType.WEB_AUTOMATION,
                "capabilities": ["web_scraping", "browser_automation", "form_filling"],
                "dependencies": ["selenium"]
            },
            "context-engineering": {
                "type": ServiceType.SPECIALIZED_AI,
                "capabilities": ["context_optimization", "prompt_engineering"],
                "dependencies": []
            },
            "crewai": {
                "type": ServiceType.AGENT_FRAMEWORK,
                "capabilities": ["multi_agent_collaboration", "role_based_agents", "workflows"],
                "dependencies": []
            },
            "dify": {
                "type": ServiceType.AGENT_FRAMEWORK,
                "capabilities": ["llm_ops", "workflow_orchestration", "api_management"],
                "dependencies": ["postgresql", "redis"]
            },
            "documind": {
                "type": ServiceType.DOCUMENT_PROCESSING,
                "capabilities": ["document_parsing", "text_extraction", "pdf_processing"],
                "dependencies": []
            },
            "enhanced-model-manager": {
                "type": ServiceType.AI_MODEL,
                "capabilities": ["model_management", "deepseek_integration", "model_switching"],
                "dependencies": ["deepseek", "qwen"]
            },
            "faiss": {
                "type": ServiceType.KNOWLEDGE_MANAGEMENT,
                "capabilities": ["vector_search", "similarity_search", "indexing"],
                "dependencies": []
            },
            "finrobot": {
                "type": ServiceType.SPECIALIZED_AI,
                "capabilities": ["financial_analysis", "market_data", "trading_insights"],
                "dependencies": []
            },
            "flowise": {
                "type": ServiceType.AGENT_FRAMEWORK,
                "capabilities": ["visual_workflow", "llm_chains", "drag_drop_ai"],
                "dependencies": []
            },
            "fms-fsdp": {
                "type": ServiceType.ML_FRAMEWORK,
                "capabilities": ["distributed_training", "model_parallelism", "fsdp"],
                "dependencies": ["pytorch"]
            },
            "gpt-engineer": {
                "type": ServiceType.CODE_GENERATION,
                "capabilities": ["project_generation", "architecture_design", "full_stack"],
                "dependencies": []
            },
            "jax": {
                "type": ServiceType.ML_FRAMEWORK,
                "capabilities": ["jax_framework", "neural_networks", "scientific_computing"],
                "dependencies": []
            },
            "knowledge-manager": {
                "type": ServiceType.KNOWLEDGE_MANAGEMENT,
                "capabilities": ["knowledge_graphs", "information_retrieval", "semantic_search"],
                "dependencies": []
            },
            "langchain-agents": {
                "type": ServiceType.AGENT_FRAMEWORK,
                "capabilities": ["langchain_integration", "tool_usage", "memory_management"],
                "dependencies": []
            },
            "langflow": {
                "type": ServiceType.AGENT_FRAMEWORK,
                "capabilities": ["visual_programming", "langchain_ui", "flow_building"],
                "dependencies": []
            },
            "llamaindex": {
                "type": ServiceType.KNOWLEDGE_MANAGEMENT,
                "capabilities": ["data_indexing", "query_engine", "rag_pipeline"],
                "dependencies": []
            },
            "localagi": {
                "type": ServiceType.AI_MODEL,
                "capabilities": ["local_llm", "offline_inference", "privacy_focused"],
                "dependencies": []
            },
            "privategpt": {
                "type": ServiceType.AI_MODEL,
                "capabilities": ["private_documents", "local_rag", "privacy_preserving"],
                "dependencies": []
            },
            "pytorch": {
                "type": ServiceType.ML_FRAMEWORK,
                "capabilities": ["deep_learning", "neural_networks", "gpu_acceleration"],
                "dependencies": []
            },
            "realtimestt": {
                "type": ServiceType.SPECIALIZED_AI,
                "capabilities": ["speech_to_text", "real_time_transcription", "voice_processing"],
                "dependencies": []
            },
            "reasoning-engine": {
                "type": ServiceType.SPECIALIZED_AI,
                "capabilities": ["logical_reasoning", "inference_engine", "decision_making"],
                "dependencies": []
            },
            "skyvern": {
                "type": ServiceType.WEB_AUTOMATION,
                "capabilities": ["advanced_web_automation", "ai_powered_browsing"],
                "dependencies": []
            },
            "tabbyml": {
                "type": ServiceType.CODE_GENERATION,
                "capabilities": ["code_completion", "code_suggestions", "ide_integration"],
                "dependencies": []
            },
            "tensorflow": {
                "type": ServiceType.ML_FRAMEWORK,
                "capabilities": ["machine_learning", "tensorflow_framework", "model_serving"],
                "dependencies": []
            }
        }
        
        for repo_dir in self.docker_path.iterdir():
            if repo_dir.is_dir() and repo_dir.name in repo_configs:
                config = repo_configs[repo_dir.name]
                dockerfile_path = repo_dir / "Dockerfile"
                
                if dockerfile_path.exists():
                    port = self.port_manager.allocate_port(repo_dir.name)
                    
                    repository = AIRepository(
                        name=repo_dir.name,
                        path=str(repo_dir),
                        service_type=config["type"],
                        status=ServiceStatus.STOPPED,
                        port=port,
                        dockerfile_path=str(dockerfile_path),
                        config={},
                        dependencies=config["dependencies"],
                        capabilities=config["capabilities"]
                    )
                    
                    self.repositories[repo_dir.name] = repository
                    logger.info(f"Discovered repository: {repo_dir.name} on port {port}")
        
        logger.info(f"Discovered {len(self.repositories)} AI repositories")
    
    def _load_service_configs(self):
        """Load individual service configurations"""
        for repo_name, repo in self.repositories.items():
            config_file = Path(repo.path) / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        repo.config = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load config for {repo_name}: {e}")
    
    async def start_service(self, repo_name: str) -> bool:
        """Start a specific AI service"""
        if repo_name not in self.repositories:
            logger.error(f"Repository {repo_name} not found")
            return False
        
        repo = self.repositories[repo_name]
        
        if repo.status == ServiceStatus.RUNNING:
            logger.info(f"Service {repo_name} is already running")
            return True
        
        logger.info(f"Starting service: {repo_name}")
        repo.status = ServiceStatus.STARTING
        
        try:
            # Check dependencies first
            if not await self._check_dependencies(repo):
                repo.status = ServiceStatus.ERROR
                repo.error_message = "Dependencies not met"
                return False
            
            # Build and start container
            success = await self._build_and_run_container(repo)
            
            if success:
                repo.status = ServiceStatus.RUNNING
                repo.startup_time = time.time()
                logger.info(f"Service {repo_name} started successfully on port {repo.port}")
                return True
            else:
                repo.status = ServiceStatus.ERROR
                repo.error_message = "Failed to start container"
                return False
        
        except Exception as e:
            logger.error(f"Error starting service {repo_name}: {e}")
            repo.status = ServiceStatus.ERROR
            repo.error_message = str(e)
            return False
    
    async def stop_service(self, repo_name: str) -> bool:
        """Stop a specific AI service"""
        if repo_name not in self.repositories:
            logger.error(f"Repository {repo_name} not found")
            return False
        
        repo = self.repositories[repo_name]
        logger.info(f"Stopping service: {repo_name}")
        
        try:
            if repo.container_id:
                # Stop the container
                result = subprocess.run([
                    "docker", "stop", repo.container_id
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    repo.status = ServiceStatus.STOPPED
                    repo.container_id = None
                    logger.info(f"Service {repo_name} stopped successfully")
                    return True
                else:
                    logger.error(f"Failed to stop container: {result.stderr}")
                    return False
            else:
                repo.status = ServiceStatus.STOPPED
                return True
        
        except Exception as e:
            logger.error(f"Error stopping service {repo_name}: {e}")
            return False
    
    async def restart_service(self, repo_name: str) -> bool:
        """Restart a specific AI service"""
        await self.stop_service(repo_name)
        await asyncio.sleep(2)
        return await self.start_service(repo_name)
    
    async def start_all_services(self) -> Dict[str, bool]:
        """Start all discovered AI services"""
        logger.info("Starting all AI services...")
        results = {}
        
        # Start services in batches to avoid overwhelming the system
        batch_size = 5
        repos = list(self.repositories.keys())
        
        for i in range(0, len(repos), batch_size):
            batch = repos[i:i + batch_size]
            tasks = [self.start_service(repo_name) for repo_name in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for repo_name, result in zip(batch, batch_results):
                results[repo_name] = result if not isinstance(result, Exception) else False
            
            # Wait between batches
            if i + batch_size < len(repos):
                await asyncio.sleep(5)
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Started {successful}/{len(results)} services successfully")
        return results
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health checks on all running services"""
        health_results = {}
        
        for repo_name, repo in self.repositories.items():
            if repo.status == ServiceStatus.RUNNING:
                health = await self._perform_health_check(repo)
                health_results[repo_name] = health
        
        return health_results
    
    async def _perform_health_check(self, repo: AIRepository) -> Dict[str, Any]:
        """Perform health check on a specific service"""
        try:
            url = f"http://localhost:{repo.port}{repo.health_endpoint}"
            
            async with asyncio.timeout(10):
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    repo.last_health_check = time.time()
                    return {
                        "status": "healthy",
                        "response_time": response.elapsed.total_seconds(),
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                        "timestamp": time.time()
                    }
        
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _check_dependencies(self, repo: AIRepository) -> bool:
        """Check if service dependencies are available"""
        for dep in repo.dependencies:
            if dep in self.repositories:
                dep_repo = self.repositories[dep]
                if dep_repo.status != ServiceStatus.RUNNING:
                    logger.warning(f"Dependency {dep} for {repo.name} is not running")
                    return False
        return True
    
    async def _build_and_run_container(self, repo: AIRepository) -> bool:
        """Build and run Docker container for a service"""
        try:
            # Build the image
            build_cmd = [
                "docker", "build",
                "-t", f"sutazai-{repo.name}:latest",
                repo.path
            ]
            
            logger.info(f"Building image for {repo.name}...")
            build_result = subprocess.run(build_cmd, capture_output=True, text=True)
            
            if build_result.returncode != 0:
                logger.error(f"Build failed for {repo.name}: {build_result.stderr}")
                return False
            
            # Run the container
            run_cmd = [
                "docker", "run", "-d",
                "--name", f"sutazai-{repo.name}",
                "-p", f"{repo.port}:8000",
                "--network", "sutazai-network",
                f"sutazai-{repo.name}:latest"
            ]
            
            logger.info(f"Running container for {repo.name}...")
            run_result = subprocess.run(run_cmd, capture_output=True, text=True)
            
            if run_result.returncode == 0:
                repo.container_id = run_result.stdout.strip()
                logger.info(f"Container started for {repo.name}: {repo.container_id}")
                return True
            else:
                logger.error(f"Run failed for {repo.name}: {run_result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Error building/running container for {repo.name}: {e}")
            return False
    
    def get_service_status(self, repo_name: str = None) -> Dict[str, Any]:
        """Get status of services"""
        if repo_name:
            if repo_name in self.repositories:
                repo = self.repositories[repo_name]
                return asdict(repo)
            else:
                return {"error": f"Service {repo_name} not found"}
        else:
            return {
                "total_services": len(self.repositories),
                "running": len([r for r in self.repositories.values() if r.status == ServiceStatus.RUNNING]),
                "stopped": len([r for r in self.repositories.values() if r.status == ServiceStatus.STOPPED]),
                "error": len([r for r in self.repositories.values() if r.status == ServiceStatus.ERROR]),
                "services": {name: repo.status.value for name, repo in self.repositories.items()}
            }
    
    def get_services_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get services that have a specific capability"""
        matching_services = []
        
        for repo_name, repo in self.repositories.items():
            if capability in repo.capabilities:
                matching_services.append({
                    "name": repo_name,
                    "status": repo.status.value,
                    "port": repo.port,
                    "capabilities": repo.capabilities
                })
        
        return matching_services
    
    def get_services_by_type(self, service_type: ServiceType) -> List[Dict[str, Any]]:
        """Get services of a specific type"""
        matching_services = []
        
        for repo_name, repo in self.repositories.items():
            if repo.service_type == service_type:
                matching_services.append({
                    "name": repo_name,
                    "status": repo.status.value,
                    "port": repo.port,
                    "type": repo.service_type.value
                })
        
        return matching_services
    
    async def create_service_network(self):
        """Create Docker network for services"""
        try:
            # Check if network exists
            check_cmd = ["docker", "network", "ls", "--filter", "name=sutazai-network"]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if "sutazai-network" not in result.stdout:
                # Create network
                create_cmd = ["docker", "network", "create", "sutazai-network"]
                create_result = subprocess.run(create_cmd, capture_output=True, text=True)
                
                if create_result.returncode == 0:
                    logger.info("Created Docker network: sutazai-network")
                    return True
                else:
                    logger.error(f"Failed to create network: {create_result.stderr}")
                    return False
            else:
                logger.info("Docker network sutazai-network already exists")
                return True
        
        except Exception as e:
            logger.error(f"Error creating network: {e}")
            return False

class PortManager:
    """Manages port allocation for services"""
    
    def __init__(self, start_port: int = 8100, end_port: int = 9000):
        self.start_port = start_port
        self.end_port = end_port
        self.allocated_ports = {}
        self.current_port = start_port
    
    def allocate_port(self, service_name: str) -> int:
        """Allocate a port for a service"""
        if service_name in self.allocated_ports:
            return self.allocated_ports[service_name]
        
        while self.current_port <= self.end_port:
            if self.current_port not in self.allocated_ports.values():
                self.allocated_ports[service_name] = self.current_port
                port = self.current_port
                self.current_port += 1
                return port
            self.current_port += 1
        
        raise Exception("No available ports")
    
    def deallocate_port(self, service_name: str):
        """Deallocate a port for a service"""
        if service_name in self.allocated_ports:
            del self.allocated_ports[service_name]

# Example usage and testing
async def main():
    """Example usage of AI Repository Manager"""
    manager = AIRepositoryManager()
    
    # Create network
    await manager.create_service_network()
    
    # Show discovered services
    status = manager.get_service_status()
    print(f"Discovered {status['total_services']} AI services:")
    
    for service_name in status['services']:
        repo = manager.repositories[service_name]
        print(f"  - {service_name}: {repo.service_type.value} ({', '.join(repo.capabilities)})")
    
    # Start a few key services
    key_services = ["enhanced-model-manager", "crewai", "langchain-agents", "documind"]
    
    for service in key_services:
        if service in manager.repositories:
            success = await manager.start_service(service)
            print(f"Starting {service}: {'✅' if success else '❌'}")
    
    # Perform health checks
    health_results = await manager.health_check_all()
    print(f"\nHealth check results:")
    for service, health in health_results.items():
        print(f"  - {service}: {health['status']}")

if __name__ == "__main__":
    asyncio.run(main())