"""
Service Registry for automation/advanced automation System
Manages inter-service communication and discovery
"""
from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ServiceRegistry:
    """Central registry for all AI services"""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {
            # Core Infrastructure
            "ollama": {"url": "http://ollama:10104", "type": "llm", "priority": 1},
            "postgres": {"url": "postgresql://sutazai:sutazai_password@postgres:5432/sutazai", "type": "database"},
            "redis": {"url": "redis://:redis_password@redis:6379", "type": "cache"},
            "neo4j": {"url": "bolt://neo4j:7687", "type": "graph"},
            
            # Vector Databases
            "chromadb": {"url": "http://chromadb:8000", "type": "vector", "priority": 1},
            "qdrant": {"url": "http://qdrant:6333", "type": "vector", "priority": 2},
            "faiss": {"url": "http://faiss:8000", "type": "vector", "priority": 3},
            
            # AI Agents
            "autogpt": {"url": "http://autogpt:8080", "type": "agent", "capabilities": ["task_automation"]},
            "crewai": {"url": "http://crewai:8080", "type": "agent", "capabilities": ["multi_agent"]},
            "letta": {"url": "http://letta:8080", "type": "agent", "capabilities": ["memory"]},
            "aider": {"url": "http://aider:8080", "type": "agent", "capabilities": ["code_assistant"]},
            "gpt-engineer": {"url": "http://gpt-engineer:8080", "type": "agent", "capabilities": ["code_generation"]},
            "localagi": {"url": "http://localagi:8090", "type": "orchestrator", "capabilities": ["orchestration"]},
            "tabbyml": {"url": "http://tabbyml:8080", "type": "agent", "capabilities": ["code_completion"]},
            "semgrep": {"url": "http://semgrep:8080", "type": "agent", "capabilities": ["security_scan"]},
            "autogen": {"url": "http://autogen:8080", "type": "agent", "capabilities": ["agent_config"]},
            "agentzero": {"url": "http://agentzero:8080", "type": "agent", "capabilities": ["general"]},
            "bigagi": {"url": "http://bigagi:3000", "type": "agent", "capabilities": ["ui_agent"]},
            "browser-use": {"url": "http://browser-use:8080", "type": "agent", "capabilities": ["web_automation"]},
            "skyvern": {"url": "http://skyvern:8080", "type": "agent", "capabilities": ["browser_automation"]},
            "dify": {"url": "http://dify:5000", "type": "agent", "capabilities": ["workflow"]},
            "agentgpt": {"url": "http://agentgpt:3000", "type": "agent", "capabilities": ["autonomous"]},
            "privategpt": {"url": "http://privategpt:8080", "type": "agent", "capabilities": ["private_docs"]},
            "llamaindex": {"url": "http://llamaindex:8080", "type": "agent", "capabilities": ["indexing"]},
            "flowise": {"url": "http://flowise:3000", "type": "agent", "capabilities": ["visual_flow"]},
            "shellgpt": {"url": "http://shellgpt:8080", "type": "agent", "capabilities": ["terminal"]},
            "pentestgpt": {"url": "http://pentestgpt:8080", "type": "agent", "capabilities": ["security_test"]},
            "finrobot": {"url": "http://finrobot:8080", "type": "agent", "capabilities": ["finance"]},
            "opendevin": {"url": "http://opendevin:3000", "type": "agent", "capabilities": ["ai_coding"]},
            "documind": {"url": "http://documind:8000", "type": "agent", "capabilities": ["document_processing"]},
            
            # ML Frameworks
            "pytorch": {"url": "http://pytorch:8888", "type": "ml_framework"},
            "tensorflow": {"url": "http://tensorflow:8888", "type": "ml_framework"},
            "jax": {"url": "http://jax:8080", "type": "ml_framework"},
            
            # Monitoring
            "prometheus": {"url": "http://prometheus:9090", "type": "monitoring"},
            "grafana": {"url": "http://grafana:3000", "type": "monitoring"},
            "loki": {"url": "http://loki:3100", "type": "logging"},
        }
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """Initialize the service registry"""
        self._session = aiohttp.ClientSession()
        await self._health_check_all()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session:
            await self._session.close()
    
    async def _health_check_all(self):
        """Check health of all services"""
        tasks = []
        for service_name, service_info in self.services.items():
            if service_info.get("type") in ["agent", "orchestrator"]:
                tasks.append(self._check_service_health(service_name, service_info))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        healthy_count = sum(1 for r in results if r is True)
        logger.info(f"Service health check: {healthy_count}/{len(tasks)} services healthy")
    
    async def _check_service_health(self, name: str, info: Dict[str, Any]) -> bool:
        """Check if a service is healthy"""
        try:
            url = f"{info['url']}/health"
            async with self._session.get(url, timeout=5) as response:
                info['healthy'] = response.status == 200
                info['last_check'] = datetime.now()
                return info['healthy']
        except Exception as e:
            logger.warning(f"Service {name} health check failed: {e}")
            info['healthy'] = False
            info['last_check'] = datetime.now()
            return False
    
    def get_service(self, name: str) -> Optional[Dict[str, Any]]:
        """Get service information"""
        return self.services.get(name)
    
    def get_services_by_type(self, service_type: str) -> List[Dict[str, Any]]:
        """Get all services of a specific type"""
        return [
            {**info, "name": name}
            for name, info in self.services.items()
            if info.get("type") == service_type
        ]
    
    def get_services_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get all services with a specific capability"""
        return [
            {**info, "name": name}
            for name, info in self.services.items()
            if capability in info.get("capabilities", [])
        ]
    
    async def call_service(self, service_name: str, endpoint: str, method: str = "GET", **kwargs) -> Any:
        """Call a service endpoint"""
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service {service_name} not found")
        
        url = f"{service['url']}{endpoint}"
        
        try:
            async with self._session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Service call failed: {service_name} returned {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error calling service {service_name}: {e}")
            return None
    
    def get_best_service_for_task(self, task_type: str) -> Optional[str]:
        """Get the best service for a specific task type"""
        capability_mapping = {
            "code_generation": ["gpt-engineer", "aider", "opendevin"],
            "code_completion": ["tabbyml", "aider"],
            "task_automation": ["autogpt", "agentgpt"],
            "multi_agent": ["crewai", "autogen"],
            "web_automation": ["browser-use", "skyvern"],
            "document_processing": ["documind", "privategpt"],
            "security": ["semgrep", "pentestgpt"],
            "finance": ["finrobot"],
            "orchestration": ["localagi"],
        }
        
        services = capability_mapping.get(task_type, [])
        
        # Return first healthy service
        for service_name in services:
            service = self.get_service(service_name)
            if service and service.get('healthy', False):
                return service_name
        
        # Return first available if none are healthy
        return services[0] if services else None

# Global service registry instance
service_registry = ServiceRegistry()