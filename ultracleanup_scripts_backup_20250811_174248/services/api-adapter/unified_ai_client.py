#!/usr/bin/env python3
"""
Purpose: Unified API client for distributed AI services
Usage: from unified_ai_client import UnifiedAIClient
Requirements: requests, redis, consul
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import requests
import redis
import consul
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """Types of AI services"""
    LLM = "llm"
    VECTOR_DB = "vector_db"
    AGENT = "agent"
    WORKFLOW = "workflow"
    TOOL = "tool"

@dataclass
class ServiceInfo:
    """Service information"""
    name: str
    type: ServiceType
    endpoint: str
    health_endpoint: str
    version: str
    capabilities: List[str]

class UnifiedAIClient:
    """Unified client for all AI services"""
    
    def __init__(self, api_gateway_url: str = "http://localhost:8000"):
        self.api_gateway_url = api_gateway_url
        self.consul_client = consul.Consul(host='localhost', port=8500)
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Setup HTTP client with retry logic
        self.session = requests.Session()
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Service registry
        self.services = {
            # LLM Services
            'ollama': ServiceInfo(
                name='ollama',
                type=ServiceType.LLM,
                endpoint=f'{api_gateway_url}/api/ollama',
                health_endpoint=f'{api_gateway_url}/api/ollama/api/tags',
                version='latest',
                capabilities=['chat', 'completion', 'embedding']
            ),
            
            # Vector Databases
            'chromadb': ServiceInfo(
                name='chromadb',
                type=ServiceType.VECTOR_DB,
                endpoint=f'{api_gateway_url}/api/chromadb',
                health_endpoint=f'{api_gateway_url}/api/chromadb/api/v1/heartbeat',
                version='latest',
                capabilities=['store', 'search', 'delete']
            ),
            'qdrant': ServiceInfo(
                name='qdrant',
                type=ServiceType.VECTOR_DB,
                endpoint=f'{api_gateway_url}/api/qdrant',
                health_endpoint=f'{api_gateway_url}/api/qdrant/health',
                version='latest',
                capabilities=['store', 'search', 'delete', 'filter']
            ),
            
            # Agent Services
            'langchain': ServiceInfo(
                name='langchain',
                type=ServiceType.AGENT,
                endpoint=f'{api_gateway_url}/api/langchain',
                health_endpoint=f'{api_gateway_url}/api/langchain/health',
                version='1.0.0',
                capabilities=['chain', 'agent', 'memory']
            ),
            'autogpt': ServiceInfo(
                name='autogpt',
                type=ServiceType.AGENT,
                endpoint=f'{api_gateway_url}/api/autogpt',
                health_endpoint=f'{api_gateway_url}/api/autogpt/health',
                version='latest',
                capabilities=['autonomous', 'task', 'planning']
            ),
            
            # Workflow Services
            'n8n': ServiceInfo(
                name='n8n',
                type=ServiceType.WORKFLOW,
                endpoint=f'{api_gateway_url}/api/n8n',
                health_endpoint=f'{api_gateway_url}/api/n8n/healthz',
                version='latest',
                capabilities=['workflow', 'automation', 'integration']
            )
        }
    
    def discover_services(self) -> Dict[str, ServiceInfo]:
        """Discover available services from Consul"""
        try:
            _, services = self.consul_client.health.service("ai-services", passing=True)
            
            for service in services:
                service_name = service['Service']['Service']
                if service_name not in self.services:
                    # Dynamically add discovered services
                    self.services[service_name] = ServiceInfo(
                        name=service_name,
                        type=ServiceType.AGENT,  # Default type
                        endpoint=f"{self.api_gateway_url}/api/{service_name}",
                        health_endpoint=f"{self.api_gateway_url}/api/{service_name}/health",
                        version='unknown',
                        capabilities=[]
                    )
            
        except Exception as e:
            logger.error(f"Failed to discover services: {e}")
        
        return self.services
    
    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        if service_name not in self.services:
            return False
        
        try:
            service = self.services[service_name]
            response = self.session.get(service.health_endpoint, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_healthy_services(self, service_type: Optional[ServiceType] = None) -> List[str]:
        """Get list of healthy services"""
        healthy = []
        
        for name, info in self.services.items():
            if service_type and info.type != service_type:
                continue
            
            if self.check_service_health(name):
                healthy.append(name)
        
        return healthy
    
    # LLM Operations
    def chat_completion(self, 
                       prompt: str, 
                       model: str = "tinyllama",
                       service: str = "ollama",
                       **kwargs) -> Dict[str, Any]:
        """Execute chat completion"""
        cache_key = f"chat:{service}:{model}:{hash(prompt)}"
        
        # Check cache
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Make request
        endpoint = f"{self.services[service].endpoint}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = self.session.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Cache result
            self.redis_client.set(cache_key, json.dumps(result), ex=3600)
            
            return result
        
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    # Vector Database Operations
    def vector_store(self,
                    documents: List[Dict[str, Any]],
                    collection: str,
                    service: str = "chromadb") -> Dict[str, Any]:
        """Store vectors in database"""
        endpoint = f"{self.services[service].endpoint}/api/v1/collections/{collection}/add"
        
        try:
            response = self.session.post(endpoint, json={"documents": documents})
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Vector store failed: {e}")
            raise
    
    def vector_search(self,
                     query: str,
                     collection: str,
                     k: int = 10,
                     service: str = "chromadb") -> List[Dict[str, Any]]:
        """Search vectors in database"""
        endpoint = f"{self.services[service].endpoint}/api/v1/collections/{collection}/query"
        
        payload = {
            "query_texts": [query],
            "n_results": k
        }
        
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    # Agent Operations
    def execute_chain(self,
                     prompt: str,
                     chain_type: str = "simple",
                     service: str = "langchain",
                     **kwargs) -> Dict[str, Any]:
        """Execute an agent chain"""
        endpoint = f"{self.services[service].endpoint}/execute"
        
        payload = {
            "prompt": prompt,
            "chain_type": chain_type,
            **kwargs
        }
        
        try:
            response = self.session.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Chain execution failed: {e}")
            raise
    
    def create_autonomous_task(self,
                              goal: str,
                              service: str = "autogpt") -> Dict[str, Any]:
        """Create an autonomous task"""
        endpoint = f"{self.services[service].endpoint}/tasks"
        
        payload = {
            "goal": goal,
            "auto_execute": True
        }
        
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Task creation failed: {e}")
            raise
    
    # Workflow Operations
    def create_workflow(self,
                       name: str,
                       nodes: List[Dict[str, Any]],
                       service: str = "n8n") -> Dict[str, Any]:
        """Create a workflow"""
        endpoint = f"{self.services[service].endpoint}/workflows"
        
        payload = {
            "name": name,
            "nodes": nodes,
            "active": True
        }
        
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            raise
    
    def execute_workflow(self,
                        workflow_id: str,
                        data: Dict[str, Any],
                        service: str = "n8n") -> Dict[str, Any]:
        """Execute a workflow"""
        endpoint = f"{self.services[service].endpoint}/workflows/{workflow_id}/execute"
        
        try:
            response = self.session.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    # Utility Methods
    def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get service metrics"""
        metrics_key = f"metrics:{service_name}"
        
        metrics = {
            "requests": self.redis_client.hget("metrics:requests", service_name) or 0,
            "tokens": self.redis_client.hget("metrics:tokens", service_name) or 0,
            "avg_execution_time": 0
        }
        
        # Calculate average execution time
        exec_times = self.redis_client.lrange("metrics:execution_times", 0, -1)
        if exec_times:
            times = [float(t) for t in exec_times]
            metrics["avg_execution_time"] = sum(times) / len(times)
        
        return metrics
    
    def batch_process(self,
                     tasks: List[Dict[str, Any]],
                     service: str,
                     max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Process multiple tasks in batch"""
        results = []
        
        # Simple batch processing (could be enhanced with asyncio)
        for task in tasks:
            try:
                if task['type'] == 'chat':
                    result = self.chat_completion(**task['params'], service=service)
                elif task['type'] == 'chain':
                    result = self.execute_chain(**task['params'], service=service)
                else:
                    result = {"error": f"Unknown task type: {task['type']}"}
                
                results.append(result)
                
            except Exception as e:
                results.append({"error": str(e)})
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = UnifiedAIClient()
    
    # Discover services
    services = client.discover_services()
    print(f"Discovered {len(services)} services")
    
    # Check health
    healthy = client.get_healthy_services()
    print(f"Healthy services: {healthy}")
    
    # Example: Chat completion
    try:
        response = client.chat_completion(
            prompt="What is the capital of France?",
            model="tinyllama"
        )
        print(f"Chat response: {response}")
    except Exception as e:
        print(f"Chat failed: {e}")
    
    # Example: Execute chain
    try:
        response = client.execute_chain(
            prompt="Explain advanced computing in simple terms",
            chain_type="simple"
        )
        print(f"Chain response: {response}")
    except Exception as e:
        print(f"Chain failed: {e}")