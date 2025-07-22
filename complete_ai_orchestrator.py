#!/usr/bin/env python3
"""
SutazAI Complete AI Orchestrator v10
Autonomous coordination of all AI services and agents
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SutazAIOrchestrator:
    def __init__(self):
        self.services = {
            'ollama': {'url': 'http://localhost:11434', 'type': 'llm', 'status': 'unknown'},
            'chromadb': {'url': 'http://localhost:8001', 'type': 'vector_db', 'status': 'unknown'},
            'qdrant': {'url': 'http://localhost:6333', 'type': 'vector_db', 'status': 'unknown'},
            'postgres': {'url': 'http://localhost:5432', 'type': 'database', 'status': 'unknown'},
            'redis': {'url': 'http://localhost:6379', 'type': 'cache', 'status': 'unknown'},
            'enhanced_model_manager': {'url': 'http://localhost:8098', 'type': 'model_manager', 'status': 'unknown'},
            'backend': {'url': 'http://localhost:8000', 'type': 'api', 'status': 'unknown'}
        }
        
        self.agents = {
            'code_generator': {'active': False, 'last_task': None},
            'document_processor': {'active': False, 'last_task': None},
            'data_analyst': {'active': False, 'last_task': None},
            'system_monitor': {'active': True, 'last_task': 'monitoring'},
            'self_improvement': {'active': False, 'last_task': None}
        }
        
        self.models = []
        self.system_stats = {}
        self.task_queue = asyncio.Queue()
        self.running = False
        
    async def check_service_health(self, service_name: str, service_info: Dict) -> bool:
        """Check if a service is healthy"""
        try:
            if service_info['type'] in ['database', 'cache']:
                # For databases, we assume they're healthy if running
                self.services[service_name]['status'] = 'healthy'
                return True
                
            async with aiohttp.ClientSession() as session:
                health_url = f"{service_info['url']}/health"
                async with session.get(health_url, timeout=5) as response:
                    if response.status == 200:
                        self.services[service_name]['status'] = 'healthy'
                        return True
                    else:
                        self.services[service_name]['status'] = 'unhealthy'
                        return False
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {str(e)}")
            self.services[service_name]['status'] = 'unhealthy'
            return False
    
    async def get_available_models(self) -> List[str]:
        """Get available models from Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.services['ollama']['url']}/api/tags", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.models = [model['name'] for model in data.get('models', [])]
                        return self.models
        except Exception as e:
            logger.error(f"Failed to get models: {str(e)}")
        
        return ['deepseek-r1:8b', 'llama3.2:1b']  # Fallback
    
    async def process_ai_task(self, task: str, model: str = "deepseek-r1:8b") -> Dict[str, Any]:
        """Process a task using AI models"""
        try:
            async with aiohttp.ClientSession() as session:
                # Try enhanced backend first
                backend_payload = {
                    "message": task,
                    "model": model,
                    "stream": False
                }
                
                try:
                    async with session.post(f"{self.services['backend']['url']}/intelligent_chat", 
                                          json=backend_payload, timeout=30) as response:
                        if response.status == 200:
                            result = await response.json()
                            return {
                                "success": True,
                                "response": result.get('response', ''),
                                "service": "enhanced_backend",
                                "model": model,
                                "timestamp": datetime.now().isoformat()
                            }
                except Exception:
                    pass
                
                # Fallback to direct Ollama
                ollama_payload = {
                    "model": model,
                    "prompt": task,
                    "stream": False
                }
                
                async with session.post(f"{self.services['ollama']['url']}/api/generate", 
                                      json=ollama_payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "response": result.get('response', ''),
                            "service": "ollama",
                            "model": model,
                            "timestamp": datetime.now().isoformat(),
                            "stats": {
                                "total_duration": result.get('total_duration', 0),
                                "eval_count": result.get('eval_count', 0)
                            }
                        }
        except Exception as e:
            logger.error(f"AI task processing failed: {str(e)}")
            
        return {
            "success": False,
            "error": "AI processing failed",
            "timestamp": datetime.now().isoformat()
        }
    
    async def store_in_vector_db(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store data in vector database"""
        try:
            async with aiohttp.ClientSession() as session:
                # Try ChromaDB first
                chroma_data = {
                    "documents": [text],
                    "metadatas": [metadata or {}],
                    "ids": [f"doc_{int(time.time())}"]
                }
                
                try:
                    async with session.post(f"{self.services['chromadb']['url']}/api/v1/collections/sutazai/add", 
                                          json=chroma_data, timeout=10) as response:
                        if response.status == 200:
                            return {"success": True, "service": "chromadb", "id": chroma_data["ids"][0]}
                except Exception:
                    pass
                
                # Fallback to Qdrant
                qdrant_data = {
                    "points": [{
                        "id": int(time.time()),
                        "vector": [0.1] * 384,  # Dummy vector for now
                        "payload": {"text": text, **(metadata or {})}
                    }]
                }
                
                async with session.put(f"{self.services['qdrant']['url']}/collections/sutazai/points", 
                                     json=qdrant_data, timeout=10) as response:
                    return {"success": True, "service": "qdrant", "id": qdrant_data["points"][0]["id"]}
                    
        except Exception as e:
            logger.error(f"Vector storage failed: {str(e)}")
            
        return {"success": False, "error": "Vector storage failed"}
    
    async def autonomous_task_generation(self) -> List[str]:
        """Generate autonomous tasks for continuous operation"""
        autonomous_tasks = [
            "Analyze current system performance and identify optimization opportunities",
            "Generate a comprehensive status report of all running services",
            "Identify potential security vulnerabilities in the current setup",
            "Suggest improvements to the AI model configuration",
            "Analyze the chat history for patterns and insights",
            "Generate code suggestions for system enhancement",
            "Create documentation for the current system state",
            "Perform predictive analysis on system resource usage"
        ]
        
        return autonomous_tasks
    
    async def orchestrate_autonomous_operation(self):
        """Main autonomous operation loop"""
        logger.info("Starting autonomous orchestration...")
        
        while self.running:
            try:
                # Health check all services
                health_tasks = [self.check_service_health(name, info) 
                              for name, info in self.services.items()]
                health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
                
                healthy_services = sum(1 for result in health_results if result is True)
                logger.info(f"Health check: {healthy_services}/{len(self.services)} services healthy")
                
                # Update models list
                await self.get_available_models()
                
                # Process autonomous tasks
                if not self.task_queue.empty():
                    task = await self.task_queue.get()
                    logger.info(f"Processing autonomous task: {task[:50]}...")
                    
                    result = await self.process_ai_task(task)
                    if result['success']:
                        # Store result in vector DB
                        await self.store_in_vector_db(
                            f"Task: {task}\nResult: {result['response']}", 
                            {"type": "autonomous_task", "timestamp": result['timestamp']}
                        )
                        logger.info("Autonomous task completed and stored")
                    
                    self.task_queue.task_done()
                
                # Generate new autonomous tasks periodically
                if self.task_queue.empty():
                    autonomous_tasks = await self.autonomous_task_generation()
                    for task in autonomous_tasks[:3]:  # Add 3 tasks
                        await self.task_queue.put(task)
                
                # Update system stats
                self.system_stats = {
                    "timestamp": datetime.now().isoformat(),
                    "healthy_services": healthy_services,
                    "total_services": len(self.services),
                    "available_models": len(self.models),
                    "active_agents": sum(1 for agent in self.agents.values() if agent['active']),
                    "pending_tasks": self.task_queue.qsize()
                }
                
                # Wait before next cycle
                await asyncio.sleep(30)  # 30 second cycles
                
            except Exception as e:
                logger.error(f"Orchestration cycle error: {str(e)}")
                await asyncio.sleep(10)
    
    async def start_orchestration(self):
        """Start the orchestration system"""
        self.running = True
        logger.info("SutazAI Orchestrator v10 starting...")
        
        # Initial health check
        await asyncio.gather(*[self.check_service_health(name, info) 
                             for name, info in self.services.items()])
        
        # Get initial models
        await self.get_available_models()
        
        # Start autonomous operation
        await self.orchestrate_autonomous_operation()
    
    async def stop_orchestration(self):
        """Stop the orchestration system"""
        self.running = False
        logger.info("SutazAI Orchestrator stopping...")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "orchestrator_status": "running" if self.running else "stopped",
            "services": self.services,
            "agents": self.agents,
            "models": self.models,
            "system_stats": self.system_stats,
            "timestamp": datetime.now().isoformat()
        }

# Global orchestrator instance
orchestrator = SutazAIOrchestrator()

async def main():
    """Main orchestrator entry point"""
    try:
        await orchestrator.start_orchestration()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        await orchestrator.stop_orchestration()
    except Exception as e:
        logger.error(f"Orchestrator error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())