"""
SutazAI - Example Optimized Agent
Demonstrates how to use the new BaseAgent class
Replaces 100+ duplicate agent implementations
"""

from typing import Dict, List, Any
from .base_agent_optimized import BaseAgent, AgentConfig


class ExampleOptimizedAgent(BaseAgent):
    """
    Example implementation showing how to create optimized agents
    
    This replaces the pattern found in 100+ agent directories:
    - /agents/data-analysis-engineer/
    - /agents/observability-monitoring-engineer/  
    - /agents/product-strategy-architect/
    - etc.
    
    All agents now inherit from BaseAgent for:
    - Zero code duplication
    - Consistent health checks
    - Standardized metrics
    - Unified logging
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Agent-specific state
        self.processing_queue = []
        self.completed_tasks = 0
        
    async def initialize(self):
        """Custom initialization for this agent type"""
        self.logger.info("Initializing Example Agent", 
                        agent_type=self.config.agent_type)
        
        # Example: Subscribe to specific Redis channels
        if self.redis_client:
            await self.redis_set(
                f"agent:{self.config.agent_id}:status", 
                "initialized"
            )
            
        # Example: Register with service discovery
        registration_data = {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "port": self.config.port,
            "capabilities": await self.get_capabilities()
        }
        
        try:
            await self.make_request(
                "POST",
                f"{self.config.api_endpoint}/api/v1/agents/register",
                json=registration_data
            )
            self.logger.info("Registered with service discovery")
        except Exception as e:
            self.logger.warning("Failed to register with service discovery", error=str(e))
            
    async def cleanup(self):
        """Custom cleanup for this agent type"""
        self.logger.info("Cleaning up Example Agent")
        
        # Example: Unregister from service discovery
        try:
            await self.make_request(
                "DELETE",
                f"{self.config.api_endpoint}/api/v1/agents/{self.config.agent_id}"
            )
        except Exception as e:
            self.logger.warning("Failed to unregister", error=str(e))
            
    async def check_health(self) -> str:
        """Custom health check logic"""
        try:
            # Example health checks specific to this agent
            if len(self.processing_queue) > 1000:
                return "queue_overloaded"
                
            if self.error_count > 10:
                return "error_threshold_exceeded"
                
            # Check external dependencies
            response = await self.make_request(
                "GET", 
                f"{self.config.api_endpoint}/health"
            )
            
            if response.get("status") != "healthy":
                return "backend_unhealthy"
                
            return "healthy"
            
        except Exception as e:
            return f"health_check_failed: {str(e)}"
            
    async def get_capabilities(self) -> List[str]:
        """Return agent capabilities"""
        return [
            "task_processing",
            "data_analysis", 
            "report_generation",
            "monitoring",
            "alerting"
        ]
        
    # Custom agent endpoints would be added to self.app in __init__
    def add_custom_endpoints(self):
        """Add agent-specific endpoints"""
        
        @self.app.post("/tasks")
        async def create_task(task_data: Dict[str, Any]):
            """Custom endpoint for task creation"""
            task_id = len(self.processing_queue) + 1
            
            task = {
                "id": task_id,
                "data": task_data,
                "status": "queued",
                "created_at": time.time()
            }
            
            self.processing_queue.append(task)
            
            await self.log_activity("task_created", {
                "task_id": task_id,
                "queue_size": len(self.processing_queue)
            })
            
            return {"task_id": task_id, "status": "queued"}
            
        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: int):
            """Get task status"""
            task = next((t for t in self.processing_queue if t["id"] == task_id), None)
            
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
                
            return task
            
        @self.app.get("/queue/status")
        async def get_queue_status():
            """Get processing queue status"""
            return {
                "queue_size": len(self.processing_queue),
                "completed_tasks": self.completed_tasks,
                "agent_id": self.config.agent_id
            }


# Example usage - this would be in the agent's main.py or app.py
if __name__ == "__main__":
    import os
    
    config = AgentConfig(
        agent_id=os.getenv("AGENT_ID", "example-agent-001"),
        agent_type=os.getenv("AGENT_TYPE", "example-optimized"),
        agent_name="Example Optimized Agent",
        port=int(os.getenv("PORT", "8080")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        api_endpoint=os.getenv("API_ENDPOINT", "http://localhost:8000")
    )
    
    agent = ExampleOptimizedAgent(config)
    agent.add_custom_endpoints()  # Add custom endpoints
    agent.run()  # Start the server