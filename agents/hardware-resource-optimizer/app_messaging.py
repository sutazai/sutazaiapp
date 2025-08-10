#!/usr/bin/env python3
"""
Hardware Resource Optimizer Agent with RabbitMQ Messaging Integration
Enhanced version with real message handling capabilities.
"""
import os
import sys
import asyncio
import psutil
import docker
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/opt/sutazaiapp')

from agents.core.messaging_agent_base import MessagingAgent
from schemas.task_messages import (
    TaskRequestMessage, TaskCompletionMessage, TaskStatusUpdateMessage
)
from schemas.resource_messages import (
    ResourceStatusMessage, ResourceRequestMessage, ResourceAllocationMessage
)
from schemas.base import TaskStatus, ResourceType, Priority
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger(__name__)


class HardwareResourceOptimizerMessaging(MessagingAgent):
    """Hardware Resource Optimizer with full messaging capabilities"""
    
    def __init__(self):
        super().__init__(
            agent_id="hardware_resource_optimizer",
            agent_type="hardware_resource_optimizer",
            capabilities=[
                "resource_monitoring",
                "memory_optimization",
                "disk_cleanup",
                "docker_optimization",
                "cpu_management"
            ],
            version="2.0.0",
            port=8116
        )
        
        # Docker client
        self.docker_client = self._init_docker_client()
        
        # FastAPI app
        self.app = FastAPI(
            title="Hardware Resource Optimizer with Messaging",
            version="2.0.0"
        )
        self._setup_routes()
        
        # Resource monitoring state
        self.resource_cache = {}
        self.monitoring_task = None
        
    def _init_docker_client(self):
        """Initialize Docker client"""
        try:
            client = docker.from_env()
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"Docker unavailable: {e}")
            return None
    
    async def _register_default_handlers(self):
        """Register message handlers"""
        await self.register_handler("task.request", self.handle_task_request)
        await self.register_handler("resource.request", self.handle_resource_request)
        await self.register_handler("system.health", self.handle_health_request)
    
    async def handle_task_request(self, message_data: Dict, raw_message):
        """Handle incoming task requests"""
        try:
            task_type = message_data.get("task_type")
            task_id = message_data.get("task_id")
            
            # Update status
            await self._send_task_status(task_id, TaskStatus.IN_PROGRESS, 0.0)
            
            # Route based on task type
            if task_type == "optimize_memory":
                result = await self.optimize_memory()
            elif task_type == "clean_disk":
                result = await self.clean_disk_space()
            elif task_type == "optimize_docker":
                result = await self.optimize_docker()
            elif task_type == "monitor_resources":
                result = await self.get_resource_status()
            else:
                result = {"error": f"Unknown task type: {task_type}"}
            
            # Send completion
            await self._send_task_completion(task_id, result)
            
        except Exception as e:
            logger.error(f"Error handling task request: {e}")
            await self.send_error(str(e), "TASK_PROCESSING_ERROR", task_id)
    
    async def handle_resource_request(self, message_data: Dict, raw_message):
        """Handle resource allocation requests"""
        try:
            request_id = message_data.get("request_id")
            resources = message_data.get("resources", {})
            
            # Get current resource availability
            status = await self.get_resource_status()
            
            # Check if resources are available
            can_allocate = True
            allocated = {}
            
            for resource_type, amount in resources.items():
                if resource_type == "cpu":
                    available = 100 - status["cpu_percent"]
                    if amount <= available:
                        allocated["cpu"] = amount
                    else:
                        can_allocate = False
                elif resource_type == "memory":
                    available = status["memory_available_gb"]
                    if amount <= available:
                        allocated["memory"] = amount
                    else:
                        can_allocate = False
            
            # Send allocation response
            response = ResourceAllocationMessage(
                source_agent=self.agent_id,
                request_id=request_id,
                allocation_id=f"alloc_{request_id}",
                allocated=can_allocate,
                allocated_resources=allocated if can_allocate else {},
                expires_at=datetime.utcnow(),
                rejection_reason=None if can_allocate else "Insufficient resources"
            )
            
            await self.rabbitmq.publish(response)
            
        except Exception as e:
            logger.error(f"Error handling resource request: {e}")
    
    async def handle_health_request(self, message_data: Dict, raw_message):
        """Handle system health requests"""
        try:
            # Send current resource status
            status = await self.get_resource_status()
            
            msg = ResourceStatusMessage(
                source_agent=self.agent_id,
                total_capacity={
                    ResourceType.CPU: psutil.cpu_count(),
                    ResourceType.MEMORY: psutil.virtual_memory().total / (1024**3),
                    ResourceType.DISK: psutil.disk_usage('/').total / (1024**3)
                },
                available_capacity={
                    ResourceType.CPU: (100 - status["cpu_percent"]) / 100 * psutil.cpu_count(),
                    ResourceType.MEMORY: status["memory_available_gb"],
                    ResourceType.DISK: status["disk_free_gb"]
                },
                allocated_capacity={},
                reserved_capacity={},
                active_allocations=[],
                pending_requests=0,
                allocation_metrics={
                    "cpu_usage_percent": status["cpu_percent"],
                    "memory_usage_percent": status["memory_percent"],
                    "disk_usage_percent": status["disk_percent"]
                }
            )
            
            await self.rabbitmq.publish(msg)
            
        except Exception as e:
            logger.error(f"Error handling health request: {e}")
    
    async def _send_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float,
        message: str = None
    ):
        """Send task status update"""
        try:
            update = TaskStatusUpdateMessage(
                source_agent=self.agent_id,
                task_id=task_id,
                status=status,
                progress=progress,
                message=message
            )
            await self.rabbitmq.publish(update)
        except Exception as e:
            logger.error(f"Failed to send task status: {e}")
    
    async def _send_task_completion(self, task_id: str, result: Dict):
        """Send task completion message"""
        try:
            completion = TaskCompletionMessage(
                source_agent=self.agent_id,
                task_id=task_id,
                status=TaskStatus.COMPLETED if not result.get("error") else TaskStatus.FAILED,
                result=result,
                error=result.get("error"),
                execution_time_seconds=0,  # Would track actual time
                resource_usage={
                    "cpu": psutil.cpu_percent(),
                    "memory": psutil.virtual_memory().percent
                }
            )
            await self.rabbitmq.publish(completion)
        except Exception as e:
            logger.error(f"Failed to send task completion: {e}")
    
    async def get_resource_status(self) -> Dict[str, Any]:
        """Get current system resource status"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.used / disk.total * 100,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting resource status: {e}")
            return {}
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize system memory"""
        try:
            before = psutil.virtual_memory()
            
            # Clear Python garbage collection
            import gc
            gc.collect()
            
            # Clear system caches (if root)
            if os.geteuid() == 0:
                os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
            
            after = psutil.virtual_memory()
            freed = (after.available - before.available) / (1024**3)
            
            return {
                "success": True,
                "memory_freed_gb": max(0, freed),
                "memory_available_gb": after.available / (1024**3)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def clean_disk_space(self) -> Dict[str, Any]:
        """Clean disk space"""
        try:
            before = psutil.disk_usage('/')
            
            # Clean temp files
            import tempfile
            import shutil
            
            temp_dir = tempfile.gettempdir()
            for item in Path(temp_dir).glob("*"):
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass
            
            after = psutil.disk_usage('/')
            freed = (after.free - before.free) / (1024**3)
            
            return {
                "success": True,
                "disk_freed_gb": max(0, freed),
                "disk_free_gb": after.free / (1024**3)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def optimize_docker(self) -> Dict[str, Any]:
        """Optimize Docker resources"""
        try:
            if not self.docker_client:
                return {"success": False, "error": "Docker not available"}
            
            # Prune unused containers, images, volumes
            prune_results = {
                "containers": self.docker_client.containers.prune(),
                "images": self.docker_client.images.prune(),
                "volumes": self.docker_client.volumes.prune()
            }
            
            return {
                "success": True,
                "pruned": prune_results
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def start_resource_monitoring(self):
        """Start periodic resource monitoring"""
        while not self.shutdown_event.is_set():
            try:
                # Get resource status
                status = await self.get_resource_status()
                
                # Publish status
                msg = ResourceStatusMessage(
                    source_agent=self.agent_id,
                    total_capacity={
                        ResourceType.CPU: psutil.cpu_count(),
                        ResourceType.MEMORY: psutil.virtual_memory().total / (1024**3),
                        ResourceType.DISK: psutil.disk_usage('/').total / (1024**3)
                    },
                    available_capacity={
                        ResourceType.CPU: (100 - status["cpu_percent"]) / 100 * psutil.cpu_count(),
                        ResourceType.MEMORY: status["memory_available_gb"],
                        ResourceType.DISK: status["disk_free_gb"]
                    },
                    allocated_capacity={},
                    reserved_capacity={},
                    active_allocations=[],
                    pending_requests=0,
                    allocation_metrics=status
                )
                
                await self.rabbitmq.publish(msg)
                
                # Wait 60 seconds
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health():
            return JSONResponse({
                "status": "healthy",
                "agent": self.agent_id,
                "messaging": await self.rabbitmq.health_check(),
                "resources": await self.get_resource_status()
            })
        
        @self.app.post("/optimize/memory")
        async def optimize_memory():
            result = await self.optimize_memory()
            # Also send via messaging
            await self._send_task_completion("api_memory_optimize", result)
            return JSONResponse(result)
        
        @self.app.post("/optimize/disk")
        async def optimize_disk():
            result = await self.clean_disk_space()
            await self._send_task_completion("api_disk_clean", result)
            return JSONResponse(result)
        
        @self.app.post("/optimize/docker")
        async def optimize_docker():
            result = await self.optimize_docker()
            await self._send_task_completion("api_docker_optimize", result)
            return JSONResponse(result)
        
        @self.app.get("/resources")
        async def get_resources():
            return JSONResponse(await self.get_resource_status())
    
    async def run(self):
        """Run the agent with both messaging and HTTP"""
        try:
            # Initialize messaging
            if not await self.initialize():
                return
            
            # Start consuming messages
            await self.start_consuming()
            
            # Start resource monitoring
            self.monitoring_task = asyncio.create_task(self.start_resource_monitoring())
            
            # Start FastAPI in background
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            # Run server and wait for shutdown
            await asyncio.gather(
                server.serve(),
                self.shutdown_event.wait()
            )
            
        except Exception as e:
            logger.error(f"Agent run error: {e}")
        finally:
            await self.shutdown()


async def main():
    """Main entry point"""
    agent = HardwareResourceOptimizerMessaging()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())