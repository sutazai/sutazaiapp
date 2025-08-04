#!/usr/bin/env python3
"""
Agent: hardware-resource-optimizer
Category: optimization
Model Type: Sonnet
"""

import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from agents.core.base_agent_v2 import BaseAgentV2
except ImportError:
    # Try alternative import for Docker container
    sys.path.insert(0, '/opt/sutazaiapp')
    from agents.core.base_agent_v2 import BaseAgentV2
import asyncio
from typing import Dict, Any
import psutil
import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import threading

class HardwareResourceOptimizerAgent(BaseAgentV2):
    """Agent implementation for hardware-resource-optimizer"""
    
    def __init__(self):
        super().__init__(
            config_path="/app/configs/hardware-resource-optimizer_universal.json"
        )
        self.agent_id = "hardware-resource-optimizer"
        self.name = "Hardware Resource Optimizer"
        self.port = int(os.getenv("PORT", "8080"))
        self.description = "Specialized agent for optimizing system performance within hardware constraints"
        
        # Setup FastAPI app
        self.app = FastAPI(title="Hardware Resource Optimizer Agent", version="1.0.0")
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes for the agent"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            health_status = await self.health_check()
            return JSONResponse(content=health_status)
            
        @self.app.get("/metrics")
        async def metrics():
            """Get agent metrics"""
            return JSONResponse(content={
                "tasks_processed": self.metrics.tasks_processed,
                "tasks_failed": self.metrics.tasks_failed,
                "avg_processing_time": self.metrics.avg_processing_time,
                "uptime": time.time() - self.metrics.startup_time.timestamp(),
                "status": self.status.value if hasattr(self, 'status') else "active"
            })
            
        @self.app.post("/monitor")
        async def monitor_resources():
            """Monitor current resource usage"""
            result = await self._monitor_resources()
            return JSONResponse(content=result)
            
        @self.app.post("/optimize")
        async def optimize_performance():
            """Optimize system performance"""
            result = await self._optimize_performance({})
            return JSONResponse(content=result)
            
        @self.app.post("/analyze")
        async def analyze_resources():
            """Analyze resource usage patterns"""
            result = await self._analyze_resources({"analysis_type": "current"})
            return JSONResponse(content=result)
            
        @self.app.post("/capacity-plan")
        async def capacity_planning():
            """Generate capacity planning recommendations"""
            result = await self._plan_capacity({})
            return JSONResponse(content=result)
    
    def start_server(self):
        """Start the FastAPI server in a separate thread"""
        def run_server():
            uvicorn.run(
                self.app, 
                host="0.0.0.0", 
                port=self.port,
                log_level="info"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        return server_thread
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming tasks"""
        try:
            task_type = task.get("type", "unknown")
            
            if task_type == "health":
                return {"status": "healthy", "agent": self.agent_id}
            
            elif task_type == "monitor_resources":
                return await self._monitor_resources()
            
            elif task_type == "optimize_performance":
                return await self._optimize_performance(task)
            
            elif task_type == "resource_analysis":
                return await self._analyze_resources(task)
            
            elif task_type == "capacity_planning":
                return await self._plan_capacity(task)
            
            # Default processing with Ollama
            result = await self._process_with_ollama(task)
            
            return {
                "status": "success",
                "result": result,
                "agent": self.agent_id
            }
            
        except Exception as e:
            self.logger.error(f"Error processing task: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": self.agent_id
            }
    
    async def _monitor_resources(self) -> Dict[str, Any]:
        """Monitor current system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory usage
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Linux/Unix)
            try:
                load_avg = os.getloadavg()
            except (AttributeError, OSError):
                load_avg = [0.0, 0.0, 0.0]
            
            return {
                "status": "success",
                "resource_data": {
                    "cpu": {
                        "usage_percent": cpu_percent,
                        "count": cpu_count,
                        "frequency": cpu_freq._asdict() if cpu_freq else None
                    },
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "used": memory.used,
                        "percent": memory.percent
                    },
                    "swap": {
                        "total": swap.total,
                        "used": swap.used,
                        "percent": swap.percent
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "free": disk.free,
                        "percent": (disk.used / disk.total) * 100
                    },
                    "network": {
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv
                    },
                    "system": {
                        "process_count": process_count,
                        "load_average": load_avg
                    }
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring resources: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _optimize_performance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system performance based on current conditions"""
        try:
            # Get current resource state
            resources = await self._monitor_resources()
            resource_data = resources.get("resource_data", {})
            
            recommendations = []
            
            # CPU optimization recommendations
            cpu_usage = resource_data.get("cpu", {}).get("usage_percent", 0)
            if cpu_usage > 80:
                recommendations.append({
                    "type": "cpu_optimization",
                    "priority": "high",
                    "recommendation": "High CPU usage detected. Consider scaling resources or optimizing CPU-intensive processes."
                })
            
            # Memory optimization recommendations
            memory_percent = resource_data.get("memory", {}).get("percent", 0)
            if memory_percent > 85:
                recommendations.append({
                    "type": "memory_optimization",
                    "priority": "high",
                    "recommendation": "High memory usage detected. Consider increasing memory or optimizing memory-intensive applications."
                })
            
            # Disk optimization recommendations
            disk_percent = resource_data.get("disk", {}).get("percent", 0)
            if disk_percent > 85:
                recommendations.append({
                    "type": "disk_optimization",
                    "priority": "medium",
                    "recommendation": "High disk usage detected. Consider cleaning up temporary files or expanding storage."
                })
            
            # Load average recommendations
            load_avg = resource_data.get("system", {}).get("load_average", [0, 0, 0])
            cpu_count = resource_data.get("cpu", {}).get("count", 1)
            if load_avg[0] > cpu_count * 0.8:
                recommendations.append({
                    "type": "load_optimization",
                    "priority": "high",
                    "recommendation": f"High system load detected ({load_avg[0]:.2f}). Consider distributing workload or scaling resources."
                })
            
            return {
                "status": "success",
                "optimization_result": {
                    "recommendations": recommendations,
                    "current_resources": resource_data,
                    "optimization_timestamp": time.time()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing performance: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _analyze_resources(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource usage patterns and trends"""
        try:
            analysis_type = task.get("analysis_type", "current")
            
            if analysis_type == "current":
                resources = await self._monitor_resources()
                resource_data = resources.get("resource_data", {})
                
                # Analyze current state
                analysis = {
                    "cpu_health": "optimal" if resource_data.get("cpu", {}).get("usage_percent", 0) < 70 else "stressed",
                    "memory_health": "optimal" if resource_data.get("memory", {}).get("percent", 0) < 80 else "stressed",
                    "disk_health": "optimal" if resource_data.get("disk", {}).get("percent", 0) < 80 else "stressed",
                    "overall_health": "good"
                }
                
                # Determine overall health
                stress_indicators = sum(1 for health in analysis.values() if health == "stressed")
                if stress_indicators > 1:
                    analysis["overall_health"] = "critical"
                elif stress_indicators == 1:
                    analysis["overall_health"] = "warning"
                
                return {
                    "status": "success",
                    "analysis": analysis,
                    "resource_data": resource_data,
                    "analysis_timestamp": time.time()
                }
            
            return {
                "status": "success",
                "message": f"Analysis type '{analysis_type}' not yet implemented"
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing resources: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _plan_capacity(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Plan capacity based on current usage and future projections"""
        try:
            resources = await self._monitor_resources()
            resource_data = resources.get("resource_data", {})
            
            # Simple capacity planning based on current usage
            cpu_usage = resource_data.get("cpu", {}).get("usage_percent", 0)
            memory_usage = resource_data.get("memory", {}).get("percent", 0)
            disk_usage = resource_data.get("disk", {}).get("percent", 0)
            
            capacity_plan = {
                "current_utilization": {
                    "cpu": cpu_usage,
                    "memory": memory_usage,
                    "disk": disk_usage
                },
                "capacity_recommendations": []
            }
            
            # CPU capacity planning
            if cpu_usage > 60:
                capacity_plan["capacity_recommendations"].append({
                    "resource": "cpu",
                    "current_usage": cpu_usage,
                    "recommendation": "Consider CPU scaling within 30 days",
                    "priority": "medium" if cpu_usage < 80 else "high"
                })
            
            # Memory capacity planning
            if memory_usage > 70:
                capacity_plan["capacity_recommendations"].append({
                    "resource": "memory",
                    "current_usage": memory_usage,
                    "recommendation": "Consider memory upgrade or optimization",
                    "priority": "medium" if memory_usage < 85 else "high"
                })
            
            # Disk capacity planning
            if disk_usage > 70:
                capacity_plan["capacity_recommendations"].append({
                    "resource": "disk",
                    "current_usage": disk_usage,
                    "recommendation": "Consider disk expansion or cleanup",
                    "priority": "medium" if disk_usage < 85 else "high"
                })
            
            return {
                "status": "success",
                "capacity_plan": capacity_plan,
                "planning_timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error planning capacity: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _process_with_ollama(self, task: Dict[str, Any]) -> Any:
        """Process task using Ollama model"""
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        
        # Create a hardware optimization prompt
        prompt = f"""
        As a hardware resource optimizer, analyze this task and provide optimization recommendations:
        Task: {json.dumps(task, indent=2)}
        
        Please provide:
        1. Analysis of the request
        2. Hardware optimization recommendations
        3. Performance improvement suggestions
        4. Resource allocation strategies
        """
        
        try:
            # Use the parent class's Ollama integration if available
            if hasattr(self, 'query_ollama'):
                response = await self.query_ollama(prompt, model)
                return {
                    "ollama_response": response,
                    "model_used": model,
                    "task": task
                }
        except Exception as e:
            self.logger.error(f"Ollama processing error: {e}")
        
        # Fallback response
        return {
            "message": f"Processed by {self.name} using model {model}",
            "task": task,
            "optimization_note": "Advanced optimization requires specific task parameters"
        }

    def start(self):
        """Start the agent with FastAPI server"""
        # Start the FastAPI server
        server_thread = self.start_server()
        
        # Start the main agent loop
        try:
            self.run()
        except KeyboardInterrupt:
            self.logger.info("Agent stopped by user")
        except Exception as e:
            self.logger.error(f"Agent error: {e}")
            raise

if __name__ == "__main__":
    agent = HardwareResourceOptimizerAgent()
    agent.start()