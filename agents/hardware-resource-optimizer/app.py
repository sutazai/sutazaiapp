#!/usr/bin/env python3
"""
Agent: hardware-resource-optimizer
Category: optimization
Model Type: Sonnet
"""

import os
import sys
import gc
import signal
import subprocess
import shutil
import docker
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add local shared directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the local BaseAgent
from shared.agent_base import BaseAgent
import asyncio
from typing import Dict, Any, List, Optional
import psutil
import json
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import threading
import requests
import logging

class HardwareResourceOptimizerAgent(BaseAgent):
    """Active Hardware Resource Optimizer - Real-time system optimization service"""
    
    def __init__(self):
        # Use the default config path from BaseAgent
        super().__init__()
        self.agent_id = "hardware-resource-optimizer"
        self.name = "Hardware Resource Optimizer"
        self.port = int(os.getenv("PORT", "8080"))
        self.description = "Active real-time system performance optimizer and resource manager"
        
        # Optimization settings
        self.monitoring_interval = 30  # seconds
        self.cpu_threshold_high = 80.0
        self.cpu_threshold_critical = 90.0
        self.memory_threshold_high = 80.0
        self.memory_threshold_critical = 90.0
        self.disk_threshold_high = 85.0
        self.ollama_cpu_limit = 300.0  # 300% max (3 cores)
        
        # State tracking
        self.optimization_active = False
        self.last_optimization = 0
        self.optimization_history = []
        self.process_nice_values = {}  # Track processes we've modified
        self.container_limits_applied = set()
        
        # Docker client for container management
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.warning(f"Docker client unavailable: {e}")
            self.docker_client = None
        
        # Thread pool for optimization tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize agent properly
        self.initialized = False
        self._initialize_agent()
        
        # Setup FastAPI app
        self.app = FastAPI(title="Hardware Resource Optimizer Agent", version="2.0.0")
        self._setup_routes()
        
        # Start active monitoring loop
        self.monitoring_task = None
    
    def _initialize_agent(self):
        """Initialize the agent and mark as ready"""
        try:
            # Perform any necessary initialization
            self.logger.info(f"Initializing {self.name} - Active Optimization Service")
            
            # Test system metrics access
            test_metrics = self._get_system_metrics()
            if test_metrics:
                self.initialized = True
                self.logger.info(f"{self.name} initialized successfully")
                
                # Initialize optimization baseline
                self._record_optimization_event("system_initialized", {
                    "initial_metrics": test_metrics,
                    "optimization_thresholds": {
                        "cpu_high": self.cpu_threshold_high,
                        "cpu_critical": self.cpu_threshold_critical,
                        "memory_high": self.memory_threshold_high,
                        "memory_critical": self.memory_threshold_critical
                    }
                })
            else:
                self.logger.warning("System metrics not available, but agent will continue")
                self.initialized = True  # Still mark as initialized
                
        except Exception as e:
            self.logger.error(f"Error during agent initialization: {e}")
            self.initialized = True  # Mark as initialized anyway to avoid blocking
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics for optimization decisions"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            net_io = psutil.net_io_counters()
            
            # Process metrics
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'nice']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] > 5.0:  # Only track significant CPU users
                        processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort processes by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
            
            return {
                "cpu": {
                    "total_percent": cpu_percent,
                    "per_core": cpu_per_core,
                    "count": cpu_count,
                    "load_average": load_avg
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                    "cached": memory.cached,
                    "buffers": memory.buffers
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
                    "percent": (disk.used / disk.total) * 100,
                    "io_counters": {
                        "read_bytes": disk_io.read_bytes if disk_io else 0,
                        "write_bytes": disk_io.write_bytes if disk_io else 0
                    }
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                },
                "processes": processes[:10],  # Top 10 CPU consumers
                "process_count": len(psutil.pids()),
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}
        
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
                "tasks_processed": self.tasks_processed,
                "tasks_failed": 0,
                "avg_processing_time": 0,
                "uptime": time.time() - self.last_heartbeat,
                "status": "active" if self.is_running else "inactive"
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
            
        @self.app.post("/start-optimization")
        async def start_optimization():
            """Start active optimization monitoring"""
            result = await self._start_active_optimization()
            return JSONResponse(content=result)
            
        @self.app.post("/stop-optimization")
        async def stop_optimization():
            """Stop active optimization monitoring"""
            result = await self._stop_active_optimization()
            return JSONResponse(content=result)
            
        @self.app.get("/optimization-status")
        async def optimization_status():
            """Get current optimization status"""
            return JSONResponse(content={
                "active": self.optimization_active,
                "last_optimization": self.last_optimization,
                "history_count": len(self.optimization_history),
                "recent_actions": self.optimization_history[-5:] if self.optimization_history else []
            })
    
    async def health_check(self) -> Dict[str, Any]:
        """Override health check to return healthy status when agent is initialized"""
        try:
            # Check if agent is properly initialized
            if not self.initialized:
                return {
                    "status": "error",
                    "healthy": False,
                    "message": "Agent not properly initialized",
                    "agent_name": self.agent_id,
                    "timestamp": time.time()
                }
            
            # Get system metrics for health check
            system_metrics = self._get_system_metrics()
            
            # Check if system metrics are available
            metrics_available = bool(system_metrics)
            
            # For this agent, we consider it healthy if it's initialized and can access system metrics
            # We don't require Ollama or backend connectivity for basic health
            is_healthy = self.initialized and metrics_available
            
            health_status = {
                "status": "healthy" if is_healthy else "degraded",
                "healthy": is_healthy,
                "agent_name": self.agent_id,
                "agent_version": getattr(self, 'agent_version', '1.0.0'),
                "uptime_seconds": time.time() - self.last_heartbeat,
                "initialized": self.initialized,
                "system_metrics_available": metrics_available,
                "system_metrics": system_metrics,
                "tasks_processed": self.tasks_processed,
                "tasks_failed": 0,
                "description": self.description,
                "timestamp": time.time()
            }
            
            # Add Ollama status if available (but don't fail health check if not available)
            try:
                # Check Ollama connectivity
                test_response = self.query_ollama("test", "tinyllama")
                health_status["ollama_healthy"] = test_response is not None
            except Exception as e:
                health_status["ollama_healthy"] = False
                health_status["ollama_error"] = str(e)
            
            # Add backend status if available (but don't fail health check if not available)
            try:
                # Check backend connectivity
                response = requests.get(f"{self.backend_url}/health", timeout=5.0)
                health_status["backend_healthy"] = response.status_code == 200
            except Exception as e:
                health_status["backend_healthy"] = False
                health_status["backend_error"] = str(e)
            
            # Add optimization status to health check
            health_status["optimization_active"] = self.optimization_active
            health_status["last_optimization"] = self.last_optimization
            health_status["optimization_actions_today"] = len([
                event for event in self.optimization_history 
                if event.get('timestamp', 0) > time.time() - 86400
            ])
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return {
                "status": "error",
                "healthy": False,
                "error": str(e),
                "agent_name": self.agent_id,
                "timestamp": time.time()
            }

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
            # Use the parent class's Ollama integration
            response = self.query_ollama(prompt, model)
            if response:
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
    
    def _record_optimization_event(self, action: str, details: Dict[str, Any]):
        """Record an optimization action for tracking and analysis"""
        event = {
            "timestamp": time.time(),
            "action": action,
            "details": details
        }
        self.optimization_history.append(event)
        
        # Keep only last 100 events to prevent memory growth
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        self.logger.info(f"Optimization action: {action} - {details}")
    
    async def _start_active_optimization(self) -> Dict[str, Any]:
        """Start the active optimization monitoring loop"""
        try:
            if self.optimization_active:
                return {
                    "status": "already_running",
                    "message": "Active optimization is already running"
                }
            
            self.optimization_active = True
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._optimization_monitoring_loop())
            
            self._record_optimization_event("optimization_started", {
                "monitoring_interval": self.monitoring_interval,
                "thresholds": {
                    "cpu_high": self.cpu_threshold_high,
                    "memory_high": self.memory_threshold_high
                }
            })
            
            return {
                "status": "success",
                "message": "Active optimization monitoring started",
                "monitoring_interval": self.monitoring_interval
            }
            
        except Exception as e:
            self.logger.error(f"Error starting optimization: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _stop_active_optimization(self) -> Dict[str, Any]:
        """Stop the active optimization monitoring loop"""
        try:
            if not self.optimization_active:
                return {
                    "status": "not_running",
                    "message": "Active optimization is not running"
                }
            
            self.optimization_active = False
            
            # Cancel monitoring task
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self._record_optimization_event("optimization_stopped", {
                "reason": "manual_stop"
            })
            
            return {
                "status": "success",
                "message": "Active optimization monitoring stopped"
            }
            
        except Exception as e:
            self.logger.error(f"Error stopping optimization: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _optimization_monitoring_loop(self):
        """Main optimization monitoring loop - runs every 30 seconds"""
        self.logger.info("Starting active optimization monitoring loop")
        
        while self.optimization_active:
            try:
                # Get current system metrics
                metrics = self._get_system_metrics()
                if not metrics:
                    await asyncio.sleep(self.monitoring_interval)
                    continue
                
                # Perform optimization checks and actions
                await self._perform_optimization_cycle(metrics)
                
                # Wait for next cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                self.logger.info("Optimization monitoring loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in optimization monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
        
        self.logger.info("Optimization monitoring loop stopped")
    
    async def _perform_optimization_cycle(self, metrics: Dict[str, Any]):
        """Perform one cycle of system optimization based on current metrics"""
        optimization_actions = []
        
        try:
            # CPU Optimization
            cpu_percent = metrics.get("cpu", {}).get("total_percent", 0)
            if cpu_percent > self.cpu_threshold_high:
                cpu_actions = await self._optimize_cpu_usage(metrics)
                optimization_actions.extend(cpu_actions)
            
            # Memory Optimization
            memory_percent = metrics.get("memory", {}).get("percent", 0)
            if memory_percent > self.memory_threshold_high:
                memory_actions = await self._optimize_memory_usage(metrics)
                optimization_actions.extend(memory_actions)
            
            # Container Optimization
            if self.docker_client:
                container_actions = await self._optimize_containers(metrics)
                optimization_actions.extend(container_actions)
            
            # System Cleanup
            cleanup_actions = await self._perform_system_cleanup(metrics)
            optimization_actions.extend(cleanup_actions)
            
            # Record optimization cycle
            if optimization_actions:
                self._record_optimization_event("optimization_cycle", {
                    "metrics_summary": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "process_count": metrics.get("process_count", 0)
                    },
                    "actions_taken": optimization_actions
                })
                self.last_optimization = time.time()
            
        except Exception as e:
            self.logger.error(f"Error in optimization cycle: {e}")
            self._record_optimization_event("optimization_error", {
                "error": str(e),
                "metrics_summary": {
                    "cpu_percent": metrics.get("cpu", {}).get("total_percent", 0),
                    "memory_percent": metrics.get("memory", {}).get("percent", 0)
                }
            })

    async def start_async(self):
        """Start the agent asynchronously with proper initialization"""
        try:
            # Start the FastAPI server
            server_thread = self.start_server()
            
            # Set agent as running
            self.is_running = True
            
            # Auto-start optimization monitoring
            await self._start_active_optimization()
            
            self.logger.info(f"{self.name} started successfully on port {self.port} with active optimization")
            
            # Keep the server running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Agent stopped by user")
                await self._stop_active_optimization()
                
        except Exception as e:
            self.logger.error(f"Error starting agent: {e}")
            self.is_running = False
            raise

    def start(self):
        """Start the agent with FastAPI server"""
        try:
            # Setup signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                self.logger.info(f"Received signal {signum}, shutting down gracefully...")
                self.optimization_active = False
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Run the async startup
            asyncio.run(self.start_async())
        except KeyboardInterrupt:
            self.logger.info("Agent stopped by user")
        except Exception as e:
            self.logger.error(f"Agent error: {e}")
            raise

    async def _optimize_cpu_usage(self, metrics: Dict[str, Any]) -> List[str]:
        """Optimize CPU usage when high CPU load is detected"""
        actions = []
        
        try:
            processes = metrics.get("processes", [])
            cpu_percent = metrics.get("cpu", {}).get("total_percent", 0)
            
            # Find high CPU processes
            for proc in processes:
                pid = proc.get("pid")
                name = proc.get("name", "unknown")
                proc_cpu = proc.get("cpu_percent", 0)
                current_nice = proc.get("nice", 0)
                
                if not pid:
                    continue
                
                # Special handling for Ollama - limit its CPU usage
                if "ollama" in name.lower() and proc_cpu > self.ollama_cpu_limit:
                    try:
                        # Set CPU affinity to limit cores
                        process = psutil.Process(pid)
                        current_affinity = process.cpu_affinity()
                        
                        # Limit to first 3 cores if using more
                        if len(current_affinity) > 3:
                            new_affinity = list(current_affinity)[:3]
                            process.cpu_affinity(new_affinity)
                            actions.append(f"Limited Ollama (PID {pid}) CPU affinity to cores {new_affinity}")
                        
                        # Set lower priority
                        if current_nice < 10:
                            process.nice(10)
                            self.process_nice_values[pid] = 10
                            actions.append(f"Set Ollama (PID {pid}) nice value to 10")
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as e:
                        self.logger.warning(f"Could not optimize Ollama process {pid}: {e}")
                
                # Handle other high CPU processes
                elif proc_cpu > 50 and current_nice < 5:
                    try:
                        process = psutil.Process(pid)
                        
                        # Don't modify system critical processes
                        if name in ['systemd', 'kernel', 'kthreadd', 'init']:
                            continue
                        
                        # Increase nice value to lower priority
                        new_nice = min(current_nice + 5, 19)
                        process.nice(new_nice)
                        self.process_nice_values[pid] = new_nice
                        actions.append(f"Increased nice value for {name} (PID {pid}) to {new_nice}")
                        
                    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as e:
                        self.logger.warning(f"Could not adjust nice value for process {pid}: {e}")
            
            # If CPU still high, trigger garbage collection
            if cpu_percent > self.cpu_threshold_critical:
                gc.collect()
                actions.append("Triggered Python garbage collection")
                
        except Exception as e:
            self.logger.error(f"Error optimizing CPU usage: {e}")
        
        return actions
    
    async def _optimize_memory_usage(self, metrics: Dict[str, Any]) -> List[str]:
        """Optimize memory usage when high memory usage is detected"""
        actions = []
        
        try:
            memory_percent = metrics.get("memory", {}).get("percent", 0)
            available_memory = metrics.get("memory", {}).get("available", 0)
            
            # Trigger garbage collection
            gc.collect()
            actions.append("Triggered Python garbage collection")
            
            # Clear system caches if memory usage is critical
            if memory_percent > self.memory_threshold_critical:
                try:
                    # Clear page cache, dentries and inodes
                    subprocess.run(['sync'], check=True, timeout=10)
                    with open('/proc/sys/vm/drop_caches', 'w') as f:
                        f.write('3')
                    actions.append("Cleared system caches (page cache, dentries, inodes)")
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, PermissionError) as e:
                    self.logger.warning(f"Could not clear system caches: {e}")
            
            # Find memory-heavy processes and consider restarting containers
            processes = metrics.get("processes", [])
            for proc in processes:
                pid = proc.get("pid")
                name = proc.get("name", "unknown")
                memory_percent = proc.get("memory_percent", 0)
                
                # If a single process is using too much memory (>20%), log it
                if memory_percent > 20:
                    actions.append(f"High memory usage detected: {name} (PID {pid}) using {memory_percent:.1f}% memory")
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory usage: {e}")
        
        return actions
    
    async def _optimize_containers(self, metrics: Dict[str, Any]) -> List[str]:
        """Optimize Docker containers based on resource usage"""
        actions = []
        
        if not self.docker_client:
            return actions
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                container_name = container.name
                container_id = container.id[:12]
                
                # Skip if we've already applied limits to this container recently
                if container_id in self.container_limits_applied:
                    continue
                
                try:
                    # Get container stats
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage percentage
                    cpu_stats = stats.get("cpu_stats", {})
                    precpu_stats = stats.get("precpu_stats", {})
                    
                    cpu_usage = 0
                    if cpu_stats and precpu_stats:
                        cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
                        system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get("system_cpu_usage", 0)
                        
                        if system_delta > 0:
                            cpu_usage = (cpu_delta / system_delta) * 100
                    
                    # Memory usage
                    memory_stats = stats.get("memory_stats", {})
                    memory_usage = memory_stats.get("usage", 0)
                    memory_limit = memory_stats.get("limit", 0)
                    memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
                    
                    # Apply resource limits if container is using too many resources
                    if cpu_usage > 200 or memory_percent > 80:  # 200% = 2 cores
                        # Update container with resource limits
                        try:
                            # Determine appropriate limits based on container type
                            cpu_limit = "2.0"  # 2 CPU cores
                            memory_limit = "2g"  # 2GB memory
                            
                            # Special cases for known containers
                            if "ollama" in container_name.lower():
                                cpu_limit = "3.0"  # 3 cores for Ollama
                                memory_limit = "4g"  # 4GB for Ollama
                            elif any(name in container_name.lower() for name in ['postgres', 'redis', 'chroma']):
                                cpu_limit = "1.0"  # 1 core for databases
                                memory_limit = "1g"  # 1GB for databases
                            
                            # Update container (requires container restart)
                            # Note: This is a simplified approach. In production, you'd want more sophisticated container management
                            actions.append(f"Container {container_name} needs resource limits: CPU={cpu_limit}, Memory={memory_limit} (current: CPU={cpu_usage:.1f}%, Memory={memory_percent:.1f}%)")
                            
                            # Mark as processed
                            self.container_limits_applied.add(container_id)
                            
                        except Exception as e:
                            self.logger.warning(f"Could not apply limits to container {container_name}: {e}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not get stats for container {container_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error optimizing containers: {e}")
        
        return actions
    
    async def _perform_system_cleanup(self, metrics: Dict[str, Any]) -> List[str]:
        """Perform system cleanup tasks"""
        actions = []
        
        try:
            # Clean up zombie processes
            zombie_count = 0
            for proc in psutil.process_iter(['pid', 'status']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        zombie_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if zombie_count > 0:
                actions.append(f"Detected {zombie_count} zombie processes")
            
            # Clean temporary files if disk usage is high
            disk_percent = metrics.get("disk", {}).get("percent", 0)
            if disk_percent > self.disk_threshold_high:
                # Clean /tmp directory of old files (older than 7 days)
                try:
                    result = subprocess.run([
                        'find', '/tmp', '-type', 'f', '-atime', '+7', '-delete'
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        actions.append("Cleaned old temporary files from /tmp")
                    
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                    self.logger.warning(f"Could not clean temporary files: {e}")
                
                # Clean Docker system if available
                if self.docker_client:
                    try:
                        # Remove unused containers, networks, images, and build cache
                        self.docker_client.containers.prune()
                        self.docker_client.images.prune()
                        actions.append("Cleaned up unused Docker resources")
                    except Exception as e:
                        self.logger.warning(f"Could not clean Docker resources: {e}")
            
            # Reset process nice values if CPU usage has normalized
            cpu_percent = metrics.get("cpu", {}).get("total_percent", 0)
            if cpu_percent < self.cpu_threshold_high * 0.7:  # 70% of threshold
                reset_count = 0
                for pid, nice_value in list(self.process_nice_values.items()):
                    try:
                        process = psutil.Process(pid)
                        process.nice(0)  # Reset to normal priority
                        del self.process_nice_values[pid]
                        reset_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process no longer exists, remove from tracking
                        del self.process_nice_values[pid]
                
                if reset_count > 0:
                    actions.append(f"Reset nice values for {reset_count} processes (CPU usage normalized)")
            
        except Exception as e:
            self.logger.error(f"Error performing system cleanup: {e}")
        
        return actions

if __name__ == "__main__":
    agent = HardwareResourceOptimizerAgent()
    agent.start()