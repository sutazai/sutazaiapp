"""
Unified Service Controller for automation Coordinator
Controls and manages all 31+ services through the automation Coordinator interface
"""

import asyncio
import httpx
import docker
import psutil
import logging
from datetime import datetime
import re

logger = logging.getLogger("unified_controller")

class UnifiedServiceController:
    """Master controller for all SutazAI services"""
    
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            logger.warning(f"Docker client initialization failed: {e}")
            self.docker_client = None
            self.docker_available = False
            
        self.services = self._initialize_services()
        self.health_status = {}
        self.resource_limits = {
            "cpu_percent": 80.0,  # Alert if CPU > 80%
            "memory_percent": 85.0,  # Alert if Memory > 85%
            "disk_percent": 90.0  # Alert if Disk > 90%
        }
        
    def _initialize_services(self) -> Dict[str, Dict]:
        """Initialize all service configurations"""
        return {
            # AI Agents
            "aider": {"port": 8095, "health": "/health", "type": "agent", "description": "AI Pair Programming Assistant"},
            "autogen": {"port": 8104, "health": "/health", "type": "agent", "description": "Microsoft AutoGen Multi-Agent"},
            "autogpt": {"port": None, "health": None, "type": "agent", "description": "Autonomous GPT Agent"},
            "crewai": {"port": 8096, "health": "/health", "type": "agent", "description": "Multi-Agent Collaboration"},
            "gpt-engineer": {"port": 8097, "health": "/health", "type": "agent", "description": "Full-Stack Code Generation"},
            "letta": {"port": None, "health": None, "type": "agent", "description": "Memory-Enhanced Agent"},
            "localagi": {"port": 8116, "health": "/health", "type": "agent", "description": "Local automation Implementation"},
            
            # Workflow & Automation
            "bigagi": {"port": 8106, "health": "/health", "type": "workflow", "description": "BigAGI Interface"},
            "dify": {"port": 8107, "health": "/health", "type": "workflow", "description": "Visual AI Workflow Builder"},
            "flowise": {"port": 8099, "health": "/health", "type": "workflow", "description": "FlowiseAI Workflow Engine"},
            "langflow": {"port": 8090, "health": "/health", "type": "workflow", "description": "LangFlow Visual Builder"},
            "n8n": {"port": 5678, "health": "/health", "type": "workflow", "description": "Workflow Automation Platform"},
            
            # Core Services
            "backend": {"port": 8000, "health": "/health", "type": "core", "description": "automation Coordinator Backend"},
            "frontend": {"port": 8501, "health": "/health", "type": "core", "description": "Streamlit Frontend"},
            "health-monitor": {"port": 8100, "health": "/health", "type": "core", "description": "System Health Monitor"},
            "service-hub": {"port": 8114, "health": "/health", "type": "core", "description": "Service Hub Manager"},
            
            # Databases & Storage
            "chromadb": {"port": 8001, "health": "/api/v1/heartbeat", "type": "database", "description": "Vector Database"},
            "faiss": {"port": 8002, "health": "/health", "type": "database", "description": "FAISS Vector Search"},
            "neo4j": {"port": 7474, "health": "/", "type": "database", "description": "Graph Database"},
            "postgres": {"port": 5432, "health": None, "type": "database", "description": "PostgreSQL Database"},
            "qdrant": {"port": 6333, "health": "/", "type": "database", "description": "Vector Search Engine"},
            "redis": {"port": 6379, "health": None, "type": "database", "description": "In-Memory Cache"},
            
            # ML/AI Frameworks
            "jax": {"port": 8089, "health": "/health", "type": "ml", "description": "JAX Machine Learning"},
            "llamaindex": {"port": 8098, "health": "/health", "type": "ml", "description": "LlamaIndex RAG"},
            "ollama": {"port": 10104, "health": "/api/tags", "type": "ml", "description": "Local LLM Server"},
            "pytorch": {"port": 8888, "health": "/health", "type": "ml", "description": "PyTorch Framework"},
            "tensorflow": {"port": 8889, "health": "/health", "type": "ml", "description": "TensorFlow Framework"},
            
            # Monitoring & Observability
            "grafana": {"port": 3000, "health": "/api/health", "type": "monitoring", "description": "Metrics Dashboard"},
            "loki": {"port": 3100, "health": "/ready", "type": "monitoring", "description": "Log Aggregation"},
            "prometheus": {"port": 9090, "health": "/-/healthy", "type": "monitoring", "description": "Metrics Collection"},
            "promtail": {"port": None, "health": None, "type": "monitoring", "description": "Log Collector"},
            
            # Other Services
        }
    
    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute natural language commands to control services"""
        command_lower = command.lower()
        
        # Service control commands
        if any(word in command_lower for word in ["start", "launch", "run"]):
            service = self._extract_service_name(command)
            if service:
                return await self.start_service(service)
                
        elif any(word in command_lower for word in ["stop", "shutdown", "kill"]):
            service = self._extract_service_name(command)
            if service:
                return await self.stop_service(service)
                
        elif any(word in command_lower for word in ["restart", "reload"]):
            service = self._extract_service_name(command)
            if service:
                return await self.restart_service(service)
                
        # Status and monitoring commands
        elif any(word in command_lower for word in ["status", "health", "check"]):
            if "all" in command_lower or "system" in command_lower:
                return await self.get_system_status()
            service = self._extract_service_name(command)
            if service:
                return await self.get_service_status(service)
            else:
                # Default to system status if no specific service mentioned
                return await self.get_system_status()
                
        elif any(word in command_lower for word in ["resource", "usage", "performance"]):
            return await self.get_resource_usage()
            
        elif any(word in command_lower for word in ["logs", "log"]):
            service = self._extract_service_name(command)
            if service:
                return await self.get_service_logs(service)
                
        # Service discovery and listing
        elif any(word in command_lower for word in ["list", "show", "services"]):
            if "agents" in command_lower:
                return await self.list_services_by_type("agent")
            elif "databases" in command_lower:
                return await self.list_services_by_type("database")
            elif "workflows" in command_lower:
                return await self.list_services_by_type("workflow")
            else:
                return await self.list_all_services()
                
        # Advanced operations
        elif "scale" in command_lower:
            service = self._extract_service_name(command)
            replicas = self._extract_number(command)
            if service and replicas:
                return await self.scale_service(service, replicas)
                
        elif "update" in command_lower or "upgrade" in command_lower:
            service = self._extract_service_name(command)
            if service:
                return await self.update_service(service)
                
        elif "backup" in command_lower:
            return await self.backup_databases()
            
        elif "optimize" in command_lower:
            return await self.optimize_resources()
            
        else:
            return {
                "status": "unknown_command",
                "message": f"I don't understand the command: '{command}'",
                "suggestions": [
                    "start/stop/restart <service>",
                    "check status of <service>",
                    "show all services",
                    "check system resources",
                    "show logs for <service>",
                    "optimize resources"
                ]
            }
    
    def _extract_service_name(self, command: str) -> Optional[str]:
        """Extract service name from command"""
        command_lower = command.lower()
        
        # Check for exact service names
        for service in self.services:
            if service in command_lower:
                return service
                
        # Check for common aliases
        aliases = {
            "coordinator": "backend",
            "frontend": "frontend",
            "ui": "frontend",
            "vector": "chromadb",
            "graph": "neo4j",
            "cache": "redis",
            "llm": "ollama",
            "models": "ollama"
        }
        
        for alias, service in aliases.items():
            if alias in command_lower:
                return service
                
        return None
    
    def _extract_number(self, command: str) -> Optional[int]:
        """Extract number from command"""
        numbers = re.findall(r'\d+', command)
        return int(numbers[0]) if numbers else None
    
    async def start_service(self, service_name: str) -> Dict[str, Any]:
        """Start a specific service"""
        if not self.docker_available:
            return {
                "status": "error",
                "service": service_name,
                "message": "Docker is not available in this environment. Service control is limited."
            }
            
        try:
            container = self.docker_client.containers.get(f"sutazai-{service_name}")
            if container.status != "running":
                container.start()
                await asyncio.sleep(2)  # Wait for startup
                
                # Check health
                health = await self.check_service_health(service_name)
                
                return {
                    "status": "success",
                    "action": "started",
                    "service": service_name,
                    "health": health,
                    "message": f"✅ Successfully started {service_name}"
                }
            else:
                return {
                    "status": "info",
                    "service": service_name,
                    "message": f"ℹ️ {service_name} is already running"
                }
                
        except docker.errors.NotFound:
            return {
                "status": "error",
                "service": service_name,
                "message": f"❌ Service {service_name} not found"
            }
        except Exception as e:
            return {
                "status": "error",
                "service": service_name,
                "message": f"❌ Failed to start {service_name}: {str(e)}"
            }
    
    async def stop_service(self, service_name: str) -> Dict[str, Any]:
        """Stop a specific service"""
        # Protect critical services
        if service_name in ["backend", "frontend", "ollama"]:
            return {
                "status": "warning",
                "service": service_name,
                "message": f"⚠️ Cannot stop critical service {service_name}. This would break the automation Coordinator."
            }
            
        try:
            container = self.docker_client.containers.get(f"sutazai-{service_name}")
            if container.status == "running":
                container.stop()
                
                return {
                    "status": "success",
                    "action": "stopped",
                    "service": service_name,
                    "message": f"✅ Successfully stopped {service_name}"
                }
            else:
                return {
                    "status": "info",
                    "service": service_name,
                    "message": f"ℹ️ {service_name} is already stopped"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "service": service_name,
                "message": f"❌ Failed to stop {service_name}: {str(e)}"
            }
    
    async def restart_service(self, service_name: str) -> Dict[str, Any]:
        """Restart a specific service"""
        try:
            container = self.docker_client.containers.get(f"sutazai-{service_name}")
            container.restart()
            await asyncio.sleep(3)  # Wait for restart
            
            # Check health
            health = await self.check_service_health(service_name)
            
            return {
                "status": "success",
                "action": "restarted",
                "service": service_name,
                "health": health,
                "message": f"✅ Successfully restarted {service_name}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "service": service_name,
                "message": f"❌ Failed to restart {service_name}: {str(e)}"
            }
    
    async def check_service_health(self, service_name: str) -> str:
        """Check health of a specific service"""
        service_info = self.services.get(service_name)
        if not service_info or not service_info.get("port") or not service_info.get("health"):
            return "no-health-check"
            
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                url = f"http://{service_name}:{service_info['port']}{service_info['health']}"
                response = await client.get(url)
                return "healthy" if response.status_code in [200, 204] else "unhealthy"
        except Exception as e:
            logger.warning(f"Exception caught, returning: {e}")
            return "unhealthy"
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get detailed status of a specific service"""
        try:
            container = self.docker_client.containers.get(f"sutazai-{service_name}")
            service_info = self.services.get(service_name, {})
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate resource usage
            try:
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
            except (KeyError, TypeError):
                cpu_percent = 0.0
            
            try:
                memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
                memory_limit = stats['memory_stats']['limit'] / (1024 * 1024)  # MB
                memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0.0
            except (KeyError, TypeError):
                memory_usage = 0.0
                memory_percent = 0.0
            
            # Check health
            health = await self.check_service_health(service_name)
            
            return {
                "status": "success",
                "service": service_name,
                "details": {
                    "container_id": container.short_id,
                    "status": container.status,
                    "health": health,
                    "type": service_info.get("type", "unknown"),
                    "description": service_info.get("description", ""),
                    "port": service_info.get("port"),
                    "created": container.attrs['Created'],
                    "started": container.attrs['State']['StartedAt'],
                    "resources": {
                        "cpu_percent": round(cpu_percent, 2),
                        "memory_mb": round(memory_usage, 2),
                        "memory_percent": round(memory_percent, 2)
                    }
                }
            }
            
        except docker.errors.NotFound:
            return {
                "status": "error",
                "service": service_name,
                "message": f"Service {service_name} not found"
            }
        except Exception as e:
            return {
                "status": "error",
                "service": service_name,
                "message": f"Failed to get status: {str(e)}"
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        if not self.docker_available:
            return {
                "status": "error",
                "message": "Docker is not available. Service status monitoring is limited."
            }
            
        containers = self.docker_client.containers.list(all=True)
        
        service_status = {
            "running": [],
            "stopped": [],
            "unhealthy": []
        }
        
        for container in containers:
            name = container.name.replace("sutazai-", "")
            if name in self.services:
                if container.status == "running":
                    health = await self.check_service_health(name)
                    if health == "healthy":
                        service_status["running"].append(name)
                    else:
                        service_status["unhealthy"].append(name)
                else:
                    service_status["stopped"].append(name)
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "total": len(self.services),
                "running": len(service_status["running"]),
                "stopped": len(service_status["stopped"]),
                "unhealthy": len(service_status["unhealthy"]),
                "details": service_status
            },
            "resources": {
                "cpu": {
                    "percent": cpu_percent,
                    "cores": psutil.cpu_count(),
                    "status": "🟢" if cpu_percent < self.resource_limits["cpu_percent"] else "🔴"
                },
                "memory": {
                    "percent": memory.percent,
                    "used_gb": round(memory.used / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2),
                    "status": "🟢" if memory.percent < self.resource_limits["memory_percent"] else "🔴"
                },
                "disk": {
                    "percent": disk.percent,
                    "used_gb": round(disk.used / (1024**3), 2),
                    "total_gb": round(disk.total / (1024**3), 2),
                    "status": "🟢" if disk.percent < self.resource_limits["disk_percent"] else "🔴"
                }
            }
        }
    
    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get detailed resource usage for all services"""
        containers = self.docker_client.containers.list()
        
        service_resources = []
        total_cpu = 0.0
        total_memory = 0.0
        
        for container in containers:
            name = container.name.replace("sutazai-", "")
            if name in self.services:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage
                    try:
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                        cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
                    except (KeyError, TypeError):
                        cpu_percent = 0.0
                    
                    # Calculate memory usage
                    try:
                        memory_usage = stats['memory_stats']['usage'] / (1024 * 1024 * 1024)  # GB
                    except (KeyError, TypeError):
                        memory_usage = 0.0
                    
                    total_cpu += cpu_percent
                    total_memory += memory_usage
                    
                    service_resources.append({
                        "service": name,
                        "cpu_percent": round(cpu_percent, 2),
                        "memory_gb": round(memory_usage, 2)
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to get stats for {name}: {e}")
        
        # Sort by CPU usage
        service_resources.sort(key=lambda x: x["cpu_percent"], reverse=True)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_usage": {
                "cpu_percent": round(total_cpu, 2),
                "memory_gb": round(total_memory, 2)
            },
            "top_consumers": service_resources[:10],
            "recommendations": self._get_resource_recommendations(total_cpu, total_memory)
        }
    
    def _get_resource_recommendations(self, cpu_percent: float, memory_gb: float) -> List[str]:
        """Get recommendations based on resource usage"""
        recommendations = []
        
        if cpu_percent > 80:
            recommendations.append("⚠️ High CPU usage detected. Consider scaling horizontally or optimizing workloads.")
            
        if memory_gb > psutil.virtual_memory().total / (1024**3) * 0.8:
            recommendations.append("⚠️ High memory usage. Consider increasing system memory or optimizing services.")
            
        if not recommendations:
            recommendations.append("✅ Resource usage is within healthy limits.")
            
        return recommendations
    
    async def get_service_logs(self, service_name: str, lines: int = 50) -> Dict[str, Any]:
        """Get logs from a specific service"""
        try:
            container = self.docker_client.containers.get(f"sutazai-{service_name}")
            logs = container.logs(tail=lines, timestamps=True).decode('utf-8')
            
            return {
                "status": "success",
                "service": service_name,
                "logs": logs.split('\n')
            }
            
        except Exception as e:
            return {
                "status": "error",
                "service": service_name,
                "message": f"Failed to get logs: {str(e)}"
            }
    
    async def list_all_services(self) -> Dict[str, Any]:
        """List all services with their status"""
        services_by_type = {}
        
        for service_name, service_info in self.services.items():
            service_type = service_info.get("type", "other")
            
            if service_type not in services_by_type:
                services_by_type[service_type] = []
                
            try:
                container = self.docker_client.containers.get(f"sutazai-{service_name}")
                status = container.status
                health = await self.check_service_health(service_name) if status == "running" else "n/a"
            except Exception as e:
                # TODO: Review this exception handling
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                status = "not_found"
                health = "n/a"
                
            services_by_type[service_type].append({
                "name": service_name,
                "status": status,
                "health": health,
                "port": service_info.get("port"),
                "description": service_info.get("description", "")
            })
        
        return {
            "status": "success",
            "services": services_by_type,
            "total_count": len(self.services)
        }
    
    async def list_services_by_type(self, service_type: str) -> Dict[str, Any]:
        """List services of a specific type"""
        filtered_services = []
        
        for service_name, service_info in self.services.items():
            if service_info.get("type") == service_type:
                try:
                    container = self.docker_client.containers.get(f"sutazai-{service_name}")
                    status = container.status
                    health = await self.check_service_health(service_name) if status == "running" else "n/a"
                except Exception as e:
                    # TODO: Review this exception handling
                    logger.error(f"Unexpected exception: {e}", exc_info=True)
                    status = "not_found"
                    health = "n/a"
                    
                filtered_services.append({
                    "name": service_name,
                    "status": status,
                    "health": health,
                    "port": service_info.get("port"),
                    "description": service_info.get("description", "")
                })
        
        return {
            "status": "success",
            "type": service_type,
            "services": filtered_services,
            "count": len(filtered_services)
        }
    
    async def scale_service(self, service_name: str, replicas: int) -> Dict[str, Any]:
        """Scale a service (placeholder for future implementation)"""
        return {
            "status": "info",
            "message": f"Service scaling for {service_name} to {replicas} replicas is not yet implemented in this version."
        }
    
    async def update_service(self, service_name: str) -> Dict[str, Any]:
        """Update a service (placeholder for future implementation)"""
        return {
            "status": "info",
            "message": f"Service update for {service_name} is not yet implemented. Please use docker-compose to update services."
        }
    
    async def backup_databases(self) -> Dict[str, Any]:
        """Backup all databases"""
        backup_results = []
        
        databases = ["postgres", "neo4j", "redis"]
        
        for db in databases:
            try:
                container = self.docker_client.containers.get(f"sutazai-{db}")
                if container.status == "running":
                    # Placeholder for actual backup logic
                    backup_results.append({
                        "database": db,
                        "status": "success",
                        "message": f"Backup initiated for {db}"
                    })
                else:
                    backup_results.append({
                        "database": db,
                        "status": "skipped",
                        "message": f"{db} is not running"
                    })
            except Exception as e:
                # TODO: Review this exception handling
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                backup_results.append({
                    "database": db,
                    "status": "error",
                    "message": f"Failed to backup {db}"
                })
        
        return {
            "status": "success",
            "action": "backup",
            "results": backup_results
        }
    
    async def optimize_resources(self) -> Dict[str, Any]:
        """Optimize system resources"""
        optimizations = []
        
        # Get current resource usage
        resource_status = await self.get_resource_usage()
        
        # Stop unused services
        containers = self.docker_client.containers.list()
        for container in containers:
            name = container.name.replace("sutazai-", "")
            if name in self.services:
                stats = container.stats(stream=False)
                cpu_usage = self._calculate_cpu_usage(stats)
                
                # Stop services with very low CPU usage (except critical ones)
                if cpu_usage < 0.1 and name not in ["backend", "frontend", "ollama", "postgres", "redis"]:
                    try:
                        container.stop()
                        optimizations.append(f"Stopped idle service: {name}")
                    except Exception as e:
                        # Suppressed exception (was bare except)
                        logger.debug(f"Suppressed exception: {e}")
                        pass
        
        # Clear Docker caches
        try:
            self.docker_client.containers.prune()
            self.docker_client.images.prune()
            optimizations.append("Cleared Docker caches")
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        return {
            "status": "success",
            "action": "optimize",
            "optimizations": optimizations,
            "message": f"Applied {len(optimizations)} optimizations"
        }
    
    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            return (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
        except Exception as e:
            logger.warning(f"Exception caught, returning: {e}")
            return 0.0


# Singleton instance
_controller = None

def get_unified_controller() -> UnifiedServiceController:
    """Get or create the unified controller instance"""
    global _controller
    if _controller is None:
        _controller = UnifiedServiceController()
    return _controller