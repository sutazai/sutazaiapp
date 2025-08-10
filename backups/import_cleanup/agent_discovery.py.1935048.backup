"""
SutazAI Agent Discovery System
Automatic discovery, registration, and health monitoring of AI agents
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import aiohttp
import docker
import redis.asyncio as redis
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class AgentInfo:
    id: str
    name: str
    type: str
    endpoint: str
    capabilities: List[str]
    health_endpoint: str
    status: str = "unknown"
    last_seen: Optional[datetime] = None
    discovery_method: str = "unknown"
    metadata: Dict[str, Any] = None

class AgentDiscoveryService:
    """
    Comprehensive agent discovery system with multiple discovery methods:
    - Docker container discovery
    - Network scanning
    - Service registry integration
    - Manual registration
    - Health monitoring and auto-recovery
    """
    
    def __init__(self, redis_url: str = "redis://:redis_password@localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Agent tracking
        self.discovered_agents: Dict[str, AgentInfo] = {}
        self.health_cache: Dict[str, Dict] = {}
        
        # Discovery configuration
        self.discovery_methods = ["docker", "network_scan", "service_registry"]
        self.discovery_interval = 60  # seconds
        self.health_check_interval = 30  # seconds
        self.health_timeout = 5  # seconds
        
        # Agent patterns and configurations
        self.agent_patterns = {
            "sutazai-": {
                "port": 8080,
                "health_path": "/health",
                "capabilities_path": "/capabilities"
            },
            "agent-": {
                "port": 8080,
                "health_path": "/status",
                "capabilities_path": "/info"
            }
        }
        
        # Network configuration
        self.network_ranges = ["172.20.0.0/16", "172.21.0.0/16"]
        self.common_ports = [8080, 8081, 8082, 8090, 8091, 8092, 3000, 5000]
        
        # Metrics
        self.metrics = {
            "agents_discovered": 0,
            "agents_healthy": 0,
            "agents_unhealthy": 0,
            "discovery_runs": 0,
            "last_discovery": None
        }
    
    async def initialize(self):
        """Initialize the agent discovery service"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Load existing agent data
            await self._load_existing_agents()
            
            # Start background tasks
            asyncio.create_task(self._discovery_loop())
            asyncio.create_task(self._health_monitor_loop())
            asyncio.create_task(self._metrics_collector())
            
            logger.info("Agent discovery service initialized")
            
        except Exception as e:
            logger.error(f"Agent discovery initialization failed: {e}")
            raise
    
    async def _load_existing_agents(self):
        """Load existing agent data from Redis"""
        try:
            agent_data = await self.redis_client.hgetall("discovered_agents")
            for agent_id, data in agent_data.items():
                agent_info = AgentInfo(**json.loads(data))
                self.discovered_agents[agent_id] = agent_info
            
            logger.info(f"Loaded {len(self.discovered_agents)} existing agents")
            
        except Exception as e:
            logger.warning(f"Failed to load existing agents: {e}")
    
    async def _discovery_loop(self):
        """Main discovery loop"""
        while True:
            try:
                await self._run_discovery()
                await asyncio.sleep(self.discovery_interval)
                
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(self.discovery_interval)
    
    async def _run_discovery(self):
        """Run agent discovery using all configured methods"""
        start_time = time.time()
        discovered_count = 0
        
        try:
            # Docker container discovery
            if "docker" in self.discovery_methods:
                docker_agents = await self._discover_docker_agents()
                discovered_count += len(docker_agents)
                await self._process_discovered_agents(docker_agents, "docker")
            
            # Network scanning
            if "network_scan" in self.discovery_methods:
                network_agents = await self._discover_network_agents()
                discovered_count += len(network_agents)
                await self._process_discovered_agents(network_agents, "network_scan")
            
            # Service registry discovery
            if "service_registry" in self.discovery_methods:
                registry_agents = await self._discover_registry_agents()
                discovered_count += len(registry_agents)
                await self._process_discovered_agents(registry_agents, "service_registry")
            
            # Update metrics
            self.metrics["discovery_runs"] += 1
            self.metrics["last_discovery"] = datetime.now().isoformat()
            
            discovery_time = time.time() - start_time
            logger.info(f"Discovery completed in {discovery_time:.2f}s, found {discovered_count} agents")
            
        except Exception as e:
            logger.error(f"Discovery run failed: {e}")
    
    async def _discover_docker_agents(self) -> List[AgentInfo]:
        """Discover agents running in Docker containers"""
        agents = []
        
        try:
            # Use Docker API to find containers
            client = docker.from_env()
            containers = client.containers.list(filters={"status": "running"})
            
            for container in containers:
                container_name = container.name
                
                # Check if container matches agent patterns
                for pattern, config in self.agent_patterns.items():
                    if pattern in container_name:
                        try:
                            # Get container network info
                            networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
                            ip_address = None
                            
                            for network_name, network_info in networks.items():
                                if network_info.get("IPAddress"):
                                    ip_address = network_info["IPAddress"]
                                    break
                            
                            if not ip_address:
                                continue
                            
                            # Create agent info
                            agent_id = container_name
                            endpoint = f"http://{ip_address}:{config['port']}"
                            health_endpoint = f"{endpoint}{config['health_path']}"
                            
                            agent_info = AgentInfo(
                                id=agent_id,
                                name=container_name.replace("sutazai-", "").replace("-", " ").title(),
                                type=self._infer_agent_type(container_name),
                                endpoint=endpoint,
                                capabilities=await self._discover_capabilities(endpoint, config),
                                health_endpoint=health_endpoint,
                                discovery_method="docker",
                                metadata={
                                    "container_id": container.short_id,
                                    "image": container.image.tags[0] if container.image.tags else "unknown",
                                    "created": container.attrs.get("Created"),
                                    "ip_address": ip_address,
                                    "port": config["port"]
                                }
                            )
                            
                            agents.append(agent_info)
                            break
                            
                        except Exception as e:
                            logger.warning(f"Failed to process container {container_name}: {e}")
            
        except Exception as e:
            logger.error(f"Docker discovery failed: {e}")
        
        return agents
    
    async def _discover_network_agents(self) -> List[AgentInfo]:
        """Discover agents through network scanning"""
        agents = []
        
        try:
            # Scan common agent ports on network ranges
            scan_tasks = []
            
            for network_range in self.network_ranges:
                for port in self.common_ports:
                    scan_tasks.append(self._scan_network_range(network_range, port))
            
            # Execute scans concurrently
            scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
            
            for result in scan_results:
                if isinstance(result, list):
                    agents.extend(result)
            
        except Exception as e:
            logger.error(f"Network discovery failed: {e}")
        
        return agents
    
    async def _scan_network_range(self, network_range: str, port: int) -> List[AgentInfo]:
        """Scan a network range for agents on a specific port"""
        agents = []
        
        try:
            import ipaddress
            network = ipaddress.ip_network(network_range, strict=False)
            
            # Limit scan to first 50 IPs to avoid overwhelming the network
            scan_tasks = []
            for i, ip in enumerate(network.hosts()):
                if i >= 50:
                    break
                scan_tasks.append(self._check_agent_endpoint(str(ip), port))
            
            # Execute scans with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(10)
            
            async def bounded_scan(ip, port):
                async with semaphore:
                    return await self._check_agent_endpoint(ip, port)
            
            scan_results = await asyncio.gather(*[bounded_scan(str(ip), port) for i, ip in enumerate(network.hosts()) if i < 50], return_exceptions=True)
            
            for result in scan_results:
                if isinstance(result, AgentInfo):
                    agents.append(result)
        
        except Exception as e:
            logger.debug(f"Network range scan failed for {network_range}:{port}: {e}")
        
        return agents
    
    async def _check_agent_endpoint(self, ip: str, port: int) -> Optional[AgentInfo]:
        """Check if an endpoint hosts an agent"""
        try:
            endpoint = f"http://{ip}:{port}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                # Try common health endpoints
                health_paths = ["/health", "/status", "/ping", "/api/health"]
                
                for health_path in health_paths:
                    try:
                        async with session.get(f"{endpoint}{health_path}") as response:
                            if response.status == 200:
                                data = await response.text()
                                
                                # Check if response looks like an agent
                                if any(keyword in data.lower() for keyword in ["agent", "sutazai", "ai", "model", "assistant"]):
                                    agent_id = f"network_agent_{ip}_{port}"
                                    
                                    return AgentInfo(
                                        id=agent_id,
                                        name=f"Network Agent {ip}:{port}",
                                        type="network_discovered",
                                        endpoint=endpoint,
                                        capabilities=await self._discover_capabilities(endpoint, {"capabilities_path": "/capabilities"}),
                                        health_endpoint=f"{endpoint}{health_path}",
                                        discovery_method="network_scan",
                                        metadata={
                                            "ip_address": ip,
                                            "port": port,
                                            "health_path": health_path
                                        }
                                    )
                    except:
                        continue
        
        except Exception:
            pass
        
        return None
    
    async def _discover_registry_agents(self) -> List[AgentInfo]:
        """Discover agents from service registry"""
        agents = []
        
        try:
            # Check if we have a service registry endpoint
            registry_agents = await self.redis_client.hgetall("service_registry")
            
            for agent_id, agent_data in registry_agents.items():
                try:
                    data = json.loads(agent_data)
                    
                    agent_info = AgentInfo(
                        id=agent_id,
                        name=data.get("name", agent_id),
                        type=data.get("type", "registry"),
                        endpoint=data.get("endpoint"),
                        capabilities=data.get("capabilities", []),
                        health_endpoint=data.get("health_endpoint"),
                        discovery_method="service_registry",
                        metadata=data.get("metadata", {})
                    )
                    
                    agents.append(agent_info)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse registry agent {agent_id}: {e}")
        
        except Exception as e:
            logger.debug(f"Service registry discovery failed: {e}")
        
        return agents
    
    async def _discover_capabilities(self, endpoint: str, config: Dict) -> List[str]:
        """Discover agent capabilities"""
        capabilities = []
        
        try:
            capabilities_path = config.get("capabilities_path", "/capabilities")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                async with session.get(f"{endpoint}{capabilities_path}") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if isinstance(data, dict):
                            capabilities = data.get("capabilities", [])
                        elif isinstance(data, list):
                            capabilities = data
        
        except Exception:
            # Fallback: infer capabilities from agent type
            capabilities = self._infer_capabilities_from_endpoint(endpoint)
        
        return capabilities
    
    def _infer_agent_type(self, container_name: str) -> str:
        """Infer agent type from container name"""
        name_lower = container_name.lower()
        
        if "ai-engineer" in name_lower:
            return "ai_engineer"
        elif "testing" in name_lower or "qa" in name_lower:
            return "testing_qa"
        elif "devops" in name_lower or "infrastructure" in name_lower:
            return "devops"
        elif "security" in name_lower:
            return "security"
        elif "automation" in name_lower:
            return "automation"
        elif "orchestrator" in name_lower:
            return "orchestrator"
        else:
            return "general"
    
    def _infer_capabilities_from_endpoint(self, endpoint: str) -> List[str]:
        """Infer capabilities from endpoint analysis"""
        # This could be enhanced to actually probe the endpoint
        return ["general_task", "text_processing"]
    
    async def _process_discovered_agents(self, agents: List[AgentInfo], discovery_method: str):
        """Process newly discovered agents"""
        for agent in agents:
            # Check if agent is already known
            existing = self.discovered_agents.get(agent.id)
            
            if existing:
                # Update existing agent info
                existing.last_seen = datetime.now()
                existing.endpoint = agent.endpoint
                existing.capabilities = agent.capabilities
                existing.metadata.update(agent.metadata or {})
            else:
                # Add new agent
                agent.last_seen = datetime.now()
                self.discovered_agents[agent.id] = agent
                self.metrics["agents_discovered"] += 1
                
                logger.info(f"New agent discovered: {agent.id} via {discovery_method}")
            
            # Store in Redis
            await self.redis_client.hset(
                "discovered_agents",
                agent.id,
                json.dumps(asdict(agent), default=str)
            )
    
    async def _health_monitor_loop(self):
        """Monitor health of discovered agents"""
        while True:
            try:
                await self._check_agent_health()
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor loop error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_agent_health(self):
        """Check health of all discovered agents"""
        if not self.discovered_agents:
            return
        
        # Create health check tasks
        health_tasks = []
        for agent in self.discovered_agents.values():
            health_tasks.append(self._check_single_agent_health(agent))
        
        # Execute health checks concurrently
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        # Process results
        healthy_count = 0
        unhealthy_count = 0
        
        for i, result in enumerate(results):
            agent = list(self.discovered_agents.values())[i]
            
            if isinstance(result, dict) and result.get("healthy", False):
                agent.status = "healthy"
                healthy_count += 1
            else:
                agent.status = "unhealthy"
                unhealthy_count += 1
        
        # Update metrics
        self.metrics["agents_healthy"] = healthy_count
        self.metrics["agents_unhealthy"] = unhealthy_count
    
    async def _check_single_agent_health(self, agent: AgentInfo) -> Dict[str, Any]:
        """Check health of a single agent"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.health_timeout)) as session:
                async with session.get(agent.health_endpoint) as response:
                    if response.status == 200:
                        data = await response.text()
                        return {
                            "healthy": True,
                            "response_time": time.time(),
                            "status_code": response.status,
                            "data": data
                        }
                    else:
                        return {
                            "healthy": False,
                            "status_code": response.status,
                            "error": f"HTTP {response.status}"
                        }
        
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _metrics_collector(self):
        """Collect and store discovery metrics"""
        while True:
            try:
                # Store metrics in Redis
                await self.redis_client.hset(
                    "discovery_metrics",
                    "current",
                    json.dumps(self.metrics, default=str)
                )
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Discovery metrics collector error: {e}")
                await asyncio.sleep(30)
    
    # Public API methods
    
    async def get_discovered_agents(self) -> List[AgentInfo]:
        """Get all discovered agents"""
        return list(self.discovered_agents.values())
    
    async def get_healthy_agents(self) -> List[AgentInfo]:
        """Get only healthy agents"""
        return [agent for agent in self.discovered_agents.values() if agent.status == "healthy"]
    
    async def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Get agents with a specific capability"""
        return [
            agent for agent in self.discovered_agents.values()
            if capability in agent.capabilities and agent.status == "healthy"
        ]
    
    async def get_agents_by_type(self, agent_type: str) -> List[AgentInfo]:
        """Get agents of a specific type"""
        return [
            agent for agent in self.discovered_agents.values()
            if agent.type == agent_type and agent.status == "healthy"
        ]
    
    async def manually_register_agent(self, agent_info: AgentInfo) -> bool:
        """Manually register an agent"""
        try:
            agent_info.discovery_method = "manual"
            agent_info.last_seen = datetime.now()
            
            self.discovered_agents[agent_info.id] = agent_info
            
            # Store in Redis
            await self.redis_client.hset(
                "discovered_agents",
                agent_info.id,
                json.dumps(asdict(agent_info), default=str)
            )
            
            logger.info(f"Agent manually registered: {agent_info.id}")
            return True
            
        except Exception as e:
            logger.error(f"Manual agent registration failed: {e}")
            return False
    
    async def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from discovery"""
        try:
            if agent_id in self.discovered_agents:
                del self.discovered_agents[agent_id]
                await self.redis_client.hdel("discovered_agents", agent_id)
                logger.info(f"Agent removed: {agent_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Agent removal failed: {e}")
            return False
    
    async def trigger_discovery(self):
        """Manually trigger agent discovery"""
        await self._run_discovery()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get discovery service metrics"""
        return {
            **self.metrics,
            "total_agents": len(self.discovered_agents),
            "discovery_methods": self.discovery_methods,
            "last_health_check": datetime.now().isoformat()
        }
    
    async def stop(self):
        """Stop the discovery service"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Agent discovery service stopped")

# Singleton instance
agent_discovery = AgentDiscoveryService()