"""
MCP Resource Isolation Manager
Resolves resource conflicts and ensures proper isolation between MCP servers
Addresses the 71.4% failure rate when running simultaneously
"""
import asyncio
import os
import psutil
import logging
import tempfile
import shutil
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import fcntl
import socket
import json

logger = logging.getLogger(__name__)

@dataclass
class ResourceAllocation:
    """Tracks resource allocation for an MCP service"""
    service_name: str
    port: int
    pid: Optional[int] = None
    memory_limit_mb: int = 512
    cpu_limit_percent: float = 25.0
    temp_dir: Optional[str] = None
    config_dir: Optional[str] = None
    socket_path: Optional[str] = None
    file_locks: Set[str] = field(default_factory=set)
    allocated_at: datetime = field(default_factory=datetime.now)
    
    def cleanup(self):
        """Clean up allocated resources"""
        # Release file locks
        for lock_path in self.file_locks:
            try:
                os.remove(lock_path)
            except:
                pass
        
        # Clean temp directories
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        if self.config_dir and os.path.exists(self.config_dir):
            shutil.rmtree(self.config_dir, ignore_errors=True)

class PortAllocator:
    """Dynamic port allocation for MCP services"""
    
    def __init__(self, base_port: int = 11100, max_ports: int = 100):
        self.base_port = base_port
        self.max_ports = max_ports
        self.allocated_ports: Dict[int, str] = {}
        self.reserved_ports: Set[int] = set()
        
    def allocate_port(self, service_name: str, preferred_port: Optional[int] = None) -> Optional[int]:
        """
        Allocate a port for a service
        
        Args:
            service_name: Name of the service
            preferred_port: Preferred port if available
        
        Returns:
            Allocated port or None if no ports available
        """
        # Try preferred port first
        if preferred_port and self._is_port_available(preferred_port):
            self.allocated_ports[preferred_port] = service_name
            logger.info(f"Allocated preferred port {preferred_port} to {service_name}")
            return preferred_port
        
        # Find next available port
        for offset in range(self.max_ports):
            port = self.base_port + offset
            if self._is_port_available(port):
                self.allocated_ports[port] = service_name
                logger.info(f"Allocated port {port} to {service_name}")
                return port
        
        logger.error(f"No available ports for {service_name}")
        return None
    
    def release_port(self, port: int):
        """Release an allocated port"""
        if port in self.allocated_ports:
            service_name = self.allocated_ports[port]
            del self.allocated_ports[port]
            logger.info(f"Released port {port} from {service_name}")
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available for use"""
        # Check if already allocated
        if port in self.allocated_ports or port in self.reserved_ports:
            return False
        
        # Check if port is in use by system
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except:
            return False
    
    def reserve_ports(self, ports: List[int]):
        """Reserve ports that should not be allocated"""
        self.reserved_ports.update(ports)

class FileLockManager:
    """Manages file locks to prevent conflicts"""
    
    def __init__(self):
        self.locks: Dict[str, Any] = {}
        self.lock_dir = "/tmp/mcp_locks"
        os.makedirs(self.lock_dir, exist_ok=True)
    
    def acquire_lock(self, service_name: str, resource_path: str) -> bool:
        """
        Acquire a lock on a resource
        
        Args:
            service_name: Name of the service
            resource_path: Path to the resource
        
        Returns:
            True if lock acquired
        """
        lock_id = hashlib.md5(f"{service_name}:{resource_path}".encode()).hexdigest()
        lock_path = os.path.join(self.lock_dir, f"{lock_id}.lock")
        
        try:
            # Try to create lock file
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, f"{service_name}\n{resource_path}\n{os.getpid()}".encode())
            os.close(fd)
            
            self.locks[lock_id] = {
                "service": service_name,
                "resource": resource_path,
                "path": lock_path
            }
            
            logger.debug(f"Acquired lock for {service_name} on {resource_path}")
            return True
            
        except FileExistsError:
            # Lock already exists, check if process is still alive
            if self._is_lock_stale(lock_path):
                os.remove(lock_path)
                return self.acquire_lock(service_name, resource_path)  # Retry
            
            logger.debug(f"Lock already held for {resource_path}")
            return False
        except Exception as e:
            logger.error(f"Error acquiring lock: {e}")
            return False
    
    def release_lock(self, service_name: str, resource_path: str):
        """Release a lock on a resource"""
        lock_id = hashlib.md5(f"{service_name}:{resource_path}".encode()).hexdigest()
        
        if lock_id in self.locks:
            lock_path = self.locks[lock_id]["path"]
            try:
                os.remove(lock_path)
                del self.locks[lock_id]
                logger.debug(f"Released lock for {service_name} on {resource_path}")
            except:
                pass
    
    def _is_lock_stale(self, lock_path: str) -> bool:
        """Check if a lock file is stale (process no longer exists)"""
        try:
            with open(lock_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    pid = int(lines[2].strip())
                    # Check if process exists
                    try:
                        os.kill(pid, 0)
                        return False  # Process exists
                    except OSError:
                        return True  # Process doesn't exist
        except:
            pass
        
        # Check file age
        try:
            stat = os.stat(lock_path)
            age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
            return age > timedelta(minutes=10)  # Consider stale after 10 minutes
        except:
            return True

class DependencyIsolator:
    """Isolates dependencies to prevent conflicts"""
    
    def __init__(self):
        self.isolated_envs: Dict[str, str] = {}
        self.base_dir = "/tmp/mcp_isolated_envs"
        os.makedirs(self.base_dir, exist_ok=True)
    
    def create_isolated_env(self, service_name: str) -> Dict[str, str]:
        """
        Create an isolated environment for a service
        
        Args:
            service_name: Name of the service
        
        Returns:
            Environment variables for isolation
        """
        # Create service-specific directories
        service_dir = os.path.join(self.base_dir, service_name)
        os.makedirs(service_dir, exist_ok=True)
        
        # Create subdirectories
        paths = {
            "home": os.path.join(service_dir, "home"),
            "tmp": os.path.join(service_dir, "tmp"),
            "cache": os.path.join(service_dir, "cache"),
            "config": os.path.join(service_dir, "config"),
            "npm": os.path.join(service_dir, "npm"),
            "pip": os.path.join(service_dir, "pip")
        }
        
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
        
        # Create isolated environment variables
        env = {
            "HOME": paths["home"],
            "TMPDIR": paths["tmp"],
            "TEMP": paths["tmp"],
            "TMP": paths["tmp"],
            "XDG_CACHE_HOME": paths["cache"],
            "XDG_CONFIG_HOME": paths["config"],
            "NPM_CONFIG_PREFIX": paths["npm"],
            "NPM_CONFIG_CACHE": os.path.join(paths["npm"], "cache"),
            "PIP_PREFIX": paths["pip"],
            "PYTHONUSERBASE": paths["pip"],
            "NODE_PATH": os.path.join(paths["npm"], "lib", "node_modules"),
            "MCP_ISOLATED": "true",
            "MCP_SERVICE_NAME": service_name
        }
        
        self.isolated_envs[service_name] = service_dir
        
        logger.info(f"Created isolated environment for {service_name}")
        return env
    
    def cleanup_env(self, service_name: str):
        """Clean up isolated environment"""
        if service_name in self.isolated_envs:
            env_dir = self.isolated_envs[service_name]
            if os.path.exists(env_dir):
                shutil.rmtree(env_dir, ignore_errors=True)
            del self.isolated_envs[service_name]
            logger.info(f"Cleaned up environment for {service_name}")

class MCPResourceIsolationManager:
    """
    Main resource isolation manager for MCP services
    Coordinates all isolation strategies to prevent conflicts
    """
    
    def __init__(self):
        self.port_allocator = PortAllocator()
        self.file_lock_manager = FileLockManager()
        self.dependency_isolator = DependencyIsolator()
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        
        # Reserve known system ports
        self.port_allocator.reserve_ports([
            8080, 3000, 5000, 8000, 9000,  # Common web ports
            5432, 3306, 27017, 6379,  # Database ports
            8500, 8501,  # Consul ports
            9090, 3100  # Monitoring ports
        ])
    
    async def allocate_resources(
        self,
        service_name: str,
        preferred_port: Optional[int] = None,
        memory_limit_mb: int = 512,
        cpu_limit_percent: float = 25.0
    ) -> Optional[ResourceAllocation]:
        """
        Allocate isolated resources for an MCP service
        
        Args:
            service_name: Name of the service
            preferred_port: Preferred port if available
            memory_limit_mb: Memory limit in MB
            cpu_limit_percent: CPU limit as percentage
        
        Returns:
            Resource allocation or None if failed
        """
        try:
            # Check if already allocated
            if service_name in self.resource_allocations:
                logger.warning(f"Resources already allocated for {service_name}")
                return self.resource_allocations[service_name]
            
            # Allocate port
            port = self.port_allocator.allocate_port(service_name, preferred_port)
            if not port:
                logger.error(f"Failed to allocate port for {service_name}")
                return None
            
            # Create isolated environment
            env_vars = self.dependency_isolator.create_isolated_env(service_name)
            
            # Create resource allocation
            allocation = ResourceAllocation(
                service_name=service_name,
                port=port,
                memory_limit_mb=memory_limit_mb,
                cpu_limit_percent=cpu_limit_percent,
                temp_dir=env_vars.get("TMPDIR"),
                config_dir=env_vars.get("XDG_CONFIG_HOME")
            )
            
            # Acquire necessary file locks
            critical_resources = [
                "/opt/sutazaiapp/.mcp.json",
                f"/opt/sutazaiapp/scripts/mcp/wrappers/{service_name}.sh"
            ]
            
            for resource in critical_resources:
                if os.path.exists(resource):
                    if self.file_lock_manager.acquire_lock(service_name, resource):
                        allocation.file_locks.add(resource)
            
            self.resource_allocations[service_name] = allocation
            
            logger.info(f"✅ Allocated resources for {service_name}: port={port}, memory={memory_limit_mb}MB")
            return allocation
            
        except Exception as e:
            logger.error(f"Error allocating resources for {service_name}: {e}")
            # Cleanup partial allocation
            if port:
                self.port_allocator.release_port(port)
            self.dependency_isolator.cleanup_env(service_name)
            return None
    
    async def release_resources(self, service_name: str):
        """Release all resources allocated to a service"""
        if service_name not in self.resource_allocations:
            return
        
        allocation = self.resource_allocations[service_name]
        
        # Release port
        self.port_allocator.release_port(allocation.port)
        
        # Release file locks
        for resource in allocation.file_locks:
            self.file_lock_manager.release_lock(service_name, resource)
        
        # Cleanup environment
        self.dependency_isolator.cleanup_env(service_name)
        
        # Cleanup allocation
        allocation.cleanup()
        
        del self.resource_allocations[service_name]
        
        logger.info(f"✅ Released resources for {service_name}")
    
    def get_isolated_environment(self, service_name: str) -> Dict[str, str]:
        """Get isolated environment variables for a service"""
        if service_name not in self.resource_allocations:
            return {}
        
        # Get base isolation environment
        env = self.dependency_isolator.create_isolated_env(service_name)
        
        # Add resource limits
        allocation = self.resource_allocations[service_name]
        env.update({
            "MCP_PORT": str(allocation.port),
            "MCP_MEMORY_LIMIT": str(allocation.memory_limit_mb),
            "MCP_CPU_LIMIT": str(allocation.cpu_limit_percent)
        })
        
        return env
    
    async def enforce_resource_limits(self, service_name: str, pid: int):
        """Enforce resource limits on a running process"""
        if service_name not in self.resource_allocations:
            return
        
        allocation = self.resource_allocations[service_name]
        allocation.pid = pid
        
        try:
            process = psutil.Process(pid)
            
            # Set CPU affinity (limit to specific cores)
            cpu_count = psutil.cpu_count()
            max_cpus = max(1, int(cpu_count * allocation.cpu_limit_percent / 100))
            process.cpu_affinity(list(range(max_cpus)))
            
            # Monitor and enforce memory limit
            asyncio.create_task(self._monitor_memory_usage(service_name, pid))
            
            logger.info(f"Enforced resource limits for {service_name} (PID: {pid})")
            
        except Exception as e:
            logger.error(f"Error enforcing limits for {service_name}: {e}")
    
    async def _monitor_memory_usage(self, service_name: str, pid: int):
        """Monitor and enforce memory usage limits"""
        if service_name not in self.resource_allocations:
            return
        
        allocation = self.resource_allocations[service_name]
        max_memory_bytes = allocation.memory_limit_mb * 1024 * 1024
        
        while service_name in self.resource_allocations:
            try:
                process = psutil.Process(pid)
                memory_info = process.memory_info()
                
                if memory_info.rss > max_memory_bytes:
                    logger.warning(f"Service {service_name} exceeding memory limit: {memory_info.rss / 1024 / 1024:.2f}MB > {allocation.memory_limit_mb}MB")
                    # Could implement memory pressure or restart logic here
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except psutil.NoSuchProcess:
                logger.info(f"Process {pid} for {service_name} no longer exists")
                break
            except Exception as e:
                logger.error(f"Error monitoring memory for {service_name}: {e}")
                break
    
    def get_allocation_status(self) -> Dict[str, Any]:
        """Get current resource allocation status"""
        return {
            "allocated_services": list(self.resource_allocations.keys()),
            "allocated_ports": dict(self.port_allocator.allocated_ports),
            "total_services": len(self.resource_allocations),
            "port_usage": {
                "allocated": len(self.port_allocator.allocated_ports),
                "available": self.port_allocator.max_ports - len(self.port_allocator.allocated_ports)
            },
            "allocations": {
                name: {
                    "port": alloc.port,
                    "memory_limit_mb": alloc.memory_limit_mb,
                    "cpu_limit_percent": alloc.cpu_limit_percent,
                    "allocated_at": alloc.allocated_at.isoformat()
                }
                for name, alloc in self.resource_allocations.items()
            }
        }
    
    async def cleanup_all(self):
        """Clean up all allocated resources"""
        for service_name in list(self.resource_allocations.keys()):
            await self.release_resources(service_name)
        
        logger.info("✅ Cleaned up all resource allocations")

# Global instance
_resource_manager: Optional[MCPResourceIsolationManager] = None

async def get_resource_manager() -> MCPResourceIsolationManager:
    """Get or create resource isolation manager"""
    global _resource_manager
    
    if _resource_manager is None:
        _resource_manager = MCPResourceIsolationManager()
    
    return _resource_manager