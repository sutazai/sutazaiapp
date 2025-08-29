#!/usr/bin/env python3
"""
Fix script for MCP Bridge critical issues
Implements async health checks and corrects port mappings
"""

import asyncio
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Correct port mappings based on actual running containers
CORRECT_AGENT_PORTS = {
    "letta": 11401,        # Was 11400 in registry
    "autogpt": 11402,      # Correct
    "crewai": 11403,       # Was 11401 in registry
    "aider": 11404,        # Was 11403 in registry  
    "langchain": 11405,    # Correct
    "bigagi": 11407,       # Correct
    "agentzero": 11408,    # Correct
    "skyvern": 11409,      # Correct
    "shellgpt": 11413,     # Correct
    "autogen": 11415,      # Correct
    "browseruse": 11703,   # Correct
    "semgrep": 11801,      # Correct
}

def fix_mcp_bridge_ports():
    """Fix port mappings in MCP Bridge server file"""
    mcp_bridge_file = Path("/opt/sutazaiapp/mcp-bridge/services/mcp_bridge_server.py")
    
    if not mcp_bridge_file.exists():
        logger.error(f"MCP Bridge file not found: {mcp_bridge_file}")
        return False
    
    content = mcp_bridge_file.read_text()
    
    # Fix the AGENT_REGISTRY port mappings
    for agent, correct_port in CORRECT_AGENT_PORTS.items():
        old_pattern = f'"{agent}": {{\n        "name":'
        if old_pattern in content:
            # Find and replace the port in the registry
            import re
            pattern = rf'("{agent}".*?"port":\s*)(\d+)'
            replacement = rf'\g<1>{correct_port}'
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            logger.info(f"Fixed port for {agent}: {correct_port}")
    
    # Backup original file
    backup_file = mcp_bridge_file.with_suffix('.py.backup')
    mcp_bridge_file.rename(backup_file)
    logger.info(f"Created backup: {backup_file}")
    
    # Write fixed content
    mcp_bridge_file.write_text(content)
    logger.info(f"Updated MCP Bridge configuration")
    
    return True

def create_async_health_check():
    """Create improved async health check module"""
    
    health_check_code = '''#!/usr/bin/env python3
"""
Async health check module for MCP Bridge
Replaces synchronous health checks with concurrent async checks
"""

import asyncio
import httpx
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AsyncHealthMonitor:
    """Asynchronous health monitoring for agents and services"""
    
    def __init__(self, timeout: float = 2.0, cache_ttl: int = 10):
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.health_cache: Dict[str, Dict[str, Any]] = {}
        self.http_client = None
    
    async def __aenter__(self):
        self.http_client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            await self.http_client.aclose()
    
    async def check_agent_health(self, agent_id: str, port: int) -> Dict[str, Any]:
        """Check single agent health with caching"""
        
        # Check cache first
        cache_key = f"agent:{agent_id}"
        if cache_key in self.health_cache:
            cached = self.health_cache[cache_key]
            if cached["expires"] > datetime.now():
                return cached["status"]
        
        # Perform health check
        status = {
            "agent_id": agent_id,
            "port": port,
            "status": "unknown",
            "last_check": datetime.now().isoformat(),
            "response_time": None,
            "error": None
        }
        
        try:
            start_time = asyncio.get_event_loop().time()
            response = await self.http_client.get(
                f"http://localhost:{port}/health"
            )
            response_time = asyncio.get_event_loop().time() - start_time
            
            if response.status_code == 200:
                status["status"] = "online"
                status["response_time"] = round(response_time * 1000, 2)  # ms
                data = response.json()
                status["details"] = data
            else:
                status["status"] = "degraded"
                status["error"] = f"HTTP {response.status_code}"
                
        except asyncio.TimeoutError:
            status["status"] = "timeout"
            status["error"] = f"Timeout after {self.timeout}s"
        except httpx.ConnectError:
            status["status"] = "offline"
            status["error"] = "Connection refused"
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
        
        # Cache the result
        self.health_cache[cache_key] = {
            "status": status,
            "expires": datetime.now() + timedelta(seconds=self.cache_ttl)
        }
        
        return status
    
    async def check_all_agents(self, agents: Dict[str, Dict]) -> Dict[str, Any]:
        """Check all agents concurrently"""
        
        tasks = []
        for agent_id, agent_info in agents.items():
            port = agent_info.get("port")
            if port:
                task = self.check_agent_health(agent_id, port)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(agents),
            "online": 0,
            "offline": 0,
            "degraded": 0,
            "agents": {}
        }
        
        for i, (agent_id, agent_info) in enumerate(agents.items()):
            if i < len(results):
                result = results[i]
                if isinstance(result, Exception):
                    health_summary["agents"][agent_id] = {
                        "status": "error",
                        "error": str(result)
                    }
                    health_summary["offline"] += 1
                else:
                    health_summary["agents"][agent_id] = result
                    status = result.get("status", "unknown")
                    if status == "online":
                        health_summary["online"] += 1
                    elif status == "offline":
                        health_summary["offline"] += 1
                    else:
                        health_summary["degraded"] += 1
        
        return health_summary

# Integration function for MCP Bridge
async def integrate_async_health_monitor(agent_registry: Dict):
    """Integration point for MCP Bridge server"""
    
    async with AsyncHealthMonitor() as monitor:
        return await monitor.check_all_agents(agent_registry)
'''
    
    health_file = Path("/opt/sutazaiapp/mcp-bridge/services/async_health_monitor.py")
    health_file.write_text(health_check_code)
    logger.info(f"Created async health monitor: {health_file}")
    
    return health_file

def create_message_persistence():
    """Create message persistence module for offline agents"""
    
    persistence_code = '''#!/usr/bin/env python3
"""
Message persistence module for MCP Bridge
Handles queuing messages for offline agents
"""

import json
import asyncio
import redis.asyncio as aioredis
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MessagePersistence:
    """Persist and replay messages for offline agents"""
    
    def __init__(self, redis_url: str = "redis://localhost:10001"):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.queue_prefix = "mcp:agent:queue:"
        self.max_queue_size = 1000
        self.default_ttl = 3600  # 1 hour
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        await self.redis_client.ping()
        logger.info("Message persistence connected to Redis")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def queue_message(self, agent_id: str, message: Dict[str, Any]) -> bool:
        """Queue a message for an offline agent"""
        
        if not self.redis_client:
            await self.connect()
        
        try:
            queue_key = f"{self.queue_prefix}{agent_id}"
            
            # Check queue size
            queue_size = await self.redis_client.llen(queue_key)
            if queue_size >= self.max_queue_size:
                # Remove oldest message
                await self.redis_client.rpop(queue_key)
                logger.warning(f"Queue full for {agent_id}, dropping oldest message")
            
            # Add message with metadata
            message_data = {
                "message": message,
                "queued_at": datetime.now().isoformat(),
                "attempts": 0
            }
            
            # Push to queue
            await self.redis_client.lpush(queue_key, json.dumps(message_data))
            
            # Set TTL on queue
            await self.redis_client.expire(queue_key, self.default_ttl)
            
            logger.info(f"Queued message for {agent_id}, queue size: {queue_size + 1}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue message for {agent_id}: {e}")
            return False
    
    async def get_pending_messages(self, agent_id: str, limit: int = 100) -> List[Dict]:
        """Get pending messages for an agent"""
        
        if not self.redis_client:
            await self.connect()
        
        try:
            queue_key = f"{self.queue_prefix}{agent_id}"
            
            # Get messages (non-blocking)
            messages = []
            for _ in range(limit):
                message_json = await self.redis_client.rpop(queue_key)
                if not message_json:
                    break
                
                try:
                    message_data = json.loads(message_json)
                    messages.append(message_data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid message in queue for {agent_id}")
            
            if messages:
                logger.info(f"Retrieved {len(messages)} pending messages for {agent_id}")
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get pending messages for {agent_id}: {e}")
            return []
    
    async def requeue_failed_message(self, agent_id: str, message_data: Dict) -> bool:
        """Requeue a failed message with incremented attempt counter"""
        
        message_data["attempts"] = message_data.get("attempts", 0) + 1
        
        # Drop message after 3 attempts
        if message_data["attempts"] >= 3:
            logger.warning(f"Dropping message for {agent_id} after 3 attempts")
            return False
        
        return await self.queue_message(agent_id, message_data["message"])
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics for all queues"""
        
        if not self.redis_client:
            await self.connect()
        
        stats = {}
        try:
            # Find all queue keys
            pattern = f"{self.queue_prefix}*"
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, match=pattern, count=100
                )
                for key in keys:
                    agent_id = key.replace(self.queue_prefix, "")
                    queue_size = await self.redis_client.llen(key)
                    stats[agent_id] = queue_size
                
                if cursor == 0:
                    break
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}

# Integration with MCP Bridge
async def setup_message_persistence():
    """Setup message persistence for MCP Bridge"""
    persistence = MessagePersistence()
    await persistence.connect()
    return persistence
'''
    
    persistence_file = Path("/opt/sutazaiapp/mcp-bridge/services/message_persistence.py")
    persistence_file.write_text(persistence_code)
    logger.info(f"Created message persistence module: {persistence_file}")
    
    return persistence_file

def main():
    """Main execution"""
    logger.info("Starting MCP Bridge fixes...")
    
    # 1. Fix port mappings
    if fix_mcp_bridge_ports():
        logger.info("✓ Fixed port mappings")
    else:
        logger.error("✗ Failed to fix port mappings")
    
    # 2. Create async health monitor
    health_file = create_async_health_check()
    logger.info(f"✓ Created async health monitor: {health_file}")
    
    # 3. Create message persistence
    persistence_file = create_message_persistence()
    logger.info(f"✓ Created message persistence: {persistence_file}")
    
    logger.info("""
    Next steps:
    1. Restart MCP Bridge: docker-compose restart sutazai-mcp-bridge
    2. Monitor logs: docker logs -f sutazai-mcp-bridge
    3. Test agent connectivity: curl http://localhost:11100/status
    """)

if __name__ == "__main__":
    main()