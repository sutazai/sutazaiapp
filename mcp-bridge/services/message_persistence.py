#!/usr/bin/env python3
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
