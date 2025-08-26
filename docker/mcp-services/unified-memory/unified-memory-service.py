#!/usr/bin/env python3
"""
Unified Memory Service - MCP Server
Consolidates extended-memory and memory-bank-mcp into a single, optimized service
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedMemoryConfig:
    """Configuration for unified memory service"""
    def __init__(self):
        self.sqlite_path = os.environ.get('SQLITE_PATH', '/var/lib/mcp/unified_memory.db')
        self.vector_store_enabled = os.environ.get('VECTOR_STORE_ENABLED', 'true').lower() == 'true'
        self.cache_enabled = os.environ.get('CACHE_ENABLED', 'true').lower() == 'true'
        self.host = os.environ.get('MCP_HOST', '0.0.0.0')
        self.port = int(os.environ.get('MCP_PORT', 3009))
        self.max_memory_size = int(os.environ.get('MAX_MEMORY_SIZE', 1048576))  # 1MB default

class MemoryRequest(BaseModel):
    """Request model for memory operations"""
    key: str
    content: Optional[str] = None
    namespace: str = "default"
    tags: List[str] = Field(default_factory=list)
    ttl: Optional[int] = None
    importance_level: int = Field(default=5, ge=1, le=10)
    enable_semantic_search: bool = False

class MemoryResponse(BaseModel):
    """Response model for memory operations"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    context_id: Optional[int] = None

class SQLiteMemoryStore:
    """SQLite-based memory storage (from extended-memory)"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create database and tables if they don't exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    namespace TEXT DEFAULT 'default',
                    tags TEXT,  -- JSON array
                    importance_level INTEGER DEFAULT 5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    size_bytes INTEGER,
                    UNIQUE(key, namespace)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at)
            """)
    
    async def store_memory(self, key: str, content: str, **kwargs) -> Dict[str, Any]:
        """Store memory in SQLite"""
        namespace = kwargs.get('namespace', 'default')
        tags = json.dumps(kwargs.get('tags', []))
        importance_level = kwargs.get('importance_level', 5)
        ttl = kwargs.get('ttl')
        
        expires_at = None
        if ttl:
            expires_at = datetime.now(timezone.utc).timestamp() + ttl
        
        size_bytes = len(content.encode('utf-8'))
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (key, content, namespace, tags, importance_level, size_bytes, expires_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (key, content, namespace, tags, importance_level, size_bytes, expires_at))
                
                context_id = cursor.lastrowid
                
                return {
                    'context_id': context_id,
                    'key': key,
                    'namespace': namespace,
                    'size_bytes': size_bytes,
                    'stored_at': datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    async def retrieve_memory(self, key: str, **kwargs) -> Dict[str, Any]:
        """Retrieve memory from SQLite"""
        namespace = kwargs.get('namespace', 'default')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM memories 
                    WHERE key = ? AND namespace = ?
                    AND (expires_at IS NULL OR expires_at > ?)
                """, (key, namespace, datetime.now(timezone.utc).timestamp()))
                
                row = cursor.fetchone()
                if not row:
                    raise KeyError(f"Memory not found: {key}")
                
                return {
                    'context_id': row['id'],
                    'key': row['key'],
                    'content': row['content'],
                    'namespace': row['namespace'],
                    'tags': json.loads(row['tags']),
                    'importance_level': row['importance_level'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'size_bytes': row['size_bytes']
                }
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            raise
    
    async def search_memory(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search memories in SQLite"""
        namespace = kwargs.get('namespace')
        limit = kwargs.get('limit', 10)
        
        conditions = ["(expires_at IS NULL OR expires_at > ?)"]
        params = [datetime.now(timezone.utc).timestamp()]
        
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        
        # Simple text search in content and tags
        conditions.append("(content LIKE ? OR tags LIKE ? OR key LIKE ?)")
        search_term = f"%{query}%"
        params.extend([search_term, search_term, search_term])
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(f"""
                    SELECT * FROM memories 
                    WHERE {' AND '.join(conditions)}
                    ORDER BY importance_level DESC, updated_at DESC
                    LIMIT ?
                """, params + [limit])
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'context_id': row['id'],
                        'key': row['key'],
                        'content': row['content'],
                        'namespace': row['namespace'],
                        'tags': json.loads(row['tags']),
                        'importance_level': row['importance_level'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'relevance_score': 1.0  # Simple scoring for now
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            raise
    
    async def delete_memory(self, key: str, **kwargs) -> bool:
        """Delete memory from SQLite"""
        namespace = kwargs.get('namespace', 'default')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM memories WHERE key = ? AND namespace = ?
                """, (key, namespace))
                
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get total counts
                cursor = conn.execute("SELECT COUNT(*) as total FROM memories")
                total_memories = cursor.fetchone()[0]
                
                # Get namespace counts
                cursor = conn.execute("""
                    SELECT namespace, COUNT(*) as count 
                    FROM memories 
                    GROUP BY namespace
                """)
                namespace_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Get size statistics
                cursor = conn.execute("""
                    SELECT SUM(size_bytes) as total_size, AVG(size_bytes) as avg_size
                    FROM memories
                """)
                size_row = cursor.fetchone()
                total_size = size_row[0] or 0
                avg_size = size_row[1] or 0
                
                # Get database file size
                db_file_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
                
                return {
                    'total_memories': total_memories,
                    'namespaces': namespace_stats,
                    'total_content_size_bytes': total_size,
                    'average_memory_size_bytes': avg_size,
                    'db_file_size_bytes': db_file_size,
                    'db_file_size_mb': round(db_file_size / 1024 / 1024, 2)
                }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    async def cleanup_expired(self) -> int:
        """Clean up expired memories"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM memories 
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                """, (datetime.now(timezone.utc).timestamp(),))
                
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Error cleaning up expired memories: {e}")
            return 0

class UnifiedMemoryService:
    """Unified memory service combining extended-memory and memory-bank features"""
    
    def __init__(self, config: UnifiedMemoryConfig):
        self.config = config
        self.sqlite_store = SQLiteMemoryStore(config.sqlite_path)
        self.app = FastAPI(title="Unified Memory Service", version="1.0.0")
        self._setup_routes()
        
        # Background cleanup will be started in run method
        self._cleanup_task = None
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/memory/store", response_model=MemoryResponse)
        async def store_memory(request: MemoryRequest):
            """Store memory with unified features"""
            try:
                result = await self.sqlite_store.store_memory(
                    key=request.key,
                    content=request.content,
                    namespace=request.namespace,
                    tags=request.tags,
                    ttl=request.ttl,
                    importance_level=request.importance_level
                )
                
                return MemoryResponse(success=True, data=result, context_id=result['context_id'])
            except Exception as e:
                logger.error(f"Store memory error: {e}")
                return MemoryResponse(success=False, error=str(e))
        
        @self.app.get("/memory/retrieve/{key}", response_model=MemoryResponse)
        async def retrieve_memory(key: str, namespace: str = "default"):
            """Retrieve memory by key"""
            try:
                result = await self.sqlite_store.retrieve_memory(key=key, namespace=namespace)
                return MemoryResponse(success=True, data=result)
            except KeyError as e:
                return MemoryResponse(success=False, error=f"Memory not found: {key}")
            except Exception as e:
                logger.error(f"Retrieve memory error: {e}")
                return MemoryResponse(success=False, error=str(e))
        
        @self.app.get("/memory/search", response_model=MemoryResponse)
        async def search_memory(query: str, namespace: Optional[str] = None, limit: int = 10):
            """Search memories"""
            try:
                results = await self.sqlite_store.search_memory(
                    query=query,
                    namespace=namespace,
                    limit=limit
                )
                return MemoryResponse(success=True, data={"results": results, "count": len(results)})
            except Exception as e:
                logger.error(f"Search memory error: {e}")
                return MemoryResponse(success=False, error=str(e))
        
        @self.app.delete("/memory/delete/{key}", response_model=MemoryResponse)
        async def delete_memory(key: str, namespace: str = "default"):
            """Delete memory by key"""
            try:
                deleted = await self.sqlite_store.delete_memory(key=key, namespace=namespace)
                if deleted:
                    return MemoryResponse(success=True, data={"deleted": True})
                else:
                    return MemoryResponse(success=False, error=f"Memory not found: {key}")
            except Exception as e:
                logger.error(f"Delete memory error: {e}")
                return MemoryResponse(success=False, error=str(e))
        
        @self.app.get("/memory/stats", response_model=MemoryResponse)
        async def get_memory_stats():
            """Get memory service statistics"""
            try:
                stats = await self.sqlite_store.get_memory_stats()
                return MemoryResponse(success=True, data=stats)
            except Exception as e:
                logger.error(f"Get stats error: {e}")
                return MemoryResponse(success=False, error=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Test database connectivity
                stats = await self.sqlite_store.get_memory_stats()
                return {
                    "status": "healthy",
                    "service": "unified-memory",
                    "version": "1.0.0",
                    "memories": stats.get('total_memories', 0),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
    
    async def _background_cleanup(self):
        """Background task to clean up expired memories"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                cleaned = await self.sqlite_store.cleanup_expired()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired memories")
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    def run(self):
        """Run the unified memory service"""
        logger.info(f"Starting Unified Memory Service on {self.config.host}:{self.config.port}")
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )

def main():
    """Main entry point"""
    config = UnifiedMemoryConfig()
    service = UnifiedMemoryService(config)
    service.run()

if __name__ == "__main__":
    main()