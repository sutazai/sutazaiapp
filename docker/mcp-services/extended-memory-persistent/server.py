#!/usr/bin/env python3
"""
Extended Memory MCP Server with SQLite Persistence
Enhanced version with full data persistence across container restarts
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict, List, Optional
import json
import os
import sqlite3
from datetime import datetime
import logging
from contextlib import contextmanager
import threading
import pickle
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Extended Memory MCP Service", version="2.0.0")

# Configuration
SQLITE_PATH = os.environ.get("SQLITE_PATH", "/var/lib/mcp/extended_memory.db")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", 3009))
ENABLE_CACHE = os.environ.get("ENABLE_CACHE", "true").lower() == "true"

# Thread-safe database connection manager
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.local = threading.local()
        self.init_database()
        
    def init_database(self):
        """Initialize database schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create main memory store table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            # Create metadata table for statistics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_updated 
                ON memory_store(updated_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_accessed 
                ON memory_store(accessed_at DESC)
            """)
            
            # Initialize metadata
            cursor.execute("""
                INSERT OR IGNORE INTO metadata (key, value) 
                VALUES ('version', '2.0.0'), ('initialized', ?)
            """, (datetime.utcnow().isoformat(),))
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get thread-safe database connection"""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.local.conn.row_factory = sqlite3.Row
        try:
            yield self.local.conn
        except Exception as e:
            self.local.conn.rollback()
            raise e

# Initialize database manager
db_manager = DatabaseManager(SQLITE_PATH)

# In-memory cache for performance (optional)
memory_cache: Dict[str, Any] = {} if ENABLE_CACHE else None

def serialize_value(value: Any) -> tuple[str, str]:
    """Serialize any Python value for storage"""
    if value is None:
        return "null", "none"
    elif isinstance(value, (str, int, float, bool)):
        return json.dumps(value), type(value).__name__
    elif isinstance(value, (dict, list)):
        return json.dumps(value), type(value).__name__
    else:
        # For complex objects, use pickle + base64
        pickled = pickle.dumps(value)
        encoded = base64.b64encode(pickled).decode('utf-8')
        return encoded, "pickle"

def deserialize_value(value_str: str, type_str: str) -> Any:
    """Deserialize stored value back to Python object"""
    if type_str == "none":
        return None
    elif type_str in ["str", "int", "float", "bool", "dict", "list"]:
        return json.loads(value_str)
    elif type_str == "pickle":
        decoded = base64.b64decode(value_str.encode('utf-8'))
        return pickle.loads(decoded)
    else:
        # Fallback to JSON
        return json.loads(value_str)

@app.get("/health")
async def health():
    """Health check endpoint with detailed status"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM memory_store")
            item_count = cursor.fetchone()["count"]
            
            cursor.execute("SELECT value FROM metadata WHERE key = 'initialized'")
            initialized = cursor.fetchone()["value"]
            
        return JSONResponse({
            "status": "healthy",
            "service": "extended-memory",
            "version": "2.0.0",
            "port": SERVICE_PORT,
            "timestamp": datetime.utcnow().isoformat(),
            "persistence": {
                "enabled": True,
                "type": "SQLite",
                "path": SQLITE_PATH,
                "initialized": initialized
            },
            "statistics": {
                "memory_items": item_count,
                "cache_enabled": ENABLE_CACHE,
                "cache_items": len(memory_cache) if memory_cache is not None else 0
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/store")
async def store(data: dict):
    """Store data in extended memory with persistence"""
    key = data.get("key")
    value = data.get("value")
    
    if not key:
        raise HTTPException(status_code=400, detail="Key is required")
    
    try:
        # Serialize the value
        value_str, type_str = serialize_value(value)
        
        # Store in database
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memory_store (key, value, type, updated_at, accessed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    type = excluded.type,
                    updated_at = excluded.updated_at,
                    access_count = access_count + 1
            """, (key, value_str, type_str, datetime.utcnow(), datetime.utcnow()))
            conn.commit()
        
        # Update cache if enabled
        if memory_cache is not None:
            memory_cache[key] = value
        
        logger.info(f"Stored key: {key}")
        return {
            "status": "stored",
            "key": key,
            "persisted": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to store key {key}: {e}")
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")

@app.get("/retrieve/{key}")
async def retrieve(key: str):
    """Retrieve data from extended memory"""
    try:
        # Check cache first if enabled
        if memory_cache is not None and key in memory_cache:
            logger.info(f"Retrieved key from cache: {key}")
            return {
                "status": "found",
                "key": key,
                "value": memory_cache[key],
                "source": "cache"
            }
        
        # Query database
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update access time and count
            cursor.execute("""
                UPDATE memory_store 
                SET accessed_at = ?, access_count = access_count + 1
                WHERE key = ?
            """, (datetime.utcnow(), key))
            
            # Retrieve value
            cursor.execute("""
                SELECT value, type FROM memory_store WHERE key = ?
            """, (key,))
            
            row = cursor.fetchone()
            conn.commit()
            
            if row:
                value = deserialize_value(row["value"], row["type"])
                
                # Update cache if enabled
                if memory_cache is not None:
                    memory_cache[key] = value
                
                logger.info(f"Retrieved key from database: {key}")
                return {
                    "status": "found",
                    "key": key,
                    "value": value,
                    "source": "database"
                }
        
        logger.info(f"Key not found: {key}")
        return {"status": "not_found", "key": key}
        
    except Exception as e:
        logger.error(f"Failed to retrieve key {key}: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.get("/list")
async def list_keys(limit: int = 100, offset: int = 0):
    """List all keys in memory with pagination"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) as count FROM memory_store")
            total_count = cursor.fetchone()["count"]
            
            # Get keys with metadata
            cursor.execute("""
                SELECT key, type, created_at, updated_at, accessed_at, access_count
                FROM memory_store
                ORDER BY accessed_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            keys_metadata = []
            for row in cursor.fetchall():
                keys_metadata.append({
                    "key": row["key"],
                    "type": row["type"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "accessed_at": row["accessed_at"],
                    "access_count": row["access_count"]
                })
        
        return {
            "keys": [item["key"] for item in keys_metadata],
            "metadata": keys_metadata,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to list keys: {e}")
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")

@app.delete("/clear")
async def clear(confirm: bool = False):
    """Clear all memory (requires confirmation)"""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Pass confirm=true to clear all memory."
        )
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memory_store")
            conn.commit()
        
        # Clear cache if enabled
        if memory_cache is not None:
            memory_cache.clear()
        
        logger.warning("All memory cleared")
        return {
            "status": "cleared",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear memory: {e}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get detailed statistics about memory usage"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_items,
                    SUM(access_count) as total_accesses,
                    AVG(access_count) as avg_accesses,
                    MAX(access_count) as max_accesses
                FROM memory_store
            """)
            stats = dict(cursor.fetchone())
            
            # Get most accessed keys
            cursor.execute("""
                SELECT key, access_count
                FROM memory_store
                ORDER BY access_count DESC
                LIMIT 10
            """)
            most_accessed = [{"key": row["key"], "count": row["access_count"]} 
                           for row in cursor.fetchall()]
            
            # Get recently accessed keys
            cursor.execute("""
                SELECT key, accessed_at
                FROM memory_store
                ORDER BY accessed_at DESC
                LIMIT 10
            """)
            recently_accessed = [{"key": row["key"], "accessed_at": row["accessed_at"]} 
                                for row in cursor.fetchall()]
        
        return {
            "statistics": stats,
            "most_accessed": most_accessed,
            "recently_accessed": recently_accessed,
            "cache_size": len(memory_cache) if memory_cache is not None else 0,
            "database_path": SQLITE_PATH
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")

@app.post("/backup")
async def create_backup():
    """Create a backup of the memory database"""
    try:
        backup_path = f"{SQLITE_PATH}.backup.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        with db_manager.get_connection() as conn:
            backup_conn = sqlite3.connect(backup_path)
            conn.backup(backup_conn)
            backup_conn.close()
        
        logger.info(f"Backup created: {backup_path}")
        return {
            "status": "backup_created",
            "path": backup_path,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "Extended Memory MCP Service v2.0",
        "capabilities": [
            "store", "retrieve", "list", "clear", 
            "stats", "backup", "health"
        ],
        "features": {
            "persistence": "SQLite database",
            "caching": ENABLE_CACHE,
            "versioning": "2.0.0",
            "backup": "Available",
            "statistics": "Detailed tracking"
        },
        "api_docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Extended Memory MCP Service on port {SERVICE_PORT}")
    logger.info(f"Database path: {SQLITE_PATH}")
    logger.info(f"Cache enabled: {ENABLE_CACHE}")
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)