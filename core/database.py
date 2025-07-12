"""
Database Management System for SutazAI
Handles all database operations with SQLite as primary database and optional PostgreSQL support
"""

import sqlite3
import asyncio
import aiosqlite
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
import json
from datetime import datetime
import os
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Comprehensive Database Manager for SutazAI System"""
    
    def __init__(self, db_path: str = "/opt/sutazaiapp/data/sutazai.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection_pool = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize database with required tables"""
        if self.initialized:
            return
            
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await self._create_tables(db)
                await db.commit()
                
            self.initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _create_tables(self, db: aiosqlite.Connection):
        """Create all required database tables"""
        
        # Agents table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                status TEXT DEFAULT 'inactive',
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP,
                performance_metrics TEXT
            )
        """)
        
        # Tasks table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                agent_name TEXT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                result TEXT,
                error_message TEXT,
                metadata TEXT,
                FOREIGN KEY (agent_name) REFERENCES agents (name)
            )
        """)
        
        # Conversations table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                user_id TEXT,
                agent_name TEXT,
                message_type TEXT CHECK (message_type IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (agent_name) REFERENCES agents (name)
            )
        """)
        
        # Documents table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT,
                file_size INTEGER,
                content_hash TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_status TEXT DEFAULT 'pending',
                extracted_text TEXT,
                metadata TEXT,
                analysis_results TEXT
            )
        """)
        
        # Vector embeddings table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS vector_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_id TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding_vector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Models table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE NOT NULL,
                model_type TEXT,
                provider TEXT,
                config TEXT,
                status TEXT DEFAULT 'inactive',
                loaded_at TIMESTAMP,
                last_used TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                performance_metrics TEXT
            )
        """)
        
        # System logs table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_level TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # API usage tracking
        await db.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                user_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_time REAL,
                status_code INTEGER,
                request_size INTEGER,
                response_size INTEGER,
                metadata TEXT
            )
        """)
        
        # Configuration table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS configurations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                config_type TEXT DEFAULT 'string',
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        await self._create_indexes(db)
    
    async def _create_indexes(self, db: aiosqlite.Connection):
        """Create database indexes for better performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_agents_status ON agents (status)",
            "CREATE INDEX IF NOT EXISTS idx_agents_name ON agents (name)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks (agent_name)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_id ON conversations (conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_agent ON conversations (agent_name)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents (file_type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (processing_status)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_source ON vector_embeddings (source_type, source_id)",
            "CREATE INDEX IF NOT EXISTS idx_models_status ON models (status)",
            "CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs (log_level)",
            "CREATE INDEX IF NOT EXISTS idx_logs_component ON system_logs (component)",
            "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_api_endpoint ON api_usage (endpoint)",
            "CREATE INDEX IF NOT EXISTS idx_api_timestamp ON api_usage (timestamp)"
        ]
        
        for index_sql in indexes:
            await db.execute(index_sql)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection with proper context management"""
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            yield db
    
    # Agent Management
    async def create_agent(self, name: str, agent_type: str, config: Dict = None) -> bool:
        """Create a new agent record"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO agents (name, type, config, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (name, agent_type, json.dumps(config or {})))
                await db.commit()
                
            logger.info(f"Created agent: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create agent {name}: {e}")
            return False
    
    async def update_agent_status(self, name: str, status: str) -> bool:
        """Update agent status"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    UPDATE agents 
                    SET status = ?, last_activity = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE name = ?
                """, (status, name))
                await db.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent status {name}: {e}")
            return False
    
    async def get_agent(self, name: str) -> Optional[Dict]:
        """Get agent information"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM agents WHERE name = ?
                """, (name,))
                row = await cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get agent {name}: {e}")
            return None
    
    async def get_all_agents(self) -> List[Dict]:
        """Get all agents"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("SELECT * FROM agents ORDER BY name")
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get all agents: {e}")
            return []
    
    # Task Management
    async def create_task(self, task_id: str, agent_name: str, title: str, 
                         description: str = None, priority: int = 5, 
                         metadata: Dict = None) -> bool:
        """Create a new task"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO tasks (task_id, agent_name, title, description, priority, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (task_id, agent_name, title, description, priority, json.dumps(metadata or {})))
                await db.commit()
                
            logger.info(f"Created task: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create task {task_id}: {e}")
            return False
    
    async def update_task_status(self, task_id: str, status: str, 
                                result: str = None, error_message: str = None) -> bool:
        """Update task status"""
        try:
            async with self.get_connection() as db:
                completed_at = datetime.now().isoformat() if status == 'completed' else None
                
                await db.execute("""
                    UPDATE tasks 
                    SET status = ?, result = ?, error_message = ?, 
                        completed_at = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = ?
                """, (status, result, error_message, completed_at, task_id))
                await db.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            return False
    
    async def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task information"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM tasks WHERE task_id = ?
                """, (task_id,))
                row = await cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None
    
    async def get_tasks_by_agent(self, agent_name: str, status: str = None) -> List[Dict]:
        """Get tasks for a specific agent"""
        try:
            async with self.get_connection() as db:
                if status:
                    cursor = await db.execute("""
                        SELECT * FROM tasks WHERE agent_name = ? AND status = ?
                        ORDER BY created_at DESC
                    """, (agent_name, status))
                else:
                    cursor = await db.execute("""
                        SELECT * FROM tasks WHERE agent_name = ?
                        ORDER BY created_at DESC
                    """, (agent_name,))
                
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get tasks for agent {agent_name}: {e}")
            return []
    
    # Conversation Management
    async def save_conversation_message(self, conversation_id: str, user_id: str,
                                      agent_name: str, message_type: str,
                                      content: str, metadata: Dict = None) -> bool:
        """Save a conversation message"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO conversations 
                    (conversation_id, user_id, agent_name, message_type, content, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (conversation_id, user_id, agent_name, message_type, content, json.dumps(metadata or {})))
                await db.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conversation message: {e}")
            return False
    
    async def get_conversation_history(self, conversation_id: str, limit: int = 100) -> List[Dict]:
        """Get conversation history"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM conversations 
                    WHERE conversation_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (conversation_id, limit))
                
                rows = await cursor.fetchall()
                return [dict(row) for row in reversed(rows)]
                
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    # Document Management
    async def save_document(self, document_id: str, filename: str, file_type: str,
                           file_size: int, content_hash: str, extracted_text: str = None,
                           metadata: Dict = None, analysis_results: Dict = None) -> bool:
        """Save document information"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO documents 
                    (document_id, filename, file_type, file_size, content_hash, 
                     extracted_text, metadata, analysis_results, processing_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed')
                """, (document_id, filename, file_type, file_size, content_hash,
                      extracted_text, json.dumps(metadata or {}), json.dumps(analysis_results or {})))
                await db.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save document {document_id}: {e}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[Dict]:
        """Get document information"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM documents WHERE document_id = ?
                """, (document_id,))
                row = await cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    async def get_all_documents(self, file_type: str = None) -> List[Dict]:
        """Get all documents"""
        try:
            async with self.get_connection() as db:
                if file_type:
                    cursor = await db.execute("""
                        SELECT * FROM documents WHERE file_type = ?
                        ORDER BY processed_at DESC
                    """, (file_type,))
                else:
                    cursor = await db.execute("""
                        SELECT * FROM documents ORDER BY processed_at DESC
                    """)
                
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []
    
    # Vector Embeddings Management
    async def save_embedding(self, embedding_id: str, source_type: str, source_id: str,
                           content: str, embedding_vector: List[float] = None,
                           metadata: Dict = None) -> bool:
        """Save vector embedding"""
        try:
            async with self.get_connection() as db:
                vector_str = json.dumps(embedding_vector) if embedding_vector else None
                
                await db.execute("""
                    INSERT OR REPLACE INTO vector_embeddings 
                    (embedding_id, source_type, source_id, content, embedding_vector, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (embedding_id, source_type, source_id, content, vector_str, json.dumps(metadata or {})))
                await db.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save embedding {embedding_id}: {e}")
            return False
    
    async def get_embeddings_by_source(self, source_type: str, source_id: str) -> List[Dict]:
        """Get embeddings by source"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM vector_embeddings 
                    WHERE source_type = ? AND source_id = ?
                    ORDER BY created_at DESC
                """, (source_type, source_id))
                
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get embeddings for {source_type}:{source_id}: {e}")
            return []
    
    # Model Management
    async def register_model(self, model_name: str, model_type: str, provider: str,
                           config: Dict = None) -> bool:
        """Register a model"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO models 
                    (model_name, model_type, provider, config)
                    VALUES (?, ?, ?, ?)
                """, (model_name, model_type, provider, json.dumps(config or {})))
                await db.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            return False
    
    async def update_model_usage(self, model_name: str) -> bool:
        """Update model usage statistics"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    UPDATE models 
                    SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
                    WHERE model_name = ?
                """, (model_name,))
                await db.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model usage {model_name}: {e}")
            return False
    
    async def get_all_models(self) -> List[Dict]:
        """Get all registered models"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("SELECT * FROM models ORDER BY model_name")
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get all models: {e}")
            return []
    
    # System Logs
    async def log_system_event(self, level: str, component: str, message: str,
                              metadata: Dict = None) -> bool:
        """Log system event"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO system_logs (log_level, component, message, metadata)
                    VALUES (?, ?, ?, ?)
                """, (level, component, message, json.dumps(metadata or {})))
                await db.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
            return False
    
    async def get_system_logs(self, level: str = None, component: str = None, 
                             limit: int = 1000) -> List[Dict]:
        """Get system logs"""
        try:
            async with self.get_connection() as db:
                query = "SELECT * FROM system_logs"
                params = []
                conditions = []
                
                if level:
                    conditions.append("log_level = ?")
                    params.append(level)
                
                if component:
                    conditions.append("component = ?")
                    params.append(component)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get system logs: {e}")
            return []
    
    # API Usage Tracking
    async def log_api_usage(self, endpoint: str, method: str, user_id: str = None,
                           response_time: float = None, status_code: int = None,
                           request_size: int = None, response_size: int = None,
                           metadata: Dict = None) -> bool:
        """Log API usage"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO api_usage 
                    (endpoint, method, user_id, response_time, status_code, 
                     request_size, response_size, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (endpoint, method, user_id, response_time, status_code,
                      request_size, response_size, json.dumps(metadata or {})))
                await db.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to log API usage: {e}")
            return False
    
    async def get_api_usage_stats(self, hours: int = 24) -> Dict:
        """Get API usage statistics"""
        try:
            async with self.get_connection() as db:
                # Total requests
                cursor = await db.execute("""
                    SELECT COUNT(*) as total_requests
                    FROM api_usage 
                    WHERE timestamp > datetime('now', '-{} hours')
                """.format(hours))
                total_requests = (await cursor.fetchone())[0]
                
                # Requests by endpoint
                cursor = await db.execute("""
                    SELECT endpoint, COUNT(*) as count
                    FROM api_usage 
                    WHERE timestamp > datetime('now', '-{} hours')
                    GROUP BY endpoint
                    ORDER BY count DESC
                """.format(hours))
                endpoints = [dict(row) for row in await cursor.fetchall()]
                
                # Average response time
                cursor = await db.execute("""
                    SELECT AVG(response_time) as avg_response_time
                    FROM api_usage 
                    WHERE timestamp > datetime('now', '-{} hours')
                    AND response_time IS NOT NULL
                """.format(hours))
                avg_response_time = (await cursor.fetchone())[0] or 0
                
                return {
                    'total_requests': total_requests,
                    'endpoints': endpoints,
                    'avg_response_time': avg_response_time,
                    'period_hours': hours
                }
                
        except Exception as e:
            logger.error(f"Failed to get API usage stats: {e}")
            return {}
    
    # Configuration Management
    async def set_config(self, key: str, value: Any, config_type: str = 'string',
                        description: str = None) -> bool:
        """Set configuration value"""
        try:
            async with self.get_connection() as db:
                value_str = json.dumps(value) if config_type != 'string' else str(value)
                
                await db.execute("""
                    INSERT OR REPLACE INTO configurations 
                    (config_key, config_value, config_type, description, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (key, value_str, config_type, description))
                await db.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False
    
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT config_value, config_type FROM configurations
                    WHERE config_key = ?
                """, (key,))
                row = await cursor.fetchone()
                
                if row:
                    value_str, config_type = row
                    if config_type == 'string':
                        return value_str
                    else:
                        return json.loads(value_str)
                
                return default
                
        except Exception as e:
            logger.error(f"Failed to get config {key}: {e}")
            return default
    
    async def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration values"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("SELECT * FROM configurations")
                rows = await cursor.fetchall()
                
                configs = {}
                for row in rows:
                    row_dict = dict(row)
                    key = row_dict['config_key']
                    value_str = row_dict['config_value']
                    config_type = row_dict['config_type']
                    
                    if config_type == 'string':
                        configs[key] = value_str
                    else:
                        configs[key] = json.loads(value_str)
                
                return configs
                
        except Exception as e:
            logger.error(f"Failed to get all configs: {e}")
            return {}
    
    # Database Maintenance
    async def vacuum_database(self) -> bool:
        """Vacuum database to optimize storage"""
        try:
            async with self.get_connection() as db:
                await db.execute("VACUUM")
                
            logger.info("Database vacuumed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            return False
    
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            async with self.get_connection() as db:
                stats = {}
                
                # Table counts
                tables = ['agents', 'tasks', 'conversations', 'documents', 
                         'vector_embeddings', 'models', 'system_logs', 'api_usage']
                
                for table in tables:
                    cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                    count = (await cursor.fetchone())[0]
                    stats[f"{table}_count"] = count
                
                # Database size
                stats['database_size'] = os.path.getsize(self.db_path)
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

# Global database instance
db_manager = DatabaseManager()

async def init_database():
    """Initialize the global database instance"""
    await db_manager.initialize()

def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    return db_manager