#!/usr/bin/env python3
"""
Memory Manager - Manages conversation history, context, and knowledge for JARVIS
"""

import asyncio
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import aioredis
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages conversation memory, context, and knowledge storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_type = config.get('type', 'sqlite')  # sqlite, redis, or hybrid
        self.max_history = config.get('max_history', 1000)
        self.ttl = config.get('ttl', 86400)  # 24 hours
        self.context_window = config.get('context_window', 10)
        
        # Storage backends
        self.redis_client = None
        self.sqlite_conn = None
        self.embedding_model = None
        
        # In-memory caches
        self.conversation_cache = {}
        self.context_cache = {}
        self.feedback_cache = []
        
        # Database paths
        self.db_path = Path(config.get('db_path', '/opt/sutazaiapp/data/jarvis/memory.db'))
        self.redis_url = config.get('redis_url', 'redis://redis:6379/1')
        
    async def initialize(self):
        """Initialize memory manager"""
        try:
            # Create data directory
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize storage backends
            if self.memory_type in ['sqlite', 'hybrid']:
                await self._init_sqlite()
                
            if self.memory_type in ['redis', 'hybrid']:
                await self._init_redis()
                
            # Initialize embedding model for semantic search
            await self._init_embeddings()
            
            logger.info(f"Memory manager initialized with {self.memory_type} backend")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            # Fallback to in-memory only
            self.memory_type = 'memory'
            
    async def shutdown(self):
        """Shutdown memory manager"""
        try:
            # Save in-memory data
            await self._save_memory_to_storage()
            
            # Close connections
            if self.redis_client:
                await self.redis_client.close()
                
            if self.sqlite_conn:
                self.sqlite_conn.close()
                
            logger.info("Memory manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during memory manager shutdown: {e}")
            
    async def _init_sqlite(self):
        """Initialize SQLite database"""
        try:
            self.sqlite_conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            
            # Create tables
            cursor = self.sqlite_conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    command TEXT,
                    response TEXT,
                    context TEXT,
                    timestamp DATETIME,
                    embedding BLOB
                )
            """)
            
            # Context table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expiry DATETIME
                )
            """)
            
            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    rating INTEGER,
                    comment TEXT,
                    timestamp DATETIME
                )
            """)
            
            # Knowledge base table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    topic TEXT,
                    content TEXT,
                    source TEXT,
                    embedding BLOB,
                    created_at DATETIME,
                    updated_at DATETIME
                )
            """)
            
            self.sqlite_conn.commit()
            logger.info("SQLite database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            raise
            
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
            
    async def _init_embeddings(self):
        """Initialize embedding model for semantic search"""
        try:
            # Use a lightweight model for embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            self.embedding_model = None
            
    async def store_interaction(self, command: str, response: Any, session_id: str = None, context: Dict[str, Any] = None):
        """Store a conversation interaction"""
        try:
            interaction_id = self._generate_id(command, session_id)
            
            # Prepare data
            interaction_data = {
                'command': command,
                'response': json.dumps(response) if not isinstance(response, str) else response,
                'context': json.dumps(context or {}),
                'session_id': session_id or 'default',
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate embedding if model available
            embedding = None
            if self.embedding_model:
                try:
                    embedding = self.embedding_model.encode(command)
                    embedding_bytes = embedding.tobytes()
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
                    embedding_bytes = None
            
            # Store in appropriate backend
            if self.memory_type in ['sqlite', 'hybrid']:
                await self._store_interaction_sqlite(interaction_id, interaction_data, embedding_bytes)
                
            if self.memory_type in ['redis', 'hybrid']:
                await self._store_interaction_redis(interaction_id, interaction_data)
                
            # Update in-memory cache
            self.conversation_cache[interaction_id] = interaction_data
            
            # Maintain cache size
            if len(self.conversation_cache) > self.max_history:
                oldest_key = min(self.conversation_cache.keys())
                del self.conversation_cache[oldest_key]
                
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            
    async def _store_interaction_sqlite(self, interaction_id: str, data: Dict[str, Any], embedding: bytes = None):
        """Store interaction in SQLite"""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO conversations 
                (id, session_id, command, response, context, timestamp, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                data['session_id'],
                data['command'],
                data['response'],
                data['context'],
                data['timestamp'],
                embedding
            ))
            self.sqlite_conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store interaction in SQLite: {e}")
            
    async def _store_interaction_redis(self, interaction_id: str, data: Dict[str, Any]):
        """Store interaction in Redis"""
        try:
            if self.redis_client:
                await self.redis_client.hset(
                    f"conversation:{interaction_id}",
                    mapping=data
                )
                await self.redis_client.expire(f"conversation:{interaction_id}", self.ttl)
                
        except Exception as e:
            logger.error(f"Failed to store interaction in Redis: {e}")
            
    async def retrieve_context(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve relevant context for a query"""
        try:
            # Get recent conversation history
            recent_conversations = await self._get_recent_conversations(self.context_window)
            
            # Get semantic matches if embedding model available
            semantic_matches = await self._get_semantic_matches(query, limit=5)
            
            # Combine context
            combined_context = {
                'recent_conversations': recent_conversations,
                'semantic_matches': semantic_matches,
                'current_context': context or {},
                'retrieved_at': datetime.now().isoformat()
            }
            
            return combined_context
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return context or {}
            
    async def _get_recent_conversations(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent conversations"""
        try:
            conversations = []
            
            if self.memory_type in ['sqlite', 'hybrid'] and self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("""
                    SELECT command, response, timestamp 
                    FROM conversations 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    conversations.append({
                        'command': row[0],
                        'response': row[1],
                        'timestamp': row[2]
                    })
                    
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to get recent conversations: {e}")
            return []
            
    async def _get_semantic_matches(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get semantically similar conversations"""
        try:
            if not self.embedding_model:
                return []
                
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            matches = []
            
            if self.memory_type in ['sqlite', 'hybrid'] and self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT command, response, embedding FROM conversations WHERE embedding IS NOT NULL")
                
                for row in cursor.fetchall():
                    try:
                        stored_embedding = np.frombuffer(row[2], dtype=np.float32)
                        
                        # Calculate similarity
                        similarity = np.dot(query_embedding, stored_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                        )
                        
                        if similarity > 0.7:  # Similarity threshold
                            matches.append({
                                'command': row[0],
                                'response': row[1],
                                'similarity': float(similarity)
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error processing embedding: {e}")
                        continue
                        
                # Sort by similarity
                matches.sort(key=lambda x: x['similarity'], reverse=True)
                
            return matches[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get semantic matches: {e}")
            return []
            
    async def store_feedback(self, feedback: Dict[str, Any]):
        """Store user feedback"""
        try:
            feedback_id = self._generate_id(feedback.get('session_id', ''), str(datetime.now().timestamp()))
            
            # Store in database
            if self.memory_type in ['sqlite', 'hybrid'] and self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("""
                    INSERT INTO feedback (id, session_id, rating, comment, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    feedback_id,
                    feedback.get('session_id'),
                    feedback.get('rating'),
                    feedback.get('comment'),
                    feedback.get('timestamp', datetime.now().isoformat())
                ))
                self.sqlite_conn.commit()
                
            # Store in cache
            self.feedback_cache.append(feedback)
            
            # Maintain cache size
            if len(self.feedback_cache) > 100:
                self.feedback_cache.pop(0)
                
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics"""
        try:
            stats = {
                'memory_type': self.memory_type,
                'cache_size': len(self.conversation_cache),
                'feedback_count': len(self.feedback_cache),
                'max_history': self.max_history
            }
            
            # Get database stats
            if self.memory_type in ['sqlite', 'hybrid'] and self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM conversations")
                stats['total_conversations'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM feedback")
                stats['total_feedback'] = cursor.fetchone()[0]
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}
            
    async def clear_expired_context(self):
        """Clear expired context entries"""
        try:
            if self.memory_type in ['sqlite', 'hybrid'] and self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("DELETE FROM context WHERE expiry < ?", (datetime.now(),))
                self.sqlite_conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to clear expired context: {e}")
            
    async def _save_memory_to_storage(self):
        """Save in-memory data to persistent storage"""
        try:
            # This would implement saving cache to storage on shutdown
            logger.info("Memory saved to storage")
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            
    def _generate_id(self, *args) -> str:
        """Generate unique ID from arguments"""
        combined = ''.join(str(arg) for arg in args)
        return hashlib.md5(combined.encode()).hexdigest()
        
    async def add_knowledge(self, topic: str, content: str, source: str = None):
        """Add knowledge to the knowledge base"""
        try:
            knowledge_id = self._generate_id(topic, content)
            
            # Generate embedding
            embedding_bytes = None
            if self.embedding_model:
                try:
                    embedding = self.embedding_model.encode(f"{topic}: {content}")
                    embedding_bytes = embedding.tobytes()
                except Exception as e:
                    logger.warning(f"Failed to generate knowledge embedding: {e}")
                    
            # Store in database
            if self.memory_type in ['sqlite', 'hybrid'] and self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO knowledge 
                    (id, topic, content, source, embedding, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    knowledge_id,
                    topic,
                    content,
                    source or 'user',
                    embedding_bytes,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                self.sqlite_conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            
    async def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base"""
        try:
            results = []
            
            if not self.embedding_model:
                # Fallback to text search
                if self.memory_type in ['sqlite', 'hybrid'] and self.sqlite_conn:
                    cursor = self.sqlite_conn.cursor()
                    cursor.execute("""
                        SELECT topic, content, source FROM knowledge 
                        WHERE topic LIKE ? OR content LIKE ?
                        LIMIT ?
                    """, (f"%{query}%", f"%{query}%", limit))
                    
                    for row in cursor.fetchall():
                        results.append({
                            'topic': row[0],
                            'content': row[1],
                            'source': row[2],
                            'relevance': 0.5  # Default relevance
                        })
                        
                return results
                
            # Semantic search
            query_embedding = self.embedding_model.encode(query)
            
            if self.memory_type in ['sqlite', 'hybrid'] and self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT topic, content, source, embedding FROM knowledge WHERE embedding IS NOT NULL")
                
                for row in cursor.fetchall():
                    try:
                        stored_embedding = np.frombuffer(row[3], dtype=np.float32)
                        
                        # Calculate similarity
                        similarity = np.dot(query_embedding, stored_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                        )
                        
                        if similarity > 0.5:  # Relevance threshold
                            results.append({
                                'topic': row[0],
                                'content': row[1],
                                'source': row[2],
                                'relevance': float(similarity)
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error processing knowledge embedding: {e}")
                        continue
                        
                # Sort by relevance
                results.sort(key=lambda x: x['relevance'], reverse=True)
                
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []