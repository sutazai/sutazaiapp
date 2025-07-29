#!/usr/bin/env python3
"""
Vector Memory System for the Brain
Implements multi-layer memory architecture with Redis, Qdrant, and ChromaDB
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid
import numpy as np

import redis.asyncio as redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb
from chromadb.config import Settings
import asyncpg

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorMemory:
    """Multi-layer vector memory system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(
            config.get('default_embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Initialize memory layers
        self._init_redis()
        self._init_qdrant()
        self._init_chroma()
        self._init_postgres()
        
        # Memory management
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour
        self.max_memories = config.get('max_memories', 1000000)
        
    def _init_redis(self):
        """Initialize Redis for L1 cache and session memory"""
        self.redis = redis.Redis(
            host=self.config.get('redis_host', 'sutazai-redis'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        logger.info("✅ Redis connection initialized")
        
    def _init_qdrant(self):
        """Initialize Qdrant for fast vector search"""
        self.qdrant = QdrantClient(
            host=self.config.get('qdrant_host', 'sutazai-qdrant'),
            port=self.config.get('qdrant_port', 6333)
        )
        
        # Create collection if doesn't exist
        try:
            self.qdrant.create_collection(
                collection_name="brain_memories",
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info("✅ Qdrant collection created")
        except:
            logger.info("✅ Qdrant collection already exists")
            
    def _init_chroma(self):
        """Initialize ChromaDB for long-term semantic memory"""
        self.chroma = chromadb.Client(Settings(
            chroma_server_host=self.config.get('chroma_host', 'sutazai-chromadb'),
            chroma_server_http_port=self.config.get('chroma_port', 8000)
        ))
        
        # Get or create collection
        self.chroma_collection = self.chroma.get_or_create_collection(
            name="brain_long_term_memory",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("✅ ChromaDB collection initialized")
        
    def _init_postgres(self):
        """Initialize PostgreSQL for structured data and audit trail"""
        # Connection will be created on demand
        self.pg_config = {
            'host': self.config.get('postgres_host', 'sutazai-postgresql'),
            'port': self.config.get('postgres_port', 5432),
            'database': self.config.get('postgres_db', 'sutazai_brain'),
            'user': self.config.get('postgres_user', 'sutazai'),
            'password': self.config.get('postgres_password', 'sutazai_password')
        }
        
    async def _get_pg_connection(self):
        """Get PostgreSQL connection"""
        return await asyncpg.connect(**self.pg_config)
        
    async def search(
        self,
        query: str,
        top_k: int = 10,
        include_metadata: bool = True,
        memory_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search across all memory layers"""
        # Check L1 cache first
        cache_key = f"search:{query}:{top_k}"
        cached = await self.redis.get(cache_key)
        if cached:
            logger.info("📌 Cache hit for search query")
            return json.loads(cached)
            
        # Generate embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search in parallel across memory layers
        results = await asyncio.gather(
            self._search_qdrant(query_embedding, top_k),
            self._search_chroma(query_embedding, top_k),
            return_exceptions=True
        )
        
        # Merge and rank results
        all_results = []
        for layer_results in results:
            if not isinstance(layer_results, Exception):
                all_results.extend(layer_results)
                
        # Sort by score and deduplicate
        seen_ids = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
                
        # Take top k
        final_results = unique_results[:top_k]
        
        # Cache results
        await self.redis.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(final_results)
        )
        
        # Log search to audit trail
        await self._log_search(query, len(final_results))
        
        return final_results
        
    async def _search_qdrant(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search in Qdrant for recent memories"""
        try:
            results = self.qdrant.search(
                collection_name="brain_memories",
                query_vector=query_embedding,
                limit=top_k
            )
            
            return [
                {
                    'id': str(hit.id),
                    'score': hit.score,
                    'content': hit.payload.get('content', ''),
                    'metadata': hit.payload.get('metadata', {}),
                    'timestamp': hit.payload.get('timestamp', ''),
                    'source': 'qdrant'
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []
            
    async def _search_chroma(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search in ChromaDB for long-term memories"""
        try:
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            memories = []
            for i in range(len(results['ids'][0])):
                memories.append({
                    'id': results['ids'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] or {},
                    'timestamp': results['metadatas'][0][i].get('timestamp', ''),
                    'source': 'chroma'
                })
                
            return memories
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            return []
            
    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: str = "general"
    ) -> str:
        """Store a memory across appropriate layers"""
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Generate embedding
        embedding = self.embedder.encode(content).tolist()
        
        # Prepare memory object
        memory = {
            'id': memory_id,
            'content': content,
            'embedding': embedding,
            'metadata': metadata or {},
            'memory_type': memory_type,
            'timestamp': timestamp
        }
        
        # Store in Redis (short-term)
        await self._store_redis(memory)
        
        # Store in Qdrant (medium-term)
        await self._store_qdrant(memory)
        
        # Store in ChromaDB (long-term) if important
        if metadata and metadata.get('importance', 0) > 0.7:
            await self._store_chroma(memory)
            
        # Log to PostgreSQL
        await self._log_memory_creation(memory)
        
        logger.info(f"💾 Stored memory {memory_id}")
        return memory_id
        
    async def store_batch(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Store multiple memories efficiently"""
        # Generate embeddings in batch
        contents = [m.get('content', '') for m in memories]
        embeddings = self.embedder.encode(contents)
        
        memory_ids = []
        for i, memory in enumerate(memories):
            memory['embedding'] = embeddings[i].tolist()
            memory['id'] = memory.get('id', str(uuid.uuid4()))
            memory['timestamp'] = memory.get('timestamp', datetime.now()).isoformat()
            memory_ids.append(memory['id'])
            
        # Batch store operations
        await asyncio.gather(
            self._store_redis_batch(memories),
            self._store_qdrant_batch(memories),
            self._store_chroma_batch([m for m in memories if m.get('metadata', {}).get('importance', 0) > 0.7])
        )
        
        # Log batch creation
        await self._log_batch_creation(len(memories))
        
        logger.info(f"💾 Stored {len(memories)} memories in batch")
        return memory_ids
        
    async def _store_redis(self, memory: Dict[str, Any]):
        """Store in Redis with TTL"""
        key = f"memory:{memory['id']}"
        # Store without embedding for Redis (too large)
        redis_memory = {k: v for k, v in memory.items() if k != 'embedding'}
        await self.redis.setex(
            key,
            self.cache_ttl,
            json.dumps(redis_memory, default=str)
        )
        
    async def _store_redis_batch(self, memories: List[Dict[str, Any]]):
        """Batch store in Redis"""
        pipe = self.redis.pipeline()
        for memory in memories:
            key = f"memory:{memory['id']}"
            redis_memory = {k: v for k, v in memory.items() if k != 'embedding'}
            pipe.setex(key, self.cache_ttl, json.dumps(redis_memory, default=str))
        await pipe.execute()
        
    async def _store_qdrant(self, memory: Dict[str, Any]):
        """Store in Qdrant"""
        point = PointStruct(
            id=memory['id'],
            vector=memory['embedding'],
            payload={
                'content': memory['content'],
                'metadata': memory['metadata'],
                'memory_type': memory['memory_type'],
                'timestamp': memory['timestamp']
            }
        )
        
        self.qdrant.upsert(
            collection_name="brain_memories",
            points=[point]
        )
        
    async def _store_qdrant_batch(self, memories: List[Dict[str, Any]]):
        """Batch store in Qdrant"""
        if not memories:
            return
            
        points = [
            PointStruct(
                id=m['id'],
                vector=m['embedding'],
                payload={
                    'content': m['content'],
                    'metadata': m['metadata'],
                    'memory_type': m.get('memory_type', 'general'),
                    'timestamp': m['timestamp']
                }
            )
            for m in memories
        ]
        
        self.qdrant.upsert(
            collection_name="brain_memories",
            points=points
        )
        
    async def _store_chroma(self, memory: Dict[str, Any]):
        """Store in ChromaDB"""
        self.chroma_collection.add(
            ids=[memory['id']],
            documents=[memory['content']],
            embeddings=[memory['embedding']],
            metadatas=[{
                **memory['metadata'],
                'memory_type': memory['memory_type'],
                'timestamp': memory['timestamp']
            }]
        )
        
    async def _store_chroma_batch(self, memories: List[Dict[str, Any]]):
        """Batch store in ChromaDB"""
        if not memories:
            return
            
        self.chroma_collection.add(
            ids=[m['id'] for m in memories],
            documents=[m['content'] for m in memories],
            embeddings=[m['embedding'] for m in memories],
            metadatas=[{
                **m['metadata'],
                'memory_type': m.get('memory_type', 'general'),
                'timestamp': m['timestamp']
            } for m in memories]
        )
        
    async def _log_search(self, query: str, results_count: int):
        """Log search to audit trail"""
        conn = await self._get_pg_connection()
        try:
            await conn.execute("""
                INSERT INTO memory_searches (query, results_count, timestamp)
                VALUES ($1, $2, $3)
            """, query, results_count, datetime.now())
        finally:
            await conn.close()
            
    async def _log_memory_creation(self, memory: Dict[str, Any]):
        """Log memory creation to audit trail"""
        conn = await self._get_pg_connection()
        try:
            await conn.execute("""
                INSERT INTO memory_audit (memory_id, action, metadata, timestamp)
                VALUES ($1, $2, $3, $4)
            """, memory['id'], 'created', json.dumps(memory['metadata']), datetime.now())
        finally:
            await conn.close()
            
    async def _log_batch_creation(self, count: int):
        """Log batch memory creation"""
        conn = await self._get_pg_connection()
        try:
            await conn.execute("""
                INSERT INTO memory_stats (action, count, timestamp)
                VALUES ($1, $2, $3)
            """, 'batch_create', count, datetime.now())
        finally:
            await conn.close()
            
    async def cleanup_old_memories(self, days: int = 30):
        """Clean up old memories based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean Qdrant
        # Note: Qdrant doesn't have direct date filtering, would need to scan
        
        # Clean ChromaDB
        # Would need to implement based on metadata filtering
        
        logger.info(f"🧹 Cleaned up memories older than {days} days")
        
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        stats = {
            'redis_keys': await self.redis.dbsize(),
            'qdrant_points': self.qdrant.count("brain_memories").count,
            'chroma_documents': len(self.chroma_collection.get()['ids']),
            'embedding_model': self.config.get('default_embedding_model'),
            'embedding_dim': self.embedding_dim
        }
        
        return stats