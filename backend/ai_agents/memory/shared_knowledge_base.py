"""
Shared Knowledge Base for automation Agents
====================================

Advanced shared memory system that enables cross-agent learning,
knowledge persistence, and intelligent information retrieval
across all 28 AI agents in the SutazAI system.
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import qdrant_client
from qdrant_client.models import VectorParams, Distance, PointStruct
import redis.asyncio as redis
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    TASK_EXECUTION = "task_execution"
    CODE_SOLUTION = "code_solution"
    RESEARCH_FINDING = "research_finding"
    ERROR_PATTERN = "error_pattern"
    OPTIMIZATION_TIP = "optimization_tip"
    AGENT_CAPABILITY = "agent_capability"
    USER_PREFERENCE = "user_preference"
    SYSTEM_CONFIGURATION = "system_configuration"
    LEARNING_OUTCOME = "learning_outcome"
    COLLABORATION_PATTERN = "collaboration_pattern"


class ConfidenceLevel(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class KnowledgeItem:
    """Structured knowledge item for the shared knowledge base"""
    id: str = None
    type: KnowledgeType = KnowledgeType.TASK_EXECUTION
    content: str = ""
    metadata: Dict[str, Any] = None
    tags: Set[str] = None
    source_agent: str = ""
    created_at: float = None
    updated_at: float = None
    access_count: int = 0
    confidence: float = 0.8
    validation_score: float = 0.0
    related_items: List[str] = None
    embeddings: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.id is None:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.id = f"{self.type.value}_{content_hash}_{int(time.time())}"
        
        if self.created_at is None:
            self.created_at = time.time()
        
        if self.updated_at is None:
            self.updated_at = self.created_at
        
        if self.metadata is None:
            self.metadata = {}
        
        if self.tags is None:
            self.tags = set()
        
        if self.related_items is None:
            self.related_items = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'type': self.type.value,
            'content': self.content,
            'metadata': json.dumps(self.metadata),
            'tags': list(self.tags),
            'source_agent': self.source_agent,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'access_count': self.access_count,
            'confidence': self.confidence,
            'validation_score': self.validation_score,
            'related_items': self.related_items,
            'embeddings': self.embeddings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeItem':
        """Create knowledge item from dictionary"""
        return cls(
            id=data['id'],
            type=KnowledgeType(data['type']),
            content=data['content'],
            metadata=json.loads(data.get('metadata', '{}')),
            tags=set(data.get('tags', [])),
            source_agent=data['source_agent'],
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            access_count=data.get('access_count', 0),
            confidence=data.get('confidence', 0.8),
            validation_score=data.get('validation_score', 0.0),
            related_items=data.get('related_items', []),
            embeddings=data.get('embeddings')
        )


class SharedKnowledgeBase:
    """
    Advanced shared knowledge base for automation agents
    
    Features:
    - Multi-modal storage (vector + graph + key-value)
    - Semantic search and similarity matching
    - Cross-agent knowledge validation
    - Intelligent knowledge ranking
    - Automatic knowledge decay and refresh
    - Real-time knowledge synchronization
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/1",
                 chromadb_host: str = "localhost",
                 chromadb_port: int = 8001,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333):
        
        self.redis_url = redis_url
        self.redis_client = None
        
        # Vector databases
        self.chroma_client = None
        self.chroma_collection = None
        self.qdrant_client = None
        
        # Embedding model
        self.embedding_model = None
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Configuration
        self.chromadb_host = chromadb_host
        self.chromadb_port = chromadb_port
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        # Internal state
        self.knowledge_cache: Dict[str, KnowledgeItem] = {}
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.validation_network: Dict[str, Set[str]] = defaultdict(set)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
    
    async def initialize(self):
        """Initialize all storage backends and start background processes"""
        
        # Initialize Redis
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        logger.info("Connected to Redis knowledge base")
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.HttpClient(
                host=self.chromadb_host, 
                port=self.chromadb_port
            )
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="agi_knowledge",
                metadata={"description": "Shared knowledge base for automation agents"}
            )
            logger.info("Connected to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            self.chroma_client = None
        
        # Initialize Qdrant
        try:
            self.qdrant_client = qdrant_client.QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            
            # Create collection if it doesn't exist
            collections = await self.qdrant_client.get_collections()
            if "agi_knowledge" not in [c.name for c in collections.collections]:
                await self.qdrant_client.create_collection(
                    collection_name="agi_knowledge",
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
            logger.info("Connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.qdrant_client = None
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
        
        # Start background processes
        self.running = True
        self.background_tasks = [
            asyncio.create_task(self._knowledge_validation_worker()),
            asyncio.create_task(self._knowledge_decay_worker()),
            asyncio.create_task(self._cache_optimization_worker()),
            asyncio.create_task(self._cross_reference_builder())
        ]
        
        logger.info("Shared knowledge base initialized successfully")
        return True
    
    async def shutdown(self):
        """Clean shutdown of knowledge base"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Shared knowledge base shut down")
    
    async def store_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """Store knowledge item across all storage backends"""
        try:
            # Generate embeddings if not present
            if knowledge.embeddings is None:
                knowledge.embeddings = self._generate_embeddings(knowledge.content)
            
            # Store in Redis (primary store)
            await self.redis_client.hset(
                "agi:knowledge",
                knowledge.id,
                json.dumps(knowledge.to_dict())
            )
            
            # Store in vector databases
            await self._store_in_vector_dbs(knowledge)
            
            # Update cache
            self.knowledge_cache[knowledge.id] = knowledge
            
            # Update access patterns
            self.access_patterns[knowledge.source_agent].append(time.time())
            
            logger.debug(f"Stored knowledge item: {knowledge.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store knowledge {knowledge.id}: {e}")
            return False
    
    async def retrieve_knowledge(self, 
                                query: str,
                                knowledge_types: Optional[List[KnowledgeType]] = None,
                                agent_id: str = "",
                                limit: int = 10,
                                min_confidence: float = 0.5) -> List[KnowledgeItem]:
        """Retrieve relevant knowledge items using semantic search"""
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings(query)
            
            # Search in vector databases
            results = []
            
            # ChromaDB search
            if self.chroma_client:
                chroma_results = await self._search_chromadb(
                    query_embedding, limit * 2
                )
                results.extend(chroma_results)
            
            # Qdrant search
            if self.qdrant_client:
                qdrant_results = await self._search_qdrant(
                    query_embedding, limit * 2
                )
                results.extend(qdrant_results)
            
            # Filter and rank results
            filtered_results = []
            for item in results:
                # Filter by knowledge type
                if knowledge_types and item.type not in knowledge_types:
                    continue
                
                # Filter by confidence
                if item.confidence < min_confidence:
                    continue
                
                # Update access count
                item.access_count += 1
                await self._update_access_count(item.id)
                
                filtered_results.append(item)
            
            # Rank by relevance and recency
            ranked_results = await self._rank_knowledge_items(
                filtered_results, query, agent_id
            )
            
            return ranked_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge for query '{query}': {e}")
            return []
    
    async def validate_knowledge(self, 
                               knowledge_id: str,
                               validator_agent: str,
                               validation_result: bool,
                               confidence: float = 0.8) -> bool:
        """Allow agents to validate each other's knowledge"""
        
        try:
            # Get existing knowledge
            knowledge = await self.get_knowledge_by_id(knowledge_id)
            if not knowledge:
                return False
            
            # Record validation
            validation_key = f"agi:validations:{knowledge_id}"
            validation_data = {
                'validator': validator_agent,
                'result': validation_result,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            await self.redis_client.lpush(
                validation_key,
                json.dumps(validation_data)
            )
            
            # Update validation network
            self.validation_network[knowledge_id].add(validator_agent)
            
            # Recalculate validation score
            new_score = await self._calculate_validation_score(knowledge_id)
            knowledge.validation_score = new_score
            
            # Update stored knowledge
            await self.store_knowledge(knowledge)
            
            logger.debug(f"Knowledge {knowledge_id} validated by {validator_agent}: {validation_result}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate knowledge {knowledge_id}: {e}")
            return False
    
    async def get_knowledge_by_id(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """Retrieve specific knowledge item by ID"""
        
        # Check cache first
        if knowledge_id in self.knowledge_cache:
            return self.knowledge_cache[knowledge_id]
        
        # Retrieve from Redis
        try:
            data = await self.redis_client.hget("agi:knowledge", knowledge_id)
            if data:
                knowledge = KnowledgeItem.from_dict(json.loads(data))
                self.knowledge_cache[knowledge_id] = knowledge
                return knowledge
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge {knowledge_id}: {e}")
        
        return None
    
    async def update_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """Update existing knowledge item"""
        knowledge.updated_at = time.time()
        return await self.store_knowledge(knowledge)
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete knowledge item from all stores"""
        try:
            # Remove from Redis
            await self.redis_client.hdel("agi:knowledge", knowledge_id)
            
            # Remove from vector databases
            if self.chroma_client:
                try:
                    self.chroma_collection.delete(ids=[knowledge_id])
                except Exception:
                    pass  # Item might not exist
            
            if self.qdrant_client:
                try:
                    await self.qdrant_client.delete(
                        collection_name="agi_knowledge",
                        points_selector=[knowledge_id]
                    )
                except Exception:
                    pass  # Item might not exist
            
            # Remove from cache
            self.knowledge_cache.pop(knowledge_id, None)
            
            logger.debug(f"Deleted knowledge item: {knowledge_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete knowledge {knowledge_id}: {e}")
            return False
    
    async def get_agent_knowledge_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get knowledge statistics for a specific agent"""
        
        stats = {
            'total_contributions': 0,
            'knowledge_types': defaultdict(int),
            'average_confidence': 0.0,
            'validation_score': 0.0,
            'access_patterns': [],
            'most_accessed_items': []
        }
        
        try:
            # Scan all knowledge items for this agent
            all_knowledge = await self.redis_client.hgetall("agi:knowledge")
            agent_items = []
            
            for item_data in all_knowledge.values():
                item = KnowledgeItem.from_dict(json.loads(item_data))
                if item.source_agent == agent_id:
                    agent_items.append(item)
                    stats['knowledge_types'][item.type.value] += 1
            
            if agent_items:
                stats['total_contributions'] = len(agent_items)
                stats['average_confidence'] = sum(item.confidence for item in agent_items) / len(agent_items)
                stats['validation_score'] = sum(item.validation_score for item in agent_items) / len(agent_items)
                
                # Most accessed items
                sorted_items = sorted(agent_items, key=lambda x: x.access_count, reverse=True)
                stats['most_accessed_items'] = [
                    {'id': item.id, 'content': item.content[:100], 'access_count': item.access_count}
                    for item in sorted_items[:5]
                ]
            
            # Access patterns
            stats['access_patterns'] = self.access_patterns.get(agent_id, [])
            
        except Exception as e:
            logger.error(f"Failed to get stats for agent {agent_id}: {e}")
        
        return stats
    
    async def get_system_knowledge_metrics(self) -> Dict[str, Any]:
        """Get overall system knowledge metrics"""
        
        metrics = {
            'total_items': 0,
            'knowledge_types': defaultdict(int),
            'average_confidence': 0.0,
            'validation_coverage': 0.0,
            'cache_hit_rate': 0.0,
            'top_contributors': [],
            'knowledge_growth_rate': 0.0,
            'timestamp': time.time()
        }
        
        try:
            # Get all knowledge items
            all_knowledge = await self.redis_client.hgetall("agi:knowledge")
            items = []
            contributor_counts = defaultdict(int)
            
            for item_data in all_knowledge.values():
                item = KnowledgeItem.from_dict(json.loads(item_data))
                items.append(item)
                metrics['knowledge_types'][item.type.value] += 1
                contributor_counts[item.source_agent] += 1
            
            if items:
                metrics['total_items'] = len(items)
                metrics['average_confidence'] = sum(item.confidence for item in items) / len(items)
                
                # Validation coverage
                validated_items = sum(1 for item in items if item.validation_score > 0)
                metrics['validation_coverage'] = validated_items / len(items)
                
                # Top contributors
                sorted_contributors = sorted(contributor_counts.items(), key=lambda x: x[1], reverse=True)
                metrics['top_contributors'] = sorted_contributors[:5]
            
            # Calculate growth rate (items created in last 24 hours)
            recent_items = sum(1 for item in items if time.time() - item.created_at < 86400)
            metrics['knowledge_growth_rate'] = recent_items
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
        
        return metrics
    
    # Internal Methods
    
    def _generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text content"""
        if self.embedding_model:
            return self.embedding_model.encode(text).tolist()
        return []
    
    async def _store_in_vector_dbs(self, knowledge: KnowledgeItem):
        """Store knowledge in vector databases"""
        
        # ChromaDB
        if self.chroma_client and knowledge.embeddings:
            try:
                self.chroma_collection.add(
                    embeddings=[knowledge.embeddings],
                    documents=[knowledge.content],
                    metadatas=[{
                        'type': knowledge.type.value,
                        'source_agent': knowledge.source_agent,
                        'confidence': knowledge.confidence,
                        'created_at': knowledge.created_at
                    }],
                    ids=[knowledge.id]
                )
            except Exception as e:
                logger.error(f"Failed to store in ChromaDB: {e}")
        
        # Qdrant
        if self.qdrant_client and knowledge.embeddings:
            try:
                await self.qdrant_client.upsert(
                    collection_name="agi_knowledge",
                    points=[PointStruct(
                        id=knowledge.id,
                        vector=knowledge.embeddings,
                        payload={
                            'content': knowledge.content,
                            'type': knowledge.type.value,
                            'source_agent': knowledge.source_agent,
                            'confidence': knowledge.confidence,
                            'created_at': knowledge.created_at
                        }
                    )]
                )
            except Exception as e:
                logger.error(f"Failed to store in Qdrant: {e}")
    
    async def _search_chromadb(self, query_embedding: List[float], limit: int) -> List[KnowledgeItem]:
        """Search ChromaDB for similar knowledge"""
        results = []
        
        if self.chroma_client:
            try:
                search_results = self.chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit
                )
                
                for i, doc_id in enumerate(search_results['ids'][0]):
                    knowledge = await self.get_knowledge_by_id(doc_id)
                    if knowledge:
                        results.append(knowledge)
                        
            except Exception as e:
                logger.error(f"ChromaDB search error: {e}")
        
        return results
    
    async def _search_qdrant(self, query_embedding: List[float], limit: int) -> List[KnowledgeItem]:
        """Search Qdrant for similar knowledge"""
        results = []
        
        if self.qdrant_client:
            try:
                search_results = await self.qdrant_client.search(
                    collection_name="agi_knowledge",
                    query_vector=query_embedding,
                    limit=limit
                )
                
                for result in search_results:
                    knowledge = await self.get_knowledge_by_id(result.id)
                    if knowledge:
                        results.append(knowledge)
                        
            except Exception as e:
                logger.error(f"Qdrant search error: {e}")
        
        return results
    
    async def _rank_knowledge_items(self, 
                                  items: List[KnowledgeItem],
                                  query: str,
                                  agent_id: str) -> List[KnowledgeItem]:
        """Rank knowledge items by relevance and other factors"""
        
        def calculate_score(item: KnowledgeItem) -> float:
            score = 0.0
            
            # Base confidence score
            score += item.confidence * 0.3
            
            # Validation score
            score += item.validation_score * 0.2
            
            # Recency score (newer items get slight boost)
            age_days = (time.time() - item.created_at) / 86400
            recency_score = max(0, 1 - age_days / 30)  # Decay over 30 days
            score += recency_score * 0.1
            
            # Access count (popular items get boost)
            access_score = min(1.0, item.access_count / 100)  # Normalize to max 1.0
            score += access_score * 0.2
            
            # Agent preference (items from similar agents)
            if item.source_agent == agent_id:
                score += 0.1  # Slight boost for own items
            
            # Content relevance (simplified - would use proper similarity in production)
            content_relevance = len(set(query.lower().split()) & 
                                  set(item.content.lower().split())) / max(len(query.split()), 1)
            score += content_relevance * 0.1
            
            return score
        
        # Sort by calculated score
        return sorted(items, key=calculate_score, reverse=True)
    
    async def _update_access_count(self, knowledge_id: str):
        """Update access count for knowledge item"""
        try:
            data = await self.redis_client.hget("agi:knowledge", knowledge_id)
            if data:
                item_data = json.loads(data)
                item_data['access_count'] = item_data.get('access_count', 0) + 1
                await self.redis_client.hset("agi:knowledge", knowledge_id, json.dumps(item_data))
        except Exception as e:
            logger.error(f"Failed to update access count for {knowledge_id}: {e}")
    
    async def _calculate_validation_score(self, knowledge_id: str) -> float:
        """Calculate validation score based on peer reviews"""
        try:
            validation_key = f"agi:validations:{knowledge_id}"
            validations = await self.redis_client.lrange(validation_key, 0, -1)
            
            if not validations:
                return 0.0
            
            total_score = 0.0
            total_weight = 0.0
            
            for validation_data in validations:
                validation = json.loads(validation_data)
                weight = validation['confidence']
                score = 1.0 if validation['result'] else 0.0
                
                total_score += score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate validation score for {knowledge_id}: {e}")
            return 0.0
    
    # Background Workers
    
    async def _knowledge_validation_worker(self):
        """Background worker for knowledge validation"""
        while self.running:
            try:
                # Implement peer validation logic
                # This could involve cross-checking with multiple agents
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Knowledge validation worker error: {e}")
                await asyncio.sleep(300)
    
    async def _knowledge_decay_worker(self):
        """Background worker for knowledge decay and cleanup"""
        while self.running:
            try:
                # Implement knowledge decay logic
                # Remove or lower confidence of old, unused knowledge
                current_time = time.time()
                threshold = current_time - (30 * 86400)  # 30 days
                
                all_knowledge = await self.redis_client.hgetall("agi:knowledge")
                for knowledge_id, item_data in all_knowledge.items():
                    item = KnowledgeItem.from_dict(json.loads(item_data))
                    
                    # If item is old and has low access, reduce confidence
                    if (item.created_at < threshold and 
                        item.access_count < 5 and 
                        item.validation_score < 0.3):
                        
                        item.confidence *= 0.9  # Reduce confidence by 10%
                        if item.confidence < 0.1:
                            await self.delete_knowledge(knowledge_id)
                        else:
                            await self.update_knowledge(item)
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Knowledge decay worker error: {e}")
                await asyncio.sleep(3600)
    
    async def _cache_optimization_worker(self):
        """Background worker for cache optimization"""
        while self.running:
            try:
                # Optimize cache by keeping frequently accessed items
                if len(self.knowledge_cache) > 1000:  # Max cache size
                    # Keep top 500 most accessed items
                    sorted_items = sorted(
                        self.knowledge_cache.items(),
                        key=lambda x: x[1].access_count,
                        reverse=True
                    )
                    
                    # Keep top 500 items
                    new_cache = dict(sorted_items[:500])
                    self.knowledge_cache = new_cache
                
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                logger.error(f"Cache optimization worker error: {e}")
                await asyncio.sleep(1800)
    
    async def _cross_reference_builder(self):
        """Background worker for building cross-references between knowledge items"""
        while self.running:
            try:
                # Build relationships between related knowledge items
                # This would involve similarity analysis and clustering
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Cross-reference builder error: {e}")
                await asyncio.sleep(3600)