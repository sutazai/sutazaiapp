"""
Vector Database Context Injection System
Implements concurrent querying of ChromaDB, Qdrant, and FAISS for knowledge enrichment
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import httpx
import json

# Vector DB specific imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Result from vector database search"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str  # 'chromadb', 'qdrant', 'faiss'
    
@dataclass
class KnowledgeContext:
    """Consolidated knowledge context from multiple vector DBs"""
    results: List[VectorSearchResult]
    query_time_ms: float
    sources_used: List[str]
    total_results: int
    enriched_context: str

class CircuitBreaker:
    """Circuit breaker for failing vector database connections"""
    
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func):
        """Decorator to wrap functions with circuit breaker"""
        async def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time > self.reset_timeout
        )
    
    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class ChromaDBClient:
    """ChromaDB client with circuit breaker protection"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 10100):
        self.host = host
        self.port = port
        self.client = None
        self.circuit_breaker = CircuitBreaker()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB library not available")
            return
        
        try:
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                settings=Settings(
                    chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                    chroma_client_auth_credentials="test-token"
                )
            )
        except Exception as e:
            logger.warning(f"ChromaDB client initialization failed: {e}")
    
    async def search(self, query: str, collection_name: str = "knowledge", limit: int = 3) -> List[VectorSearchResult]:
        """Search ChromaDB for relevant documents"""
        if not self.client:
            return []
        
        # Apply circuit breaker manually
        if self.circuit_breaker.state == "OPEN":
            if self.circuit_breaker._should_attempt_reset():
                self.circuit_breaker.state = "HALF_OPEN"
            else:
                logger.warning("ChromaDB circuit breaker is OPEN")
                return []
        
        try:
            collection = self.client.get_collection(collection_name)
            results = collection.query(
                query_texts=[query],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            vector_results = []
            if results and results["documents"]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
                ):
                    vector_results.append(VectorSearchResult(
                        content=doc,
                        metadata=metadata or {},
                        score=1.0 - distance,  # Convert distance to similarity score
                        source="chromadb"
                    ))
            
            self.circuit_breaker._on_success()
            return vector_results
        
        except Exception as e:
            self.circuit_breaker._on_failure()
            logger.warning(f"ChromaDB search failed: {e}")
            return []

class QdrantDBClient:
    """Qdrant client with circuit breaker protection"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 10101):
        self.host = host
        self.port = port
        self.client = None
        self.circuit_breaker = CircuitBreaker()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client"""
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant library not available")
            return
        
        try:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                timeout=5.0
            )
        except Exception as e:
            logger.warning(f"Qdrant client initialization failed: {e}")
    
    async def _encode_query(self, query: str) -> List[float]:
        """Encode query text to vector (simplified - would use actual embedding model)"""
        # Simple hash-based encoding for demo - in production use proper embedding model
        hash_obj = hashlib.md5(query.encode())
        hex_dig = hash_obj.hexdigest()
        # Convert to 384-dimensional vector
        vector = [float(ord(c)) / 255.0 for c in hex_dig] * 12
        return vector[:384]
    
    async def search(self, query: str, collection_name: str = "knowledge", limit: int = 3) -> List[VectorSearchResult]:
        """Search Qdrant for relevant documents"""
        if not self.client:
            return []
        
        # Apply circuit breaker manually
        if self.circuit_breaker.state == "OPEN":
            if self.circuit_breaker._should_attempt_reset():
                self.circuit_breaker.state = "HALF_OPEN"
            else:
                logger.warning("Qdrant circuit breaker is OPEN")
                return []
        
        try:
            query_vector = await self._encode_query(query)
            
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            
            vector_results = []
            for point in search_result:
                vector_results.append(VectorSearchResult(
                    content=point.payload.get("content", ""),
                    metadata=point.payload.get("metadata", {}),
                    score=point.score,
                    source="qdrant"
                ))
            
            self.circuit_breaker._on_success()
            return vector_results
        
        except Exception as e:
            self.circuit_breaker._on_failure()
            logger.warning(f"Qdrant search failed: {e}")
            return []

class FAISSClient:
    """FAISS client with circuit breaker protection"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 10103):
        self.host = host
        self.port = port
        self.circuit_breaker = CircuitBreaker()
    
    async def _query_faiss_service(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Query FAISS service via HTTP"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"http://{self.host}:{self.port}/search",
                    json={"query": query, "limit": limit}
                )
                if response.status_code == 200:
                    return response.json().get("results", [])
                else:
                    logger.warning(f"FAISS service returned status {response.status_code}")
                    return []
        except Exception as e:
            logger.warning(f"FAISS service query failed: {e}")
            return []
    
    async def search(self, query: str, limit: int = 3) -> List[VectorSearchResult]:
        """Search FAISS for relevant documents"""
        # Apply circuit breaker manually
        if self.circuit_breaker.state == "OPEN":
            if self.circuit_breaker._should_attempt_reset():
                self.circuit_breaker.state = "HALF_OPEN"
            else:
                logger.warning("FAISS circuit breaker is OPEN")
                return []
        
        try:
            results = await self._query_faiss_service(query, limit)
            
            vector_results = []
            for result in results:
                vector_results.append(VectorSearchResult(
                    content=result.get("content", ""),
                    metadata=result.get("metadata", {}),
                    score=result.get("score", 0.0),
                    source="faiss"
                ))
            
            self.circuit_breaker._on_success()
            return vector_results
        
        except Exception as e:
            self.circuit_breaker._on_failure()
            logger.warning(f"FAISS search failed: {e}")
            return []

class VectorContextInjector:
    """Main vector context injection system"""
    
    def __init__(self):
        self.chromadb_client = ChromaDBClient()
        self.qdrant_client = QdrantDBClient()
        self.faiss_client = FAISSClient()
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes
    
    def _is_knowledge_query(self, query: str) -> bool:
        """Detect if query requires knowledge context injection"""
        knowledge_keywords = [
            "what is", "how to", "explain", "define", "describe", "tell me about",
            "information", "details", "facts", "history", "background", "overview",
            "help me understand", "can you explain", "what are", "who is", "where is",
            "when was", "why does", "how does", "tutorial", "guide", "documentation"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in knowledge_keywords)
    
    async def _get_cached_results(self, query_hash: str) -> Optional[KnowledgeContext]:
        """Get cached results if available and not expired"""
        if query_hash in self.cache:
            cached_result, timestamp = self.cache[query_hash]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
            else:
                del self.cache[query_hash]
        return None
    
    def _cache_results(self, query_hash: str, context: KnowledgeContext):
        """Cache results for future use"""
        self.cache[query_hash] = (context, time.time())
    
    async def search_all_databases(self, query: str) -> KnowledgeContext:
        """Search all vector databases concurrently"""
        start_time = time.time()
        
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cached_context = await self._get_cached_results(query_hash)
        if cached_context:
            logger.info("Returning cached vector search results")
            return cached_context
        
        # Create concurrent search tasks
        search_tasks = [
            self.chromadb_client.search(query),
            self.qdrant_client.search(query),
            self.faiss_client.search(query)
        ]
        
        # Execute searches concurrently with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*search_tasks, return_exceptions=True),
                timeout=0.5  # 500ms timeout to meet requirement
            )
        except asyncio.TimeoutError:
            logger.warning("Vector database searches timed out")
            results = [[], [], []]
        
        # Process results
        all_results = []
        sources_used = []
        
        for i, (result, source) in enumerate(zip(results, ["chromadb", "qdrant", "faiss"])):
            if isinstance(result, list) and result:
                all_results.extend(result)
                sources_used.append(source)
            elif isinstance(result, Exception):
                logger.warning(f"{source} search failed: {result}")
        
        # Deduplicate results based on content similarity
        deduplicated_results = self._deduplicate_results(all_results)
        
        # Create enriched context
        enriched_context = self._create_enriched_context(deduplicated_results, query)
        
        # Create knowledge context
        context = KnowledgeContext(
            results=deduplicated_results[:9],  # Top 9 results (3 from each DB)
            query_time_ms=(time.time() - start_time) * 1000,
            sources_used=sources_used,
            total_results=len(deduplicated_results),
            enriched_context=enriched_context
        )
        
        # Cache the results
        self._cache_results(query_hash, context)
        
        logger.info(f"Vector search completed in {context.query_time_ms:.1f}ms, found {context.total_results} results")
        return context
    
    def _deduplicate_results(self, results: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """Deduplicate results based on content similarity"""
        if not results:
            return []
        
        # Sort by score (highest first)
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Simple deduplication based on content hash
        seen_hashes = set()
        deduplicated = []
        
        for result in sorted_results:
            content_hash = hashlib.md5(result.content.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(result)
        
        return deduplicated
    
    def _create_enriched_context(self, results: List[VectorSearchResult], query: str) -> str:
        """Create enriched context string from search results"""
        if not results:
            return ""
        
        context_parts = [
            f"KNOWLEDGE CONTEXT FOR: {query}",
            "=" * 50,
            ""
        ]
        
        for i, result in enumerate(results[:6], 1):  # Top 6 results
            context_parts.append(f"[{i}] Source: {result.source.upper()} (Score: {result.score:.3f})")
            context_parts.append(f"Content: {result.content[:500]}...")
            if result.metadata:
                context_parts.append(f"Metadata: {json.dumps(result.metadata)}")
            context_parts.append("")
        
        context_parts.extend([
            "=" * 50,
            "Use this knowledge context to provide accurate, informed responses.",
            ""
        ])
        
        return "\n".join(context_parts)
    
    async def analyze_user_request(self, user_message: str) -> Tuple[bool, Optional[KnowledgeContext]]:
        """
        Analyze user request to determine if knowledge context injection is needed
        Returns: (needs_context, knowledge_context)
        """
        try:
            # Detect if this is a knowledge-seeking query
            needs_context = self._is_knowledge_query(user_message)
            
            if not needs_context:
                return False, None
            
            # Search vector databases for relevant context
            knowledge_context = await self.search_all_databases(user_message)
            
            # Only return context if we found meaningful results
            if knowledge_context.total_results > 0:
                return True, knowledge_context
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Error in analyze_user_request: {e}")
            return False, None
    
    async def inject_context_into_prompt(self, original_prompt: str, knowledge_context: KnowledgeContext) -> str:
        """Inject knowledge context into the original prompt"""
        if not knowledge_context or not knowledge_context.enriched_context:
            return original_prompt
        
        enhanced_prompt = f"""
{knowledge_context.enriched_context}

ORIGINAL REQUEST: {original_prompt}

INSTRUCTIONS: Use the provided knowledge context above to give an accurate, well-informed response. Reference specific sources when appropriate.
"""
        
        return enhanced_prompt.strip()

# Global instance
vector_context_injector = VectorContextInjector()