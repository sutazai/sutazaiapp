#!/usr/bin/env python3
"""
SutazAI Vector Store Manager
Manages vector databases for embeddings and similarity search
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import httpx
import json
import numpy as np

from ..core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector stores across different services"""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.vector_stores = {}
        self.collections = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the vector store manager"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Vector Store Manager...")
            
            # Initialize vector store configurations
            await self._initialize_vector_stores()
            
            # Test connections
            await self._test_connections()
            
            # Create default collections
            await self._create_default_collections()
            
            self._initialized = True
            logger.info("Vector Store Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store Manager: {e}")
            raise
    
    async def _initialize_vector_stores(self):
        """Initialize vector store configurations"""
        self.vector_stores = {
            "chromadb": {
                "url": settings.CHROMADB_URL,
                "type": "chromadb",
                "capabilities": ["similarity_search", "metadata_filtering", "persistence"],
                "max_dimensions": 1536,
                "health_endpoint": "/api/v1/heartbeat"
            },
            "qdrant": {
                "url": settings.QDRANT_URL,
                "type": "qdrant",
                "capabilities": ["similarity_search", "filtering", "clustering", "payload_indexing"],
                "max_dimensions": 65536,
                "health_endpoint": "/health"
            },
            "faiss": {
                "url": settings.FAISS_URL,
                "type": "faiss",
                "capabilities": ["similarity_search", "clustering", "quantization"],
                "max_dimensions": 2048,
                "health_endpoint": "/health"
            }
        }
    
    async def _test_connections(self):
        """Test connections to vector stores"""
        for store_name, config in self.vector_stores.items():
            try:
                response = await self.http_client.get(f"{config['url']}{config['health_endpoint']}")
                if response.status_code == 200:
                    logger.info(f"Vector store {store_name} is healthy")
                else:
                    logger.warning(f"Vector store {store_name} health check failed: {response.status_code}")
            except Exception as e:
                logger.warning(f"Cannot reach vector store {store_name}: {e}")
    
    async def _create_default_collections(self):
        """Create default collections"""
        default_collections = [
            {
                "name": "documents",
                "description": "Document embeddings",
                "store": "chromadb",
                "dimension": 1536
            },
            {
                "name": "code_snippets",
                "description": "Code snippet embeddings",
                "store": "qdrant",
                "dimension": 768
            },
            {
                "name": "conversations",
                "description": "Conversation history embeddings",
                "store": "chromadb",
                "dimension": 1536
            }
        ]
        
        for collection_config in default_collections:
            try:
                await self.create_collection(
                    collection_config["name"],
                    collection_config["store"],
                    collection_config["dimension"],
                    collection_config["description"]
                )
            except Exception as e:
                logger.warning(f"Could not create default collection {collection_config['name']}: {e}")
    
    async def shutdown(self):
        """Shutdown the vector store manager"""
        logger.info("Shutting down Vector Store Manager...")
        self._initialized = False
        logger.info("Vector Store Manager shutdown complete")
    
    async def create_collection(self, name: str, store: str, dimension: int, description: str = None) -> Dict[str, Any]:
        """Create a new collection"""
        try:
            if store not in self.vector_stores:
                raise ValueError(f"Unknown vector store: {store}")
            
            store_config = self.vector_stores[store]
            
            if dimension > store_config["max_dimensions"]:
                raise ValueError(f"Dimension {dimension} exceeds maximum for {store}: {store_config['max_dimensions']}")
            
            collection_data = {
                "name": name,
                "store": store,
                "dimension": dimension,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
                "document_count": 0
            }
            
            # Create collection based on store type
            if store == "chromadb":
                await self._create_chromadb_collection(name, dimension, description)
            elif store == "qdrant":
                await self._create_qdrant_collection(name, dimension, description)
            elif store == "faiss":
                await self._create_faiss_collection(name, dimension, description)
            
            self.collections[name] = collection_data
            
            logger.info(f"Created collection {name} in {store}")
            return collection_data
            
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            raise
    
    async def _create_chromadb_collection(self, name: str, dimension: int, description: str = None):
        """Create ChromaDB collection"""
        request_data = {
            "name": name,
            "metadata": {
                "dimension": dimension,
                "description": description or ""
            }
        }
        
        response = await self.http_client.post(
            f"{self.vector_stores['chromadb']['url']}/api/v1/collections",
            json=request_data
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create ChromaDB collection: {response.status_code}")
    
    async def _create_qdrant_collection(self, name: str, dimension: int, description: str = None):
        """Create Qdrant collection"""
        request_data = {
            "vectors": {
                "size": dimension,
                "distance": "Cosine"
            },
            "optimizers_config": {
                "default_segment_number": 2
            }
        }
        
        response = await self.http_client.put(
            f"{self.vector_stores['qdrant']['url']}/collections/{name}",
            json=request_data
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create Qdrant collection: {response.status_code}")
    
    async def _create_faiss_collection(self, name: str, dimension: int, description: str = None):
        """Create FAISS collection"""
        request_data = {
            "name": name,
            "dimension": dimension,
            "index_type": "IVF",
            "metric": "L2",
            "description": description or ""
        }
        
        response = await self.http_client.post(
            f"{self.vector_stores['faiss']['url']}/collections",
            json=request_data
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create FAISS collection: {response.status_code}")
    
    async def add_vectors(self, collection_name: str, vectors: List[List[float]], 
                         metadata: List[Dict[str, Any]] = None, ids: List[str] = None) -> Dict[str, Any]:
        """Add vectors to a collection"""
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_data = self.collections[collection_name]
            store = collection_data["store"]
            store_config = self.vector_stores[store]
            
            # Validate vectors
            if not vectors:
                raise ValueError("No vectors provided")
            
            expected_dim = collection_data["dimension"]
            for i, vector in enumerate(vectors):
                if len(vector) != expected_dim:
                    raise ValueError(f"Vector {i} has dimension {len(vector)}, expected {expected_dim}")
            
            # Add vectors based on store type
            if store == "chromadb":
                result = await self._add_chromadb_vectors(collection_name, vectors, metadata, ids)
            elif store == "qdrant":
                result = await self._add_qdrant_vectors(collection_name, vectors, metadata, ids)
            elif store == "faiss":
                result = await self._add_faiss_vectors(collection_name, vectors, metadata, ids)
            
            # Update collection stats
            collection_data["document_count"] += len(vectors)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to add vectors to collection {collection_name}: {e}")
            raise
    
    async def _add_chromadb_vectors(self, collection_name: str, vectors: List[List[float]], 
                                   metadata: List[Dict[str, Any]] = None, ids: List[str] = None):
        """Add vectors to ChromaDB"""
        request_data = {
            "embeddings": vectors,
            "metadatas": metadata or [{}] * len(vectors),
            "ids": ids or [f"doc_{i}" for i in range(len(vectors))]
        }
        
        response = await self.http_client.post(
            f"{self.vector_stores['chromadb']['url']}/api/v1/collections/{collection_name}/add",
            json=request_data
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to add vectors to ChromaDB: {response.status_code}")
        
        return {"added_count": len(vectors)}
    
    async def _add_qdrant_vectors(self, collection_name: str, vectors: List[List[float]], 
                                 metadata: List[Dict[str, Any]] = None, ids: List[str] = None):
        """Add vectors to Qdrant"""
        points = []
        for i, vector in enumerate(vectors):
            point = {
                "id": ids[i] if ids else f"doc_{i}",
                "vector": vector,
                "payload": metadata[i] if metadata else {}
            }
            points.append(point)
        
        request_data = {"points": points}
        
        response = await self.http_client.put(
            f"{self.vector_stores['qdrant']['url']}/collections/{collection_name}/points",
            json=request_data
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to add vectors to Qdrant: {response.status_code}")
        
        return {"added_count": len(vectors)}
    
    async def _add_faiss_vectors(self, collection_name: str, vectors: List[List[float]], 
                                metadata: List[Dict[str, Any]] = None, ids: List[str] = None):
        """Add vectors to FAISS"""
        request_data = {
            "vectors": vectors,
            "metadata": metadata or [{}] * len(vectors),
            "ids": ids or [f"doc_{i}" for i in range(len(vectors))]
        }
        
        response = await self.http_client.post(
            f"{self.vector_stores['faiss']['url']}/collections/{collection_name}/add",
            json=request_data
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to add vectors to FAISS: {response.status_code}")
        
        return {"added_count": len(vectors)}
    
    async def search_vectors(self, collection_name: str, query_vector: List[float], 
                           k: int = 10, filter_condition: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_data = self.collections[collection_name]
            store = collection_data["store"]
            
            # Validate query vector
            expected_dim = collection_data["dimension"]
            if len(query_vector) != expected_dim:
                raise ValueError(f"Query vector has dimension {len(query_vector)}, expected {expected_dim}")
            
            # Search based on store type
            if store == "chromadb":
                return await self._search_chromadb(collection_name, query_vector, k, filter_condition)
            elif store == "qdrant":
                return await self._search_qdrant(collection_name, query_vector, k, filter_condition)
            elif store == "faiss":
                return await self._search_faiss(collection_name, query_vector, k, filter_condition)
            
        except Exception as e:
            logger.error(f"Failed to search vectors in collection {collection_name}: {e}")
            raise
    
    async def _search_chromadb(self, collection_name: str, query_vector: List[float], 
                              k: int, filter_condition: Dict[str, Any] = None):
        """Search ChromaDB"""
        request_data = {
            "query_embeddings": [query_vector],
            "n_results": k,
            "where": filter_condition or {}
        }
        
        response = await self.http_client.post(
            f"{self.vector_stores['chromadb']['url']}/api/v1/collections/{collection_name}/query",
            json=request_data
        )
        
        if response.status_code == 200:
            data = response.json()
            results = []
            for i, doc_id in enumerate(data["ids"][0]):
                results.append({
                    "id": doc_id,
                    "score": data["distances"][0][i],
                    "metadata": data.get("metadatas", [[{}]])[0][i]
                })
            return results
        else:
            raise Exception(f"ChromaDB search failed: {response.status_code}")
    
    async def _search_qdrant(self, collection_name: str, query_vector: List[float], 
                            k: int, filter_condition: Dict[str, Any] = None):
        """Search Qdrant"""
        request_data = {
            "vector": query_vector,
            "limit": k,
            "with_payload": True
        }
        
        if filter_condition:
            request_data["filter"] = filter_condition
        
        response = await self.http_client.post(
            f"{self.vector_stores['qdrant']['url']}/collections/{collection_name}/points/search",
            json=request_data
        )
        
        if response.status_code == 200:
            data = response.json()
            results = []
            for result in data["result"]:
                results.append({
                    "id": str(result["id"]),
                    "score": result["score"],
                    "metadata": result.get("payload", {})
                })
            return results
        else:
            raise Exception(f"Qdrant search failed: {response.status_code}")
    
    async def _search_faiss(self, collection_name: str, query_vector: List[float], 
                           k: int, filter_condition: Dict[str, Any] = None):
        """Search FAISS"""
        request_data = {
            "query_vector": query_vector,
            "k": k,
            "filter": filter_condition or {}
        }
        
        response = await self.http_client.post(
            f"{self.vector_stores['faiss']['url']}/collections/{collection_name}/search",
            json=request_data
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        else:
            raise Exception(f"FAISS search failed: {response.status_code}")
    
    async def get_collections(self) -> List[Dict[str, Any]]:
        """Get all collections"""
        return list(self.collections.values())
    
    async def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get collection statistics"""
        if collection_name not in self.collections:
            return None
        
        collection_data = self.collections[collection_name]
        store = collection_data["store"]
        
        try:
            if store == "chromadb":
                stats = await self._get_chromadb_stats(collection_name)
            elif store == "qdrant":
                stats = await self._get_qdrant_stats(collection_name)
            elif store == "faiss":
                stats = await self._get_faiss_stats(collection_name)
            
            collection_data.update(stats)
            return collection_data
            
        except Exception as e:
            logger.warning(f"Could not get stats for collection {collection_name}: {e}")
            return collection_data
    
    async def _get_chromadb_stats(self, collection_name: str):
        """Get ChromaDB collection stats"""
        response = await self.http_client.get(
            f"{self.vector_stores['chromadb']['url']}/api/v1/collections/{collection_name}"
        )
        
        if response.status_code == 200:
            return response.json()
        return {}
    
    async def _get_qdrant_stats(self, collection_name: str):
        """Get Qdrant collection stats"""
        response = await self.http_client.get(
            f"{self.vector_stores['qdrant']['url']}/collections/{collection_name}"
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "document_count": data.get("result", {}).get("points_count", 0),
                "status": data.get("result", {}).get("status", "unknown")
            }
        return {}
    
    async def _get_faiss_stats(self, collection_name: str):
        """Get FAISS collection stats"""
        response = await self.http_client.get(
            f"{self.vector_stores['faiss']['url']}/collections/{collection_name}/stats"
        )
        
        if response.status_code == 200:
            return response.json()
        return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for vector store manager"""
        try:
            health_data = {
                "status": "healthy",
                "collections": len(self.collections),
                "vector_stores": list(self.vector_stores.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check store health
            store_health = {}
            for store_name, config in self.vector_stores.items():
                try:
                    response = await self.http_client.get(f"{config['url']}{config['health_endpoint']}")
                    store_health[store_name] = {
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                        "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0
                    }
                except Exception as e:
                    store_health[store_name] = {
                        "status": "unreachable",
                        "error": str(e)
                    }
            
            health_data["store_health"] = store_health
            
            return health_data
            
        except Exception as e:
            logger.error(f"Vector store manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }