"""
Vector Database Manager for SutazAI
Manages ChromaDB, Qdrant, and FAISS for optimal performance
"""
import httpx
import asyncio
import logging
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class VectorDBManager:
    """Unified manager for vector databases"""
    
    def __init__(self, 
                 chromadb_url: str = "http://chromadb:8000",
                 qdrant_url: str = "http://qdrant:6333",
                 chromadb_token: str = "sk-dcebf71d6136dafc1405f3d3b6f7a9ce43723e36f93542fb"):
        self.chromadb_url = chromadb_url
        self.qdrant_url = qdrant_url
        self.chromadb_token = chromadb_token
        
        # Set up authenticated client for ChromaDB
        self.chromadb_headers = {
            "X-Chroma-Token": self.chromadb_token,
            "Content-Type": "application/json"
        }
        
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Collection configurations
        self.collections = {
            "documents": {
                "chromadb": "sutazai_documents",
                "qdrant": "sutazai_documents",
                "dimension": 384,  # for nomic-embed-text
                "distance": "cosine"
            },
            "code": {
                "chromadb": "sutazai_code",
                "qdrant": "sutazai_code", 
                "dimension": 384,
                "distance": "cosine"
            },
            "knowledge": {
                "chromadb": "sutazai_knowledge",
                "qdrant": "sutazai_knowledge",
                "dimension": 384,
                "distance": "cosine"
            }
        }
    
    async def initialize_collections(self) -> Dict[str, Any]:
        """Initialize all required collections in both databases"""
        results = {}
        
        # Initialize ChromaDB collections
        for name, config in self.collections.items():
            try:
                # Create ChromaDB collection
                chroma_result = await self._create_chromadb_collection(
                    config["chromadb"],
                    config["dimension"]
                )
                
                # Create Qdrant collection
                qdrant_result = await self._create_qdrant_collection(
                    config["qdrant"],
                    config["dimension"],
                    config["distance"]
                )
                
                results[name] = {
                    "chromadb": chroma_result,
                    "qdrant": qdrant_result
                }
            except Exception as e:
                logger.error(f"Failed to initialize collection {name}: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    async def _create_chromadb_collection(self, name: str, dimension: int) -> Dict[str, Any]:
        """Create a ChromaDB collection using v2 API"""
        try:
            # First try to get the collection using v2 API  
            response = await self.client.get(
                f"{self.chromadb_url}/api/v2/collections/{name}",
                headers=self.chromadb_headers
            )
            if response.status_code == 200:
                return {"status": "exists", "name": name}
            
            # Create if doesn't exist using v2 API
            response = await self.client.post(
                f"{self.chromadb_url}/api/v2/collections",
                headers=self.chromadb_headers,
                json={
                    "name": name,
                    "metadata": {
                        "dimension": dimension,
                        "created_at": datetime.utcnow().isoformat()
                    }
                }
            )
            
            if response.status_code in [200, 201]:
                return {"status": "created", "name": name}
            else:
                return {"status": "error", "message": response.text}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _create_qdrant_collection(self, name: str, dimension: int, distance: str) -> Dict[str, Any]:
        """Create a Qdrant collection"""
        try:
            # Check if collection exists
            response = await self.client.get(
                f"{self.qdrant_url}/collections/{name}"
            )
            if response.status_code == 200:
                return {"status": "exists", "name": name}
            
            # Create collection
            response = await self.client.put(
                f"{self.qdrant_url}/collections/{name}",
                json={
                    "vectors": {
                        "size": dimension,
                        "distance": distance.capitalize()
                    },
                    "optimizers_config": {
                        "default_segment_number": 2
                    },
                    "replication_factor": 1
                }
            )
            
            if response.status_code == 200:
                return {"status": "created", "name": name}
            else:
                return {"status": "error", "message": response.text}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def add_documents(self, 
                          collection: str,
                          documents: List[Dict[str, Any]],
                          embeddings: List[List[float]],
                          use_db: str = "both") -> Dict[str, Any]:
        """Add documents to vector databases"""
        results = {}
        
        if collection not in self.collections:
            return {"error": f"Unknown collection: {collection}"}
        
        config = self.collections[collection]
        
        # Add to ChromaDB
        if use_db in ["chromadb", "both"]:
            try:
                chroma_result = await self._add_to_chromadb(
                    config["chromadb"],
                    documents,
                    embeddings
                )
                results["chromadb"] = chroma_result
            except Exception as e:
                results["chromadb"] = {"error": str(e)}
        
        # Add to Qdrant
        if use_db in ["qdrant", "both"]:
            try:
                qdrant_result = await self._add_to_qdrant(
                    config["qdrant"],
                    documents,
                    embeddings
                )
                results["qdrant"] = qdrant_result
            except Exception as e:
                results["qdrant"] = {"error": str(e)}
        
        return results
    
    async def _add_to_chromadb(self,
                              collection: str,
                              documents: List[Dict[str, Any]],
                              embeddings: List[List[float]]) -> Dict[str, Any]:
        """Add documents to ChromaDB"""
        ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        texts = [doc.get("text", "") for doc in documents]
        
        response = await self.client.post(
            f"{self.chromadb_url}/api/v2/collections/{collection}/add",
            headers=self.chromadb_headers,
            json={
                "ids": ids,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "documents": texts
            }
        )
        
        if response.status_code in [200, 201]:
            return {"status": "success", "count": len(documents)}
        else:
            return {"status": "error", "message": response.text}
    
    async def _add_to_qdrant(self,
                           collection: str,
                           documents: List[Dict[str, Any]],
                           embeddings: List[List[float]]) -> Dict[str, Any]:
        """Add documents to Qdrant"""
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            points.append({
                "id": doc.get("id", i),
                "vector": embedding,
                "payload": doc.get("metadata", {})
            })
        
        response = await self.client.put(
            f"{self.qdrant_url}/collections/{collection}/points",
            json={"points": points}
        )
        
        if response.status_code == 200:
            return {"status": "success", "count": len(documents)}
        else:
            return {"status": "error", "message": response.text}
    
    async def search(self,
                    collection: str,
                    query_embedding: List[float],
                    limit: int = 10,
                    use_db: str = "qdrant") -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if collection not in self.collections:
            return []
        
        config = self.collections[collection]
        
        if use_db == "chromadb":
            return await self._search_chromadb(
                config["chromadb"],
                query_embedding,
                limit
            )
        elif use_db == "qdrant":
            return await self._search_qdrant(
                config["qdrant"],
                query_embedding,
                limit
            )
        else:
            # Search both and merge results
            chroma_results = await self._search_chromadb(
                config["chromadb"],
                query_embedding,
                limit
            )
            qdrant_results = await self._search_qdrant(
                config["qdrant"],
                query_embedding,
                limit
            )
            
            # Merge and deduplicate
            all_results = chroma_results + qdrant_results
            seen_ids = set()
            merged = []
            for result in sorted(all_results, key=lambda x: x.get("score", 0), reverse=True):
                if result["id"] not in seen_ids:
                    seen_ids.add(result["id"])
                    merged.append(result)
                    if len(merged) >= limit:
                        break
            
            return merged
    
    async def _search_chromadb(self,
                             collection: str,
                             query_embedding: List[float],
                             limit: int) -> List[Dict[str, Any]]:
        """Search in ChromaDB"""
        try:
            response = await self.client.post(
                f"{self.chromadb_url}/api/v2/collections/{collection}/query",
                headers=self.chromadb_headers,
                json={
                    "query_embeddings": [query_embedding],
                    "n_results": limit
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                for i, doc_id in enumerate(data.get("ids", [[]])[0]):
                    results.append({
                        "id": doc_id,
                        "score": 1 - data.get("distances", [[]])[0][i],  # Convert distance to similarity
                        "metadata": data.get("metadatas", [[]])[0][i],
                        "text": data.get("documents", [[]])[0][i]
                    })
                return results
            else:
                logger.error(f"ChromaDB search failed: {response.text}")
                return []
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            return []
    
    async def _search_qdrant(self,
                           collection: str,
                           query_embedding: List[float],
                           limit: int) -> List[Dict[str, Any]]:
        """Search in Qdrant"""
        try:
            response = await self.client.post(
                f"{self.qdrant_url}/collections/{collection}/points/search",
                json={
                    "vector": query_embedding,
                    "limit": limit,
                    "with_payload": True
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                for point in data.get("result", []):
                    results.append({
                        "id": point["id"],
                        "score": point["score"],
                        "metadata": point.get("payload", {}),
                        "text": point.get("payload", {}).get("text", "")
                    })
                return results
            else:
                logger.error(f"Qdrant search failed: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []
    
    async def optimize_collections(self) -> Dict[str, Any]:
        """Optimize all collections for better performance"""
        results = {}
        
        # Optimize Qdrant collections
        for name, config in self.collections.items():
            try:
                # Create index for faster search
                response = await self.client.post(
                    f"{self.qdrant_url}/collections/{config['qdrant']}/index",
                    json={
                        "field_name": "all",
                        "field_schema": "keyword"
                    }
                )
                
                # Optimize segments
                optimize_response = await self.client.post(
                    f"{self.qdrant_url}/collections/{config['qdrant']}/points/vacuum"
                )
                
                results[name] = {
                    "qdrant": {
                        "index": response.status_code == 200,
                        "optimize": optimize_response.status_code == 200
                    }
                }
            except Exception as e:
                results[name] = {"error": str(e)}
        
        return results
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        
        for name, config in self.collections.items():
            try:
                # Get ChromaDB stats
                chroma_response = await self.client.get(
                    f"{self.chromadb_url}/api/v2/collections/{config['chromadb']}",
                    headers=self.chromadb_headers
                )
                
                # Get Qdrant stats
                qdrant_response = await self.client.get(
                    f"{self.qdrant_url}/collections/{config['qdrant']}"
                )
                
                stats[name] = {
                    "chromadb": chroma_response.json() if chroma_response.status_code == 200 else {},
                    "qdrant": qdrant_response.json() if qdrant_response.status_code == 200 else {}
                }
            except Exception as e:
                stats[name] = {"error": str(e)}
        
        return stats
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()

# Singleton instance
vector_db_manager = VectorDBManager()