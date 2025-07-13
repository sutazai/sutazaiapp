"""
Vector Database Service - Simple vector storage with optional FAISS integration
Handles embeddings, similarity search, and knowledge retrieval
"""

import asyncio
import logging
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
import time

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorDatabaseService:
    def __init__(self, persist_directory: str = "/opt/sutazaiapp/data/vectordb"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Simple document storage
        self.collections = {}
        self.documents = {}
        self.faiss_indexes = {}
        
        # Initialize collections
        self._initialize_collections()
        
    def _initialize_collections(self):
        """Initialize default collections"""
        default_collections = [
            "documents",
            "code_snippets", 
            "conversations",
            "knowledge_base",
            "agent_memory"
        ]
        
        for collection_name in default_collections:
            try:
                collection_file = self.persist_directory / f"{collection_name}.json"
                if collection_file.exists():
                    with open(collection_file, 'r') as f:
                        self.collections[collection_name] = json.load(f)
                else:
                    self.collections[collection_name] = {
                        "metadata": {"description": f"Collection for {collection_name}"},
                        "documents": []
                    }
                    self._save_collection(collection_name)
                    
                logger.info(f"✅ Initialized collection: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to initialize collection {collection_name}: {e}")
                self.collections[collection_name] = {
                    "metadata": {"description": f"Collection for {collection_name}"},
                    "documents": []
                }
    
    def _save_collection(self, collection_name: str):
        """Save collection to disk"""
        try:
            collection_file = self.persist_directory / f"{collection_name}.json"
            with open(collection_file, 'w') as f:
                json.dump(self.collections[collection_name], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save collection {collection_name}: {e}")
    
    async def add_documents(
        self, 
        collection_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add documents to a collection"""
        try:
            if collection_name not in self.collections:
                self.collections[collection_name] = {
                    "metadata": {"description": f"Collection for {collection_name}"},
                    "documents": []
                }
            
            # Generate IDs if not provided
            if not ids:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            if not metadatas:
                metadatas = [{}] * len(documents)
            
            # Add documents
            for i, doc in enumerate(documents):
                doc_entry = {
                    "id": ids[i],
                    "content": doc,
                    "metadata": metadatas[i],
                    "timestamp": time.time()
                }
                self.collections[collection_name]["documents"].append(doc_entry)
            
            # Save to disk
            self._save_collection(collection_name)
            
            logger.info(f"✅ Added {len(documents)} documents to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to {collection_name}: {e}")
            return False
    
    async def search_similar(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents (simple text matching)"""
        try:
            if collection_name not in self.collections:
                logger.warning(f"Collection {collection_name} not found")
                return []
            
            documents = self.collections[collection_name]["documents"]
            query_lower = query.lower()
            
            # Simple text similarity scoring
            scored_docs = []
            for doc in documents:
                content = doc["content"].lower()
                
                # Check metadata filters
                if where:
                    match = True
                    for key, value in where.items():
                        if key not in doc["metadata"] or doc["metadata"][key] != value:
                            match = False
                            break
                    if not match:
                        continue
                
                # Simple scoring based on keyword matches
                score = 0
                query_words = query_lower.split()
                for word in query_words:
                    if word in content:
                        score += content.count(word)
                
                if score > 0:
                    scored_docs.append({
                        "document": doc["content"],
                        "metadata": doc["metadata"],
                        "id": doc["id"],
                        "score": score,
                        "timestamp": doc["timestamp"]
                    })
            
            # Sort by score and limit results
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            results = scored_docs[:n_results]
            
            logger.info(f"🔍 Found {len(results)} similar documents in {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search in {collection_name}: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        try:
            stats = {}
            for collection_name, collection_data in self.collections.items():
                docs = collection_data.get("documents", [])
                stats[collection_name] = {
                    "document_count": len(docs),
                    "total_characters": sum(len(doc.get("content", "")) for doc in docs),
                    "last_updated": max([doc.get("timestamp", 0) for doc in docs]) if docs else 0
                }
            
            return {
                "collections": stats,
                "total_collections": len(self.collections),
                "features": {
                    "numpy_available": NUMPY_AVAILABLE,
                    "faiss_available": FAISS_AVAILABLE
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            if collection_name in self.collections:
                del self.collections[collection_name]
                
                # Remove file
                collection_file = self.persist_directory / f"{collection_name}.json"
                if collection_file.exists():
                    collection_file.unlink()
                
                logger.info(f"🗑️ Deleted collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection {collection_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            total_docs = sum(len(col.get("documents", [])) for col in self.collections.values())
            
            return {
                "status": "healthy",
                "collections": len(self.collections),
                "total_documents": total_docs,
                "persist_directory": str(self.persist_directory),
                "features": {
                    "numpy_available": NUMPY_AVAILABLE,
                    "faiss_available": FAISS_AVAILABLE
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Factory function
def create_vector_database_service(config: Dict[str, Any] = None) -> VectorDatabaseService:
    """Factory function to create vector database service"""
    persist_dir = config.get("persist_directory", "/opt/sutazaiapp/data/vectordb") if config else "/opt/sutazaiapp/data/vectordb"
    return VectorDatabaseService(persist_dir)