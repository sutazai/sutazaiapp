"""
FAISS Integration for High-Performance Vector Search
"""
import numpy as np
import pickle
import os
from typing import List, Dict, Optional, Tuple, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

class FAISSManager:
    """High-performance vector search using FAISS"""
    
    def __init__(self, index_dir: str = "/app/data/faiss_indexes"):
        self.index_dir = index_dir
        self.indexes: Dict[str, Any] = {}
        self.metadata: Dict[str, List[Dict[str, Any]]] = {}
        self.dimension = 384  # Default for nomic-embed-text
        
        # Create index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using numpy fallback")
    
    def create_index(self, name: str, dimension: int = 384, index_type: str = "IVF") -> bool:
        """Create a new FAISS index"""
        if not FAISS_AVAILABLE:
            # Fallback: store vectors in numpy arrays
            self.indexes[name] = {
                "vectors": np.empty((0, dimension), dtype=np.float32),
                "dimension": dimension,
                "count": 0
            }
            self.metadata[name] = []
            return True
        
        try:
            if index_type == "Flat":
                # Exact search - best accuracy, slower for large datasets
                index = faiss.IndexFlatL2(dimension)
            elif index_type == "IVF":
                # Inverted file index - good balance of speed and accuracy
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
                # Train with dummy data
                dummy_data = np.random.random((1000, dimension)).astype('float32')
                index.train(dummy_data)
            elif index_type == "HNSW":
                # Hierarchical Navigable Small World - very fast, good accuracy
                index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                # Default to IVF
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
                dummy_data = np.random.random((1000, dimension)).astype('float32')
                index.train(dummy_data)
            
            self.indexes[name] = index
            self.metadata[name] = []
            
            # Save index
            self._save_index(name)
            
            logger.info(f"Created FAISS index '{name}' with type '{index_type}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return False
    
    def add_vectors(self, index_name: str, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to an index"""
        if index_name not in self.indexes:
            logger.error(f"Index '{index_name}' not found")
            return False
        
        if not FAISS_AVAILABLE:
            # Fallback: concatenate arrays
            index = self.indexes[index_name]
            index["vectors"] = np.vstack([index["vectors"], vectors])
            index["count"] += len(vectors)
            self.metadata[index_name].extend(metadata)
            return True
        
        try:
            index = self.indexes[index_name]
            
            # Ensure vectors are float32
            vectors = vectors.astype('float32')
            
            # Add to index
            index.add(vectors)
            
            # Store metadata
            self.metadata[index_name].extend(metadata)
            
            # Save updated index
            self._save_index(index_name)
            
            logger.info(f"Added {len(vectors)} vectors to index '{index_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False
    
    def search(self, index_name: str, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for k nearest neighbors"""
        if index_name not in self.indexes:
            logger.error(f"Index '{index_name}' not found")
            return []
        
        if not FAISS_AVAILABLE:
            # Fallback: brute force search with numpy
            index = self.indexes[index_name]
            if index["count"] == 0:
                return []
            
            # Compute distances
            query_vector = query_vector.reshape(1, -1)
            distances = np.linalg.norm(index["vectors"] - query_vector, axis=1)
            
            # Get top k
            k = min(k, len(distances))
            indices = np.argpartition(distances, k-1)[:k]
            indices = indices[np.argsort(distances[indices])]
            
            results = []
            for i, idx in enumerate(indices):
                results.append({
                    "id": idx,
                    "score": float(1 / (1 + distances[idx])),  # Convert distance to similarity
                    "metadata": self.metadata[index_name][idx]
                })
            
            return results
        
        try:
            index = self.indexes[index_name]
            query_vector = query_vector.reshape(1, -1).astype('float32')
            
            # Search
            distances, indices = index.search(query_vector, k)
            
            # Format results
            results = []
            for i in range(len(indices[0])):
                if indices[0][i] != -1:  # Valid result
                    results.append({
                        "id": int(indices[0][i]),
                        "score": float(1 / (1 + distances[0][i])),  # Convert distance to similarity
                        "metadata": self.metadata[index_name][indices[0][i]]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []
    
    def _save_index(self, index_name: str):
        """Save index to disk"""
        if not FAISS_AVAILABLE:
            # Save numpy arrays
            np.save(os.path.join(self.index_dir, f"{index_name}_vectors.npy"), 
                   self.indexes[index_name]["vectors"])
        else:
            faiss.write_index(self.indexes[index_name], 
                            os.path.join(self.index_dir, f"{index_name}.index"))
        
        # Save metadata
        with open(os.path.join(self.index_dir, f"{index_name}_metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata[index_name], f)
    
    def load_index(self, index_name: str) -> bool:
        """Load index from disk"""
        try:
            if not FAISS_AVAILABLE:
                # Load numpy arrays
                vectors_path = os.path.join(self.index_dir, f"{index_name}_vectors.npy")
                if os.path.exists(vectors_path):
                    vectors = np.load(vectors_path)
                    self.indexes[index_name] = {
                        "vectors": vectors,
                        "dimension": vectors.shape[1],
                        "count": len(vectors)
                    }
            else:
                index_path = os.path.join(self.index_dir, f"{index_name}.index")
                if os.path.exists(index_path):
                    self.indexes[index_name] = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = os.path.join(self.index_dir, f"{index_name}_metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata[index_name] = pickle.load(f)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for an index"""
        if index_name not in self.indexes:
            return {"error": "Index not found"}
        
        if not FAISS_AVAILABLE:
            index = self.indexes[index_name]
            return {
                "name": index_name,
                "dimension": index["dimension"],
                "count": index["count"],
                "type": "numpy",
                "memory_usage": index["vectors"].nbytes
            }
        
        index = self.indexes[index_name]
        return {
            "name": index_name,
            "dimension": index.d,
            "count": index.ntotal,
            "is_trained": getattr(index, 'is_trained', True),
            "metric": "L2"
        }
    
    def optimize_index(self, index_name: str) -> bool:
        """Optimize index for better performance"""
        if not FAISS_AVAILABLE or index_name not in self.indexes:
            return False
        
        try:
            index = self.indexes[index_name]
            
            # For IVF indexes, we can adjust nprobe for speed/accuracy tradeoff
            if hasattr(index, 'nprobe'):
                index.nprobe = 10  # Default is 1, higher = more accurate but slower
            
            logger.info(f"Optimized index '{index_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize index: {e}")
            return False

# Singleton instance
faiss_manager = FAISSManager()