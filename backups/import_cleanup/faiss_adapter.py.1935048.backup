"""
FAISS adapter for high-performance vector similarity search
"""
import faiss
import numpy as np
import pickle
import os
from typing import Dict, Any, List, Optional, Tuple
from ..base_adapter import ServiceAdapter
import logging

logger = logging.getLogger(__name__)


class FAISSAdapter(ServiceAdapter):
    """Adapter for FAISS vector index"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FAISS", config)
        self.index_path = config.get('index_path', '/data/faiss/index.faiss')
        self.metadata_path = config.get('metadata_path', '/data/faiss/metadata.pkl')
        self.dimension = config.get('dimension', 1536)
        self.index_type = config.get('index_type', 'Flat')  # Flat, IVF, HNSW
        self.index = None
        self.metadata = []
        
    async def initialize(self):
        """Initialize FAISS index"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Load existing index or create new one
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'rb') as f:
                        self.metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            else:
                # Create new index based on type
                if self.index_type == 'Flat':
                    self.index = faiss.IndexFlatL2(self.dimension)
                elif self.index_type == 'IVF':
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                elif self.index_type == 'HNSW':
                    self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                else:
                    raise ValueError(f"Unknown index type: {self.index_type}")
                    
                logger.info(f"Created new FAISS {self.index_type} index")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {str(e)}")
            raise
            
    async def _custom_health_check(self) -> bool:
        """Check FAISS health"""
        return self.index is not None
        
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get FAISS capabilities"""
        return {
            'service': 'FAISS',
            'type': 'vector_index',
            'features': [
                'high_performance_search',
                'batch_operations',
                'gpu_acceleration',
                'multiple_index_types',
                'billion_scale'
            ],
            'dimension': self.dimension,
            'index_type': self.index_type,
            'current_size': self.index.ntotal if self.index else 0,
            'gpu_available': faiss.get_num_gpus() > 0
        }
        
    async def add_vectors(self, 
                         vectors: np.ndarray,
                         metadata: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Add vectors to FAISS index"""
        try:
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors, dtype=np.float32)
            else:
                vectors = vectors.astype(np.float32)
                
            # Ensure 2D array
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
                
            # Add to index
            start_id = self.index.ntotal
            self.index.add(vectors)
            
            # Store metadata
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{'id': start_id + i} for i in range(len(vectors))])
                
            # Save index and metadata
            await self._save_index()
            
            return {
                'success': True,
                'added': len(vectors),
                'total_vectors': self.index.ntotal
            }
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def search_vectors(self, 
                           query_vectors: np.ndarray,
                           k: int = 10) -> Dict[str, Any]:
        """Search similar vectors"""
        try:
            if not isinstance(query_vectors, np.ndarray):
                query_vectors = np.array(query_vectors, dtype=np.float32)
            else:
                query_vectors = query_vectors.astype(np.float32)
                
            # Ensure 2D array
            if query_vectors.ndim == 1:
                query_vectors = query_vectors.reshape(1, -1)
                
            # Search
            distances, indices = self.index.search(query_vectors, k)
            
            # Gather results with metadata
            results = []
            for i in range(len(query_vectors)):
                query_results = []
                for j in range(k):
                    idx = indices[i][j]
                    if idx >= 0 and idx < len(self.metadata):
                        query_results.append({
                            'index': int(idx),
                            'distance': float(distances[i][j]),
                            'metadata': self.metadata[idx]
                        })
                results.append(query_results)
                
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _save_index(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            
    async def train_index(self, training_vectors: np.ndarray):
        """Train IVF index if needed"""
        if hasattr(self.index, 'train') and not self.index.is_trained:
            logger.info("Training FAISS index...")
            training_vectors = training_vectors.astype(np.float32)
            self.index.train(training_vectors)
            await self._save_index()
            logger.info("FAISS index training complete")
            
    async def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            'metadata_count': len(self.metadata)
        }