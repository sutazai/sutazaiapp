"""
ChromaDB adapter for vector storage and retrieval
"""
import chromadb
from chromadb.config import Settings
from typing import Dict, Any, List, Optional
import numpy as np
from ..base_adapter import ServiceAdapter
import logging

logger = logging.getLogger(__name__)


class ChromaDBAdapter(ServiceAdapter):
    """Adapter for ChromaDB vector database"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ChromaDB", config)
        self.persist_directory = config.get('persist_directory', '/data/chromadb')
        self.collection_name = config.get('collection_name', 'sutazai_vectors')
        self.client = None
        self.collection = None
        
    async def initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            settings = Settings(
                chroma_server_host=self.config.get('host', 'chromadb'),
                chroma_server_http_port=self.config.get('port', 8000),
                chroma_server_ssl_enabled=self.config.get('ssl_enabled', False)
            )
            
            self.client = chromadb.HttpClient(
                host=settings.chroma_server_host,
                port=settings.chroma_server_http_port,
                ssl=settings.chroma_server_ssl_enabled
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "SutazAI vector storage"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
            
    async def _custom_health_check(self) -> bool:
        """Check ChromaDB health"""
        try:
            if self.client:
                # Try to list collections as health check
                collections = self.client.list_collections()
                return True
        except:
            return False
            
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get ChromaDB capabilities"""
        return {
            'service': 'ChromaDB',
            'type': 'vector_database',
            'features': [
                'vector_storage',
                'similarity_search',
                'metadata_filtering',
                'persistent_storage',
                'multi_collection'
            ],
            'max_dimensions': 1536,  # OpenAI embedding dimension
            'distance_metrics': ['cosine', 'l2', 'ip'],
            'collection': self.collection_name if self.collection else None
        }
        
    async def add_vectors(self, 
                         embeddings: List[List[float]], 
                         documents: List[str],
                         metadatas: Optional[List[Dict]] = None,
                         ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Add vectors to ChromaDB"""
        try:
            if not ids:
                ids = [f"doc_{i}" for i in range(len(documents))]
                
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas or [{} for _ in documents],
                ids=ids
            )
            
            return {
                'success': True,
                'count': len(documents),
                'collection': self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def search_vectors(self, 
                           query_embeddings: List[List[float]], 
                           n_results: int = 10,
                           where: Optional[Dict] = None) -> Dict[str, Any]:
        """Search similar vectors"""
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )
            
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
            
    async def delete_vectors(self, ids: List[str]) -> Dict[str, Any]:
        """Delete vectors by IDs"""
        try:
            self.collection.delete(ids=ids)
            return {
                'success': True,
                'deleted': len(ids)
            }
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            count = self.collection.count()
            return {
                'collection': self.collection_name,
                'count': count,
                'metadata': self.collection.metadata
            }
        except Exception as e:
            return {
                'error': str(e)
            }