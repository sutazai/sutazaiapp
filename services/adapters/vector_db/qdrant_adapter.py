"""
Qdrant adapter for vector database operations
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import Dict, Any, List, Optional
import uuid
from ..base_adapter import ServiceAdapter
import logging

logger = logging.getLogger(__name__)


class QdrantAdapter(ServiceAdapter):
    """Adapter for Qdrant vector database"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Qdrant", config)
        self.host = config.get('host', 'qdrant')
        self.port = config.get('port', 6333)
        self.collection_name = config.get('collection_name', 'sutazai_vectors')
        self.vector_size = config.get('vector_size', 1536)
        self.distance = config.get('distance', 'Cosine')
        self.client = None
        
    async def initialize(self):
        """Initialize Qdrant client and collection"""
        try:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                timeout=self.timeout
            )
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=getattr(Distance, self.distance)
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {str(e)}")
            raise
            
    async def _custom_health_check(self) -> bool:
        """Check Qdrant health"""
        try:
            if self.client:
                # Try to get collection info
                self.client.get_collection(self.collection_name)
                return True
        except (ConnectionError, TimeoutError, Exception) as e:
            logger.warning(f"Exception caught, returning: {e}")
            return False
            
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get Qdrant capabilities"""
        collection_info = None
        if self.client:
            try:
                collection_info = self.client.get_collection(self.collection_name)
            except (ConnectionError, TimeoutError, Exception) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
                
        return {
            'service': 'Qdrant',
            'type': 'vector_database',
            'features': [
                'vector_storage',
                'similarity_search',
                'filtering',
                'payload_storage',
                'batch_operations',
                'snapshots',
                'clustering'
            ],
            'vector_size': self.vector_size,
            'distance_metric': self.distance,
            'collection': self.collection_name,
            'points_count': collection_info.points_count if collection_info else 0
        }
        
    async def add_vectors(self,
                         vectors: List[List[float]],
                         payloads: Optional[List[Dict]] = None,
                         ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Add vectors to Qdrant"""
        try:
            if not ids:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
                
            points = []
            for i, (vector, id_) in enumerate(zip(vectors, ids)):
                payload = payloads[i] if payloads else {}
                points.append(
                    PointStruct(
                        id=id_,
                        vector=vector,
                        payload=payload
                    )
                )
                
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
            return {
                'success': True,
                'added': len(points),
                'ids': ids
            }
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def search_vectors(self,
                           query_vector: List[float],
                           limit: int = 10,
                           filter_conditions: Optional[Dict] = None) -> Dict[str, Any]:
        """Search similar vectors"""
        try:
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)
                
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    'id': point.id,
                    'score': point.score,
                    'payload': point.payload
                })
                
            return {
                'success': True,
                'results': results,
                'count': len(results)
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
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            
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
            
    async def update_payload(self, point_id: str, payload: Dict) -> Dict[str, Any]:
        """Update point payload"""
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[point_id]
            )
            
            return {
                'success': True,
                'updated': point_id
            }
            
        except Exception as e:
            logger.error(f"Failed to update payload: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'collection': self.collection_name,
                'points_count': info.points_count,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'config': {
                    'vector_size': info.config.params.vectors.size,
                    'distance': str(info.config.params.vectors.distance)
                }
            }
        except Exception as e:
            return {
                'error': str(e)
            }
            
    async def create_snapshot(self) -> Dict[str, Any]:
        """Create collection snapshot"""
        try:
            snapshot = self.client.create_snapshot(self.collection_name)
            return {
                'success': True,
                'snapshot': snapshot
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }