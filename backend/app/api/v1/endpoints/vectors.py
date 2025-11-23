"""Vector database operations endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Annotated, Optional
from pydantic import BaseModel, Field
from app.api.dependencies.auth import get_current_active_user
from app.models.user import User
from app.services.connections import service_connections
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class VectorData(BaseModel):
    id: str
    vector: List[float] = Field(..., min_length=1, max_length=4096)
    metadata: Dict[str, Any] = {}


class VectorSearchRequest(BaseModel):
    query: List[float] = Field(..., min_length=1, max_length=4096)
    k: int = Field(5, ge=1, le=100)
    filter: Optional[Dict[str, Any]] = None


@router.post("/store")
async def store_vector(
    data: VectorData,
    current_user: Annotated[User, Depends(get_current_active_user)],
    database: str = "chromadb"
) -> Dict[str, Any]:
    """
    Store vector in specified database with real implementation
    
    Supported databases: chromadb, qdrant, faiss
    """
    try:
        vector_dim = len(data.vector)
        collection_name = f"user_{current_user.id}_vectors"
        
        if database == "chromadb":
            if not service_connections.chroma_client:
                raise HTTPException(status_code=503, detail="ChromaDB not available")
            
            try:
                # Get or create collection
                collection = service_connections.chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"user_id": str(current_user.id)}
                )
                
                # Add vector
                collection.add(
                    ids=[data.id],
                    embeddings=[data.vector],
                    metadatas=[data.metadata] if data.metadata else None
                )
                
                logger.info(f"Stored vector {data.id} in ChromaDB collection {collection_name}")
                
                return {
                    "status": "success",
                    "database": "chromadb",
                    "collection": collection_name,
                    "vector_id": data.id,
                    "dimension": vector_dim
                }
                
            except Exception as e:
                logger.error(f"ChromaDB storage error: {e}")
                raise HTTPException(status_code=500, detail=f"ChromaDB error: {str(e)}")
        
        elif database == "qdrant":
            if not service_connections.qdrant_client:
                raise HTTPException(status_code=503, detail="Qdrant not available")
            
            try:
                from qdrant_client.models import Distance, VectorParams, PointStruct
                
                # Ensure collection exists
                collections = await service_connections.qdrant_client.get_collections()
                collection_names = [c.name for c in collections.collections]
                
                if collection_name not in collection_names:
                    await service_connections.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
                    )
                
                # Upsert point
                await service_connections.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=hash(data.id) % (2**63),  # Convert string ID to int
                            vector=data.vector,
                            payload={"original_id": data.id, **data.metadata}
                        )
                    ]
                )
                
                logger.info(f"Stored vector {data.id} in Qdrant collection {collection_name}")
                
                return {
                    "status": "success",
                    "database": "qdrant",
                    "collection": collection_name,
                    "vector_id": data.id,
                    "dimension": vector_dim
                }
                
            except Exception as e:
                logger.error(f"Qdrant storage error: {e}")
                raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")
        
        elif database == "faiss":
            # FAISS requires index management - simplified in-memory example
            # In production, use persistent FAISS service
            raise HTTPException(
                status_code=501,
                detail="FAISS storage requires external service setup"
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported database: {database}. Use: chromadb, qdrant, faiss"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector storage error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")


@router.post("/search")
async def search_vectors(
    search_request: VectorSearchRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    database: str = "qdrant"
) -> List[Dict[str, Any]]:
    """
    Search for similar vectors with real implementation
    
    Supported databases: chromadb, qdrant
    """
    try:
        collection_name = f"user_{current_user.id}_vectors"
        
        if database == "chromadb":
            if not service_connections.chroma_client:
                raise HTTPException(status_code=503, detail="ChromaDB not available")
            
            try:
                collection = service_connections.chroma_client.get_collection(name=collection_name)
                
                results = collection.query(
                    query_embeddings=[search_request.query],
                    n_results=search_request.k,
                    where=search_request.filter if search_request.filter else None
                )
                
                # Format results
                similar_vectors = []
                if results['ids'] and len(results['ids']) > 0:
                    for i in range(len(results['ids'][0])):
                        similar_vectors.append({
                            "id": results['ids'][0][i],
                            "score": 1.0 - results['distances'][0][i] if results.get('distances') else None,
                            "metadata": results['metadatas'][0][i] if results.get('metadatas') else {}
                        })
                
                logger.info(f"Found {len(similar_vectors)} similar vectors in ChromaDB")
                return similar_vectors
                
            except Exception as e:
                logger.error(f"ChromaDB search error: {e}")
                # Return empty if collection doesn't exist
                if "does not exist" in str(e).lower():
                    return []
                raise HTTPException(status_code=500, detail=f"ChromaDB search error: {str(e)}")
        
        elif database == "qdrant":
            if not service_connections.qdrant_client:
                raise HTTPException(status_code=503, detail="Qdrant not available")
            
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                # Build filter if provided
                query_filter = None
                if search_request.filter:
                    conditions = [
                        FieldCondition(key=k, match=MatchValue(value=v))
                        for k, v in search_request.filter.items()
                    ]
                    query_filter = Filter(must=conditions)
                
                # Search
                results = await service_connections.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=search_request.query,
                    limit=search_request.k,
                    query_filter=query_filter
                )
                
                # Format results
                similar_vectors = [
                    {
                        "id": result.payload.get("original_id", str(result.id)),
                        "score": result.score,
                        "metadata": {k: v for k, v in result.payload.items() if k != "original_id"}
                    }
                    for result in results
                ]
                
                logger.info(f"Found {len(similar_vectors)} similar vectors in Qdrant")
                return similar_vectors
                
            except Exception as e:
                logger.error(f"Qdrant search error: {e}")
                # Return empty if collection doesn't exist
                if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                    return []
                raise HTTPException(status_code=500, detail=f"Qdrant search error: {str(e)}")
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported database: {database}. Use: chromadb, qdrant"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.delete("/{vector_id}")
async def delete_vector(
    vector_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    database: str = "chromadb"
) -> Dict[str, Any]:
    """Delete vector from specified database"""
    try:
        collection_name = f"user_{current_user.id}_vectors"
        
        if database == "chromadb":
            if not service_connections.chroma_client:
                raise HTTPException(status_code=503, detail="ChromaDB not available")
            
            collection = service_connections.chroma_client.get_collection(name=collection_name)
            collection.delete(ids=[vector_id])
            
            return {"status": "success", "database": "chromadb", "deleted_id": vector_id}
        
        elif database == "qdrant":
            if not service_connections.qdrant_client:
                raise HTTPException(status_code=503, detail="Qdrant not available")
            
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Delete by payload filter
            await service_connections.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="original_id", match=MatchValue(value=vector_id))]
                )
            )
            
            return {"status": "success", "database": "qdrant", "deleted_id": vector_id}
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported database: {database}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector deletion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")