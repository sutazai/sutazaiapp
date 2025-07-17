#!/usr/bin/env python3
"""
Vector Database Service

This module provides a lightweight vector database service for SutazAI
using either ChromaDB or FAISS as the backend.
"""

import os
import sys
import argparse
import logging
import time
from typing import Dict, List, Optional, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vector_db")

# Try to import vector database libraries, with fallbacks
try:
    import chromadb

    CHROMA_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB not available. Will try to use FAISS.")
    CHROMA_AVAILABLE = False

try:
    import faiss
    import numpy as np

    FAISS_AVAILABLE = True
except ImportError:
    if not CHROMA_AVAILABLE:
        logger.error(
            "Neither ChromaDB nor FAISS is available. Please install one of them."
        )
        logger.error("Run: pip install chromadb or pip install faiss-cpu")
    FAISS_AVAILABLE = False

# Import FastAPI and related dependencies
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    logger.error("FastAPI dependencies not found. Please install them.")
    logger.error("Run: pip install fastapi uvicorn")
    sys.exit(1)

from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
from backend.core.config import settings, get_settings
from backend.core.exceptions import DatabaseError, ResourceNotFoundError
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    ScoredPoint
)
from qdrant_client.http.models import models
from backend.models.base_models import DocumentChunk

logger = logging.getLogger(__name__)

# Define a type alias for embedding results
Embedding = List[float]

# Global Qdrant client instance
client: Optional[QdrantClient] = None
embedding_model: Optional[SentenceTransformer] = None

def get_qdrant_client() -> QdrantClient:
    """Initializes and returns the Qdrant client."""
    global client
    if client is None:
        try:
            logger.info(f"Initializing Qdrant client: host={settings.QDRANT_HOST}, port={settings.QDRANT_PORT}")
            client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY if hasattr(settings, 'QDRANT_API_KEY') else None,
                https=settings.QDRANT_HTTPS if hasattr(settings, 'QDRANT_HTTPS') else None,
            )
            # Verify connection by trying to get collections
            client.get_collections()
            logger.info("Qdrant client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise DatabaseError("Qdrant connection failed") from e
    return client

def get_embedding_model() -> SentenceTransformer:
    """Initializes and returns the sentence transformer model."""
    global embedding_model
    if embedding_model is None:
        try:
            # Use default model if setting is not present
            model_name = str(settings.EMBEDDING_MODEL_NAME) if hasattr(settings, 'EMBEDDING_MODEL_NAME') else "all-MiniLM-L6-v2"
            logger.info(f"Loading embedding model: {model_name}")
            # Initialize SentenceTransformer with the model name only
            embedding_model = SentenceTransformer(model_name)
            # Attempt to get embedding dimension to confirm model loaded
            if embedding_model:
                 assert embedding_model is not None
                 dim_check = embedding_model.get_sentence_embedding_dimension()
                 if dim_check is None:
                      raise ValueError("Could not determine embedding dimension after loading model.")
                 logger.info(f"Embedding model loaded successfully with dimension: {dim_check}.")
            else:
                 raise ValueError("Embedding model failed to initialize.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise DatabaseError("Embedding model loading failed") from e
    return embedding_model

def initialize_vector_db():
    """Initialize vector database connections and collections."""
    get_qdrant_client() # Ensure client is initialized
    get_embedding_model() # Ensure embedding model is initialized
    create_collections()

def create_collections():
    """Create necessary Qdrant collections if they don't exist."""
    client = get_qdrant_client()
    embedding_model_instance = get_embedding_model()
    # Use assert to hint non-None to type checker
    assert embedding_model_instance is not None
    embedding_dim = embedding_model_instance.get_sentence_embedding_dimension()
    if embedding_dim is None:
        raise DatabaseError("Could not determine embedding dimension")

    collections = [settings.QDRANT_COLLECTION, settings.QDRANT_CHUNK_COLLECTION]
    for collection_name in collections:
        try:
            client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists.")
        except Exception: # Catch specific exception if known, else general Exception
            logger.info(f"Collection '{collection_name}' not found, creating...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
            )
            logger.info(f"Collection '{collection_name}' created.")

            # Create payload indexes for faster filtering
            if collection_name == settings.QDRANT_COLLECTION:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="metadata.doc_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="metadata.created_at",
                    field_schema=PayloadSchemaType.DATETIME
                )
            elif collection_name == settings.QDRANT_CHUNK_COLLECTION:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="metadata.doc_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="metadata.chunk_index",
                    field_schema=PayloadSchemaType.INTEGER
                )


class VectorStore:
    """Interface for interacting with the vector database."""

    def __init__(self):
        self.client = get_qdrant_client()
        self.embedding_model = get_embedding_model()
        self.doc_collection = settings.QDRANT_COLLECTION
        self.chunk_collection = settings.QDRANT_CHUNK_COLLECTION
        self._ensure_collections_exist()

    def _ensure_collections_exist(self):
        """Ensures the document and chunk collections exist in Qdrant."""
        # Use assert to hint non-None
        assert self.embedding_model is not None
        dim = self.embedding_model.get_sentence_embedding_dimension()
        if dim is None:
             raise ValueError("Embedding model dimension could not be determined.")

        try:
            self.get_or_create_collection(self.doc_collection, dim)
            self.get_or_create_collection(self.chunk_collection, dim)
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collections exist: {e}")
            raise DatabaseError("Failed to create or verify Qdrant collections") from e

    def _embed_text(self, text: str) -> Embedding:
        """Embeds a single string of text."""
        assert self.embedding_model is not None
        embeddings_np = self.embedding_model.encode(text, convert_to_tensor=False)
        # Ensure the output is List[float]
        if isinstance(embeddings_np, np.ndarray):
            embeddings = embeddings_np.tolist() # Convert numpy array to list
        elif isinstance(embeddings_np, list): # Handle case where it might already be list
            embeddings = embeddings_np
        else:
            logger.error(f"Unexpected embedding type: {type(embeddings_np)} for text: '{text[:50]}...'")
            assert self.embedding_model is not None
            dim = self.embedding_model.get_sentence_embedding_dimension()
            return [0.0] * (dim if dim is not None else 384)

        if isinstance(embeddings, list) and all(isinstance(x, (float, int)) for x in embeddings): # Allow int/float
             return [float(x) for x in embeddings] # Ensure float
        else:
            logger.error(f"Unexpected embedding format after conversion: {type(embeddings)}")
            assert self.embedding_model is not None
            dim = self.embedding_model.get_sentence_embedding_dimension()
            return [0.0] * (dim if dim is not None else 384)

    def _embed_texts(self, texts: List[str]) -> List[Embedding]:
        """Embeds a list of texts."""
        assert self.embedding_model is not None
        embeddings_np = self.embedding_model.encode(texts, convert_to_tensor=False)

        # Ensure the output is List[List[float]]
        if isinstance(embeddings_np, np.ndarray):
            embeddings = embeddings_np.tolist() # Convert numpy array to list of lists
        elif isinstance(embeddings_np, list): # Handle case where it might already be list
            embeddings = embeddings_np
        else:
            logger.error(f"Unexpected embedding type for batch: {type(embeddings_np)}")
            assert self.embedding_model is not None
            dim = self.embedding_model.get_sentence_embedding_dimension()
            fallback_dim = dim if dim is not None else 384
            return [[0.0] * fallback_dim] * len(texts)

        # Check inner lists and convert elements to float
        processed_embeddings: List[Embedding] = []
        for emb in embeddings:
            if isinstance(emb, list) and all(isinstance(x, (float, int)) for x in emb):
                processed_embeddings.append([float(x) for x in emb])
            else:
                logger.error(f"Unexpected inner embedding format: {type(emb)}")
                assert self.embedding_model is not None
                dim = self.embedding_model.get_sentence_embedding_dimension()
                fallback_dim = dim if dim is not None else 384
                processed_embeddings.append([0.0] * fallback_dim)
        return processed_embeddings

    def get_or_create_collection(self, collection_name: str, vector_dim: int):
        """Gets a collection or creates it if it doesn't exist."""
        try:
            self.client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists.")
        except Exception:
            logger.info(f"Collection '{collection_name}' not found, creating...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
            )
            # Create payload indexes for potentially filterable fields
            # Use try-except for each index creation as fields might not always exist
            try:
                self.client.create_payload_index(collection_name=collection_name, field_name="doc_id", field_schema=PayloadSchemaType.KEYWORD)
            except Exception as e:
                 logger.warning(f"Could not create payload index for 'doc_id' in {collection_name}: {e}")
            try:
                self.client.create_payload_index(collection_name=collection_name, field_name="source", field_schema=PayloadSchemaType.KEYWORD)
            except Exception as e:
                 logger.warning(f"Could not create payload index for 'source' in {collection_name}: {e}")
            try:
                self.client.create_payload_index(collection_name=collection_name, field_name="page", field_schema=PayloadSchemaType.INTEGER)
            except Exception as e:
                 logger.warning(f"Could not create payload index for 'page' in {collection_name}: {e}")

            logger.info(f"Collection '{collection_name}' created with vector dimension {vector_dim}.")

    def delete_collection(self, collection_name: str):
        """Deletes a collection."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted.")
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise DatabaseError(f"Failed to delete collection {collection_name}") from e

    def add_documents(self, collection_name: str, documents: List[DocumentInput]) -> List[str]:
        """Adds documents to the specified collection."""
        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        payloads = [doc.metadata or {} for doc in documents] # Ensure payload is a dict

        try:
            vectors = self._embed_texts(texts)
            if len(ids) != len(vectors) or len(ids) != len(payloads):
                 raise ValueError("Mismatch between number of ids, vectors, and payloads.")

            points = [
                 models.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
                 for i in range(len(ids))
            ]

            self.client.upsert(collection_name=collection_name, points=points, wait=True)
            logger.info(f"Added/updated {len(documents)} documents in collection '{collection_name}'.")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents to collection '{collection_name}': {e}")
            raise DatabaseError("Failed to add documents") from e

    def query(self, collection_name: str, query_text: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Queries the collection for similar documents."""
        try:
            query_vector = self._embed_text(query_text)
            search_result: List[ScoredPoint] = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=n_results,
                with_payload=True
            )
            # Format results
            results = [
                {"id": hit.id, "score": hit.score, "payload": hit.payload}
                for hit in search_result
            ]
            return results
        except Exception as e:
             # Check if the error is due to collection not found (needs specific exception handling if Qdrant provides one)
             if "not found" in str(e).lower():
                 logger.warning(f"Query failed because collection '{collection_name}' does not exist.")
                 raise ResourceNotFoundError(f"Collection '{collection_name}' not found.") from e
             logger.error(f"Failed to query collection '{collection_name}': {e}")
             raise DatabaseError("Failed to query documents") from e

# Helper function (consider moving to a utils module)
def create_point(id: str, vector: Embedding, payload: Optional[dict] = None) -> PointStruct:
    """Creates a Qdrant PointStruct."""
    return PointStruct(id=id, vector=vector, payload=payload or {})


# Define API models
class DocumentInput(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class QueryInput(BaseModel):
    collection: str
    query: str
    n_results: int = 10


class CollectionInput(BaseModel):
    name: str


# Create FastAPI app
app = FastAPI(
    title="SutazAI Vector Database",
    description="Vector database service for embedding and semantic search",
)

# SECURITY FIX: Secure CORS configuration for vector DB
from backend.security.secure_config import get_allowed_origins

# Add secure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),  # Environment-specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Restrict methods
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],  # Restrict headers
)


# Vector Database abstraction
class VectorDB:
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "sutazai_documents",
        model_name: str = "all-MiniLM-L6-v2", # Default embedding model
        db_path: Optional[str] = None, # For local Qdrant
        prefer_local: bool = True,
    ):
        settings = get_settings()
        self.qdrant_url = qdrant_url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.db_path = db_path
        self.prefer_local = prefer_local
        self.client: Optional[QdrantClient] = None
        self.model: Optional[SentenceTransformer] = None
        self.dimension: Optional[int] = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize Qdrant client and Sentence Transformer model."""
        try:
            model_name = str(settings.EMBEDDING_MODEL_NAME) if hasattr(settings, 'EMBEDDING_MODEL_NAME') else "all-MiniLM-L6-v2"
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            assert self.model is not None
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model '{self.model_name}': {e}")
            self.model = None
            self.dimension = None

        # Initialize Qdrant client
        qdrant_args = {}
        if self.qdrant_url:
             qdrant_args['url'] = self.qdrant_url
        if self.api_key:
            qdrant_args['api_key'] = self.api_key
        if self.db_path and self.prefer_local:
             qdrant_args['path'] = self.db_path
             logger.info(f"Using local Qdrant database at: {self.db_path}")
        elif not self.qdrant_url:
             logger.warning("No Qdrant URL or local path provided. Vector DB operations might fail.")

        try:
             # Define expected types for QdrantClient args
             typed_qdrant_args: Dict[str, Any] = {}
             if 'url' in qdrant_args:
                 typed_qdrant_args['url'] = str(qdrant_args['url'])
             if 'api_key' in qdrant_args:
                 typed_qdrant_args['api_key'] = str(qdrant_args['api_key']) if qdrant_args['api_key'] is not None else None
             if 'path' in qdrant_args:
                 typed_qdrant_args['path'] = str(qdrant_args['path'])
             # Add other potential args with correct types if needed

             self.client = QdrantClient(**typed_qdrant_args)
             # Check connection and collection existence
             self.client.get_collections()
             self.ensure_collection()
             logger.info(f"Connected to Qdrant. Using collection: '{self.collection_name}'")
        except Exception as e:
             logger.error(f"Failed to connect to Qdrant: {e}")
             self.client = None

    def ensure_collection(self):
        """Ensure the Qdrant collection exists."""
        if not self.client:
            logger.error("Qdrant client not initialized. Cannot ensure collection.")
            return
        if self.dimension is None:
             logger.error("Embedding dimension not known. Cannot create collection.")
             return

        try:
            self.client.get_collection(collection_name=self.collection_name)
            logger.debug(f"Collection '{self.collection_name}' already exists.")
        except Exception:
            logger.info(f"Creating collection '{self.collection_name}' with dimension {self.dimension}")
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE
                    ),
                )
                self.ensure_indexes()
            except Exception as create_e:
                 logger.error(f"Failed to create Qdrant collection: {create_e}")
                 self.client = None

    def _get_client(self) -> QdrantClient:
        """Get Qdrant client, raising error if unavailable."""
        if self.client is None:
            # Attempt re-initialization? Or just raise error.
            logger.error("Qdrant client is not available.")
            raise ConnectionError("Qdrant client is not connected or initialized.")
        return self.client

    def _embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        if self.model is None:
            logger.error("Embedding model not loaded. Cannot generate embeddings.")
            return None
        try:
            valid_texts: List[str] = [str(t) if t is not None else "" for t in texts]
            embeddings_np = self.model.encode(valid_texts, convert_to_tensor=False)
            if embeddings_np is None:
                logger.error("Embeddings generation returned None.")
                return None
            # Explicitly cast to list of lists of floats
            embeddings: List[List[float]] = [[float(f) for f in emb] for emb in embeddings_np.tolist()]
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            return None

    def upsert_documents(self, documents: List[DocumentChunk]):
        client = self._get_client()
        texts = [doc.content for doc in documents if doc.content is not None]
        embedding_results = self._embed_texts(texts)
        if embedding_results is None:
             logger.error("Failed to generate embeddings for upsert.")
             return

        points_to_upsert: List[PointStruct] = []
        embedding_idx = 0
        for doc in documents:
            if doc.content is not None:
                 if embedding_idx < len(embedding_results):
                     # Ensure payload is dict or None
                     payload = doc.metadata if isinstance(doc.metadata, dict) else None
                     point = models.PointStruct(
                         id=str(doc.id),
                         vector=embedding_results[embedding_idx],
                         payload=payload or {}, # Use payload or empty dict
                     )
                     points_to_upsert.append(point)
                     embedding_idx += 1
                 else:
                      logger.warning(f"Mismatch in embedding count for doc ID {doc.id}")

        if points_to_upsert:
            client.upsert(collection_name=self.collection_name, points=points_to_upsert)
            logger.info(f"Upserted {len(points_to_upsert)} points to '{self.collection_name}'.")
        else:
             logger.info("No valid points to upsert.")

    def retrieve_document(self, doc_id: Union[int, str]) -> Optional[DocumentChunk]:
        client = self._get_client()
        try:
            # Always use string ID for Qdrant
            points: List[models.Record] = client.retrieve(collection_name=self.collection_name, ids=[str(doc_id)], with_payload=True, with_vectors=True)
            if points:
                point = points[0]
                # Ensure payload is dict
                payload = point.payload if isinstance(point.payload, dict) else {}
                return DocumentChunk(
                    id=str(point.id),
                    content="",
                    metadata=payload, # Use checked payload
                    embedding=point.vector,
                )
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
        return None

    def search_similar(self, query: str, top_k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        client = self._get_client()
        query_embedding = self._embed_texts([query])

        if query_embedding is None or not query_embedding:
             logger.error("Could not generate embedding for the query.")
             return []

        qdrant_filter = self._build_filter(filter_criteria)

        search_result: List[ScoredPoint] = client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding[0],
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True
        )
        # Map ScoredPoint back to DocumentChunk
        results = []
        for hit in search_result:
             # Ensure payload is dict
             payload = hit.payload if isinstance(hit.payload, dict) else {}
             results.append(DocumentChunk(
                 id=str(hit.id),
                 content="",
                 metadata=payload, # Use checked payload
                 score=hit.score
             ))
            return results

    def _build_filter(self, filter_criteria: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if not filter_criteria:
            return None

        must_conditions: List[models.Condition] = []
        for key, value in filter_criteria.items():
            if isinstance(value, dict) and "match" in value:
                 match_val = value["match"]
                 # Check type of match_val before creating MatchValue
                 if isinstance(match_val, (str, int, bool)):
                     condition = FieldCondition(
                         key=f"metadata.{key}", # Assume keys map to metadata fields
                         match=MatchValue(value=match_val) # Use MatchValue
                     )
                     must_conditions.append(condition)
                 else:
                      logger.warning(f"Unsupported match value type for key '{key}': {type(match_val)}")
            elif isinstance(value, (str, int, bool)):
                 # Simpler equality match
                 condition = FieldCondition(
                     key=f"metadata.{key}",
                     match=MatchValue(value=value)
                 )
                 must_conditions.append(condition)
            # Add more condition types (range, geo, etc.) if needed

        if not must_conditions:
            return None

        return Filter(must=must_conditions)

    def delete_documents(self, doc_ids: List[Union[int, str]]):
        client = self._get_client()
        # Change hint to match expected type for PointIdsList
        str_ids: List[Union[int, str]] = [str(doc_id) for doc_id in doc_ids]
        # Remove unused ignore, keep necessary ones if library calls remain untyped
        client.delete(collection_name=self.collection_name, points_selector=models.PointIdsList(points=str_ids))

    def delete_by_metadata(self, filter_criteria: Dict[str, Any]):
        client = self._get_client()
        qdrant_filter = self._build_filter(filter_criteria)
        if not qdrant_filter:
            logger.warning("Invalid or empty filter provided for deletion by metadata.")
            return
        try:
            response = client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_filter
            )
            logger.info(f"Deleted points matching filter: {filter_criteria}. Response: {response}")
        except Exception as e:
            logger.error(f"Error deleting points by filter: {e}", exc_info=True)
            raise DatabaseError("Failed to delete points by filter") from e

    def ensure_indexes(self):
        """Ensure required payload fields are indexed for efficient filtering"""
        client = self._get_client()
        try:
            collection_info = client.get_collection(collection_name=self.collection_name)
            payload_schema = collection_info.payload_schema or {}
            indexes_to_create = {
                "metadata.doc_id": PayloadSchemaType.KEYWORD,
                "metadata.source": PayloadSchemaType.KEYWORD,
                "metadata.timestamp": PayloadSchemaType.DATETIME,
            }
            for field_name, field_type in indexes_to_create.items():
                 current_index = payload_schema.get(field_name)
                 if current_index is None or current_index.data_type != field_type:
                      logger.info(f"Creating payload index for '{field_name}' in {self.collection_name}")
                      client.create_payload_index(collection_name=self.collection_name, field_name=field_name, field_schema=field_type)

            logger.info(f"Index checks completed for collection {self.collection_name}")

        except Exception as e:
            logger.error(f"Error ensuring indexes for collection {self.collection_name}: {e}", exc_info=True)

def get_vector_db_client() -> VectorDB:
    """Factory function to get a configured VectorDB instance"""
    settings_obj = get_settings()
    return VectorDB(
        qdrant_url=settings_obj.QDRANT_URL,
        api_key=settings_obj.QDRANT_API_KEY,
        collection_name=settings_obj.QDRANT_COLLECTION,
        model_name=settings_obj.EMBEDDING_MODEL_NAME,
        db_path=settings_obj.QDRANT_LOCAL_PATH if settings_obj.QDRANT_PREFER_LOCAL else None,
        prefer_local=settings_obj.QDRANT_PREFER_LOCAL
    )


# Create vector database instance
# vector_db = get_vector_db_client() # This is the VectorDB wrapper
# Use VectorStore directly for API routes for now
vector_store = VectorStore()


# API routes
@app.get("/")
async def root():
    return {"message": "SutazAI Vector Database API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/collections")
async def get_collections():
    try:
        collections = vector_store.client.get_collections()
        return {"collections": [col.name for col in collections.collections]}
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections")
async def create_collection(collection: CollectionInput):
    try:
        # Use VectorStore instance
        # Assuming dimension is known or can be retrieved
        # Use assert to hint non-None
        assert vector_store.embedding_model is not None
        dim = vector_store.embedding_model.get_sentence_embedding_dimension()
        if dim is None:
             raise HTTPException(status_code=500, detail="Could not determine embedding dimension")
        vector_store.get_or_create_collection(collection.name, dim)
        return {
            "message": f"Collection {collection.name} created or retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{name}")
async def delete_collection(name: str):
    try:
        # Use VectorStore instance
        vector_store.delete_collection(name)
        return {"message": f"Collection {name} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def add_documents(documents: List[DocumentInput], collection: str = "default"):
    try:
        # Use VectorStore instance
        ids = vector_store.add_documents(collection, documents)
        return {
            "message": f"{len(documents)} documents added to collection {collection}",
            "ids": ids,
        }
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(query_input: QueryInput):
    try:
        # Use VectorStore instance
        results = vector_store.query(
            query_input.collection, query_input.query, query_input.n_results
        )
        return {"results": results}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error querying: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Main entry point
def main():
    parser = argparse.ArgumentParser(description="SutazAI Vector Database Service")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8502, help="Port to bind the server to"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Set log level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Starting Vector Database Service on {args.host}:{args.port}")

    # Initialize Qdrant and embedding model on startup
    try:
        initialize_vector_db()
    except DatabaseError as e:
        logger.critical(f"Failed to initialize vector database components: {e}. Exiting.")
        sys.exit(1)

    # Check if we have at least one vector database backend (optional, depends on VectorDB wrapper needs)
    # if not CHROMA_AVAILABLE and not FAISS_AVAILABLE:
    #     logger.error("No vector database backend available. Please install either chromadb or faiss.")
    #     sys.exit(1)

    # Run the FastAPI application
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
