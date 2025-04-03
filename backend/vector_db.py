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

from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer
from backend.core.config import settings
from backend.core.exceptions import DatabaseError, ResourceNotFoundError
from backend.models.base_models import Document, DocumentChunk
from qdrant_client import QdrantClient, models

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
            model_name = settings.EMBEDDING_MODEL_NAME if hasattr(settings, 'EMBEDDING_MODEL_NAME') else "all-MiniLM-L6-v2"
            logger.info(f"Loading embedding model: {model_name}")
            # Initialize SentenceTransformer with the model name only
            embedding_model = SentenceTransformer(model_name)
            # Attempt to get embedding dimension to confirm model loaded
            embedding_model.get_sentence_embedding_dimension()
            logger.info("Embedding model loaded successfully.")
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
    embedding_dim = get_embedding_model().get_sentence_embedding_dimension()

    collections = [settings.QDRANT_COLLECTION, settings.QDRANT_CHUNK_COLLECTION]
    for collection_name in collections:
        try:
            client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists.")
        except Exception: # Catch specific exception if known, else general Exception
            logger.info(f"Collection '{collection_name}' not found, creating...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{collection_name}' created.")

            # Create payload indexes for faster filtering
            if collection_name == settings.QDRANT_COLLECTION:
                client.create_payload_index( # type: ignore [attr-defined]
                    collection_name=collection_name,
                    field_name="metadata.doc_id",
                    field_schema="keyword"
                )
                client.create_payload_index( # type: ignore [attr-defined]
                    collection_name=collection_name,
                    field_name="metadata.created_at",
                    field_schema="datetime"
                )
            elif collection_name == settings.QDRANT_CHUNK_COLLECTION:
                client.create_payload_index( # type: ignore [attr-defined]
                    collection_name=collection_name,
                    field_name="metadata.doc_id",
                    field_schema="keyword"
                )
                client.create_payload_index( # type: ignore [attr-defined]
                    collection_name=collection_name,
                    field_name="metadata.chunk_index",
                    field_schema="integer"
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
        # Use the initialized model
        embeddings = self.embedding_model.encode(text, convert_to_tensor=False)
        # Ensure the output is List[float]
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist() # Convert numpy array to list

        if isinstance(embeddings, list) and all(isinstance(x, float) for x in embeddings):
            return embeddings
        else:
            logger.error(f"Unexpected embedding type: {type(embeddings)} for text: '{text[:50]}...'")
            dim = self.embedding_model.get_sentence_embedding_dimension()
            # Provide a default zero vector if conversion fails
            return [0.0] * (dim if dim is not None else 384) # Use default dim if None


    def _embed_texts(self, texts: List[str]) -> List[Embedding]:
        """Embeds a list of texts."""
        # Use the initialized model
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)

        # Ensure the output is List[List[float]]
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist() # Convert numpy array to list of lists

        if isinstance(embeddings, list) and all(isinstance(emb, list) and all(isinstance(x, float) for x in emb) for emb in embeddings):
            return embeddings
        else:
            logger.error(f"Unexpected embedding type for batch: {type(embeddings)}")
            dim = self.embedding_model.get_sentence_embedding_dimension()
            # Provide a default zero vector list if conversion fails
            fallback_dim = dim if dim is not None else 384
            return [[0.0] * fallback_dim] * len(texts)

    def get_or_create_collection(self, collection_name: str, vector_dim: int):
        """Gets a collection or creates it if it doesn't exist."""
        try:
            self.client.get_collection(collection_name=collection_name) # type: ignore[no-untyped-call]
            logger.info(f"Collection '{collection_name}' already exists.")
        except Exception: # Catches specific Qdrant client exception for non-existent collection if available, else generic
            logger.info(f"Collection '{collection_name}' not found, creating...")
            self.client.create_collection( # type: ignore[no-untyped-call]
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE) # type: ignore[attr-defined]
            )
            # Create payload indexes for potentially filterable fields
            # Use try-except for each index creation as fields might not always exist
            try:
                self.client.create_payload_index(collection_name=collection_name, field_name="doc_id", field_schema="keyword") # type: ignore[no-untyped-call]
            except Exception as e:
                 logger.warning(f"Could not create payload index for 'doc_id' in {collection_name}: {e}")
            try:
                self.client.create_payload_index(collection_name=collection_name, field_name="source", field_schema="keyword") # type: ignore[no-untyped-call]
            except Exception as e:
                 logger.warning(f"Could not create payload index for 'source' in {collection_name}: {e}")
            try:
                # Index 'page' only if it's expected to be consistently present and numeric
                self.client.create_payload_index(collection_name=collection_name, field_name="page", field_schema="integer") # type: ignore[no-untyped-call]
            except Exception as e:
                 logger.warning(f"Could not create payload index for 'page' in {collection_name}: {e}")

            logger.info(f"Collection '{collection_name}' created with vector dimension {vector_dim}.")

    def delete_collection(self, collection_name: str):
        """Deletes a collection."""
        try:
            self.client.delete_collection(collection_name=collection_name) # type: ignore[no-untyped-call]
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
                 models.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) # type: ignore[attr-defined]
                 for i in range(len(ids))
            ]

            self.client.upsert(collection_name=collection_name, points=points, wait=True) # type: ignore[no-untyped-call]
            logger.info(f"Added/updated {len(documents)} documents in collection '{collection_name}'.")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents to collection '{collection_name}': {e}")
            raise DatabaseError("Failed to add documents") from e

    def query(self, collection_name: str, query_text: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Queries the collection for similar documents."""
        try:
            query_vector = self._embed_text(query_text)
            search_result = self.client.search( # type: ignore[no-untyped-call]
                collection_name=collection_name,
                query_vector=query_vector,
                limit=n_results,
                # with_payload=True # Include payload in results
            )
            # Format results
            results = [
                {"id": hit.id, "score": hit.score, "payload": hit.payload} # type: ignore[attr-defined]
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Vector Database abstraction
class VectorDB:
    def __init__(self, persist_dir: str = "data/vector_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        if CHROMA_AVAILABLE:
            logger.info("Using ChromaDB as vector database")
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collections: Dict[str, Any] = {}
        elif FAISS_AVAILABLE:
            logger.info("Using FAISS as vector database")
            self.faiss_indexes: Dict[str, faiss.Index] = {}
            self.faiss_documents: Dict[str, List[DocumentInput]] = {}
            self.faiss_ids: Dict[str, List[str]] = {}
        else:
            raise RuntimeError("No vector database backend available")

    def get_or_create_collection(self, name: str) -> Any:
        """Get or create a collection in the vector database"""
        if CHROMA_AVAILABLE:
            try:
                collection = self.client.get_collection(name)
                logger.info(f"Retrieved existing collection: {name}")
                return collection
            except Exception as e:
                logger.info(f"Creating new collection: {name}. Reason: {e}")
                collection = self.client.create_collection(name)
                self.client.create_payload_index(
                    collection_name=name,
                    field_name="metadata.source",
                    field_schema="keyword",
                )
                self.client.create_payload_index(
                    collection_name=name,
                    field_name="metadata.doc_id",
                    field_schema="keyword",
                )
                return collection
        elif FAISS_AVAILABLE:
            if name not in self.faiss_indexes:
                # For FAISS we'll use a simple index with 384 dimensions (typical for embeddings)
                dimension = 384
                logger.info(f"Creating new FAISS index for collection: {name}")
                index = faiss.IndexFlatL2(dimension)
                self.faiss_indexes[name] = index
                self.faiss_documents[name] = []
                self.faiss_ids[name] = []
            return name

    def add_documents(self, collection_name: str, documents: List[DocumentInput]):
        """Add documents to a collection"""
        collection = self.get_or_create_collection(collection_name)

        if CHROMA_AVAILABLE:
            # Extract data from the documents
            texts = [doc.text for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [
                doc.id or f"doc_{i}_{int(time.time())}"
                for i, doc in enumerate(documents)
            ]

            # Add to ChromaDB
            collection.add(documents=texts, metadatas=metadatas, ids=ids)
            return ids
        elif FAISS_AVAILABLE:
            # For FAISS we need to handle embeddings ourselves
            # This is a simplified example - in a real system we'd use a model to create embeddings
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")

            texts = [doc.text for doc in documents]
            embeddings = model.encode(texts)

            ids = [
                doc.id
                or f"doc_{len(self.faiss_ids[collection_name]) + i}_{int(time.time())}"
                for i, doc in enumerate(documents)
            ]

            # Add to FAISS index
            index = self.faiss_indexes[collection_name]
            index.add(np.array(embeddings).astype("float32"))

            # Store documents and ids
            self.faiss_documents[collection_name].extend(documents)
            self.faiss_ids[collection_name].extend(ids)

            return ids

    def query(self, collection_name: str, query_text: str, n_results: int = 5):
        """Query the vector database for similar documents"""
        if collection_name not in self.get_collections():
            raise ValueError(f"Collection {collection_name} does not exist")

        if CHROMA_AVAILABLE:
            collection = self.get_or_create_collection(collection_name)
            results = collection.query(query_texts=[query_text], n_results=n_results)
            return results
        elif FAISS_AVAILABLE:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")

            # Create query embedding
            query_embedding = (
                model.encode([query_text])[0].astype("float32").reshape(1, -1)
            )

            # Search in FAISS
            index = self.faiss_indexes[collection_name]
            D, result_indices = index.search(query_embedding, n_results)

            # Format results similar to ChromaDB
            documents = []
            metadatas = []
            ids = []
            distances = []

            for i in range(len(result_indices[0])):
                if result_indices[0][i] < len(self.faiss_documents[collection_name]):
                    idx = result_indices[0][i]
                    documents.append(self.faiss_documents[collection_name][idx].text)
                    metadatas.append(
                        self.faiss_documents[collection_name][idx].metadata
                    )
                    ids.append(self.faiss_ids[collection_name][idx])
                    distances.append(float(D[0][i]))

            return {
                "documents": [documents],
                "metadatas": [metadatas],
                "ids": [ids],
                "distances": [distances],
            }

    def get_collections(self):
        """Get all collections in the vector database"""
        if CHROMA_AVAILABLE:
            return self.client.list_collections()
        elif FAISS_AVAILABLE:
            return list(self.faiss_indexes.keys())

    def delete_collection(self, name: str):
        """Delete a collection"""
        if CHROMA_AVAILABLE:
            self.client.delete_collection(name)
        elif FAISS_AVAILABLE:
            if name in self.faiss_indexes:
                del self.faiss_indexes[name]
                del self.faiss_documents[name]
                del self.faiss_ids[name]


# Create vector database instance
vector_db = VectorDB()


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
        store = VectorStore()
        collections = store.client.get_collections()
        return {"collections": [col.name for col in collections.collections]}
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections")
async def create_collection(collection: CollectionInput):
    try:
        vector_db.get_or_create_collection(collection.name)
        return {
            "message": f"Collection {collection.name} created or retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{name}")
async def delete_collection(name: str):
    try:
        vector_db.delete_collection(name)
        return {"message": f"Collection {name} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def add_documents(documents: List[DocumentInput], collection: str = "default"):
    try:
        ids = vector_db.add_documents(collection, documents)
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
        results = vector_db.query(
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
