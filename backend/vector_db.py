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
from typing import Dict, List, Optional, Any

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


# Define API models
class DocumentInput(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = None


class QueryInput(BaseModel):
    query: str
    n_results: int = 5
    collection: str = "default"


class CollectionInput(BaseModel):
    name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
            self.collections = {}
        elif FAISS_AVAILABLE:
            logger.info("Using FAISS as vector database")
            self.faiss_indexes = {}
            self.faiss_documents = {}
            self.faiss_ids = {}
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
        collections = vector_db.get_collections()
        return {"collections": collections}
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

    # Check if we have at least one vector database backend
    if not CHROMA_AVAILABLE and not FAISS_AVAILABLE:
        logger.error(
            "No vector database backend available. Please install either chromadb or faiss."
        )
        sys.exit(1)

    # Run the FastAPI application
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
