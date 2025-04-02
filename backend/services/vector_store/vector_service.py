"""
SutazAI Vector Store Service
Provides functionality for storing and searching document vectors/embeddings
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from sentence_transformers import SentenceTransformer
import uuid
import traceback

# Import core settings
from backend.core.config import get_settings

# Configure logging
logger = logging.getLogger("vector_store")

# Get application settings
settings = get_settings()


class VectorStore:
    """
    Vector store implementation for document embeddings
    Uses Qdrant for vector search with fallback to file-based storage
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "documents",
        fallback_enabled: bool = True,
    ):
        """
        Initialize the vector store service

        Args:
            host: Host address for Qdrant (defaults to settings.QDRANT_HOST)
            port: Port for Qdrant (defaults to settings.QDRANT_PORT)
            collection_name: Name of the collection in Qdrant
            fallback_enabled: Whether to enable fallback to file-based storage
        """
        # Use settings if not explicitly provided
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name
        self.fallback_enabled = fallback_enabled
        self.vectors_dir = Path("/opt/sutazaiapp/data/vectors")
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

        # In-memory fallback storage
        self._documents = {}

        # Initialize the embedding model
        try:
            model_name = settings.SENTENCE_TRANSFORMER_MODEL
            logger.info(f"Initializing embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            self.embedding_model = None

        # Try to connect to Qdrant
        self.client = None
        self._qdrant_available = False
        self._connect_to_qdrant()

        # If Qdrant is not available and fallback is enabled, load from files
        if not self._qdrant_available and fallback_enabled:
            logger.warning("Using file-based vector store as fallback")
            self._load_documents()

        logger.info(f"Initialized VectorStore with storage path: {self.vectors_dir}")

    def _connect_to_qdrant(self, max_retries: int = 3) -> bool:
        """Attempt to connect to Qdrant service"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest
            from qdrant_client.http.exceptions import UnexpectedResponse

            # Try to connect with retries
            for attempt in range(max_retries):
                try:
                    self.client = QdrantClient(host=self.host, port=self.port)

                    # Check if collection exists, create if it doesn't
                    collections = self.client.get_collections().collections
                    collection_names = [c.name for c in collections]

                    if self.collection_name not in collection_names:
                        # Create the collection
                        self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=rest.VectorParams(
                                size=768,  # Default for most sentence transformers
                                distance=rest.Distance.COSINE,
                            ),
                        )
                        logger.info(
                            f"Created collection '{self.collection_name}' in Qdrant"
                        )

                    self._qdrant_available = True
                    logger.info(
                        f"Successfully connected to Qdrant at {self.host}:{self.port}"
                    )
                    return True

                except (ConnectionError, UnexpectedResponse) as e:
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt  # Exponential backoff
                        logger.warning(
                            f"Failed to connect to Qdrant (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait_time}s: {str(e)}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Failed to connect to Qdrant after {max_retries} attempts: {str(e)}"
                        )
                        return False
        except ImportError:
            logger.error(
                "Qdrant client not installed. Install with: pip install qdrant-client"
            )
            return False
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def _load_documents(self):
        """Load existing documents from the vector store directory"""
        try:
            for file_path in self.vectors_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        doc_data = json.load(f)
                        doc_id = file_path.stem
                        self._documents[doc_id] = doc_data
                        logger.debug(f"Loaded document {doc_id} from vector store")
                except Exception as e:
                    logger.error(f"Error loading document from {file_path}: {str(e)}")
            logger.info(f"Loaded {len(self._documents)} documents from vector store")
        except Exception as e:
            logger.error(f"Error loading documents from vector store: {str(e)}")

    def _create_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Create embeddings for a list of texts"""
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return None

        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return None

    def store_document(self, document_data: Dict[str, Any]) -> bool:
        """
        Store a document in the vector store

        Args:
            document_data: Document data including text, metadata, and document_id

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            doc_id = document_data.get("document_id")
            if not doc_id:
                doc_id = str(uuid.uuid4())
                document_data["document_id"] = doc_id

            full_text = document_data.get("full_text", "")
            if not full_text:
                logger.warning(f"Document {doc_id} has no text content")

            # Create chunks for better searching
            chunks = self._create_text_chunks(full_text)
            document_data["chunks"] = chunks

            # Store in Qdrant if available
            if self._qdrant_available and self.client:
                try:
                    from qdrant_client.http import models as rest

                    # Create embeddings for chunks
                    chunk_embeddings = self._create_embeddings(chunks)
                    if not chunk_embeddings:
                        raise ValueError("Failed to create embeddings")

                    # Create points for each chunk
                    points = []
                    for i, (chunk, embedding) in enumerate(
                        zip(chunks, chunk_embeddings)
                    ):
                        points.append(
                            rest.PointStruct(
                                id=f"{doc_id}_{i}",
                                vector=embedding,
                                payload={
                                    "document_id": doc_id,
                                    "chunk_index": i,
                                    "text": chunk,
                                    "metadata": document_data.get("metadata", {}),
                                },
                            )
                        )

                    # Upsert points in batches
                    if points:
                        self.client.upsert(
                            collection_name=self.collection_name, points=points
                        )

                    logger.info(
                        f"Stored document {doc_id} with {len(chunks)} chunks in Qdrant"
                    )
                except Exception as e:
                    logger.error(f"Error storing document in Qdrant: {str(e)}")
                    if not self.fallback_enabled:
                        return False
                    logger.warning("Falling back to file-based storage")

            # Fallback to file storage if Qdrant is unavailable or as a backup
            if not self._qdrant_available or self.fallback_enabled:
                # Store document data in memory
                self._documents[doc_id] = document_data

                # Save to disk
                file_path = self.vectors_dir / f"{doc_id}.json"
                with open(file_path, "w") as f:
                    json.dump(document_data, f)

                logger.info(f"Stored document {doc_id} in file-based vector store")

            return True
        except Exception as e:
            logger.error(f"Error storing document in vector store: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store

        Args:
            document_id: ID of the document to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete from Qdrant if available
            if self._qdrant_available and self.client:
                try:
                    # Delete by filter on document_id
                    from qdrant_client.http import models as rest

                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=rest.FilterSelector(
                            filter=rest.Filter(
                                must=[
                                    rest.FieldCondition(
                                        key="document_id",
                                        match=rest.MatchValue(value=document_id),
                                    )
                                ]
                            )
                        ),
                    )
                    logger.info(f"Deleted document {document_id} from Qdrant")
                except Exception as e:
                    logger.error(f"Error deleting document from Qdrant: {str(e)}")
                    if not self.fallback_enabled:
                        return False

            # Delete from in-memory store
            if document_id in self._documents:
                del self._documents[document_id]

            # Delete from disk
            file_path = self.vectors_dir / f"{document_id}.json"
            if file_path.exists():
                os.remove(file_path)

            logger.info(f"Deleted document {document_id} from file-based vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting document from vector store: {str(e)}")
            return False

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query

        Args:
            query: Search query text
            limit: Maximum number of results to return

        Returns:
            List of document matches with similarity scores
        """
        try:
            # Search in Qdrant if available
            if self._qdrant_available and self.client and self.embedding_model:
                try:
                    # Create query embedding
                    query_embedding = self._create_embeddings([query])
                    if not query_embedding:
                        raise ValueError("Failed to create query embedding")

                    # Search in Qdrant
                    search_result = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding[0],
                        limit=limit,
                    )

                    # Format results
                    results = []
                    for hit in search_result:
                        results.append(
                            {
                                "document_id": hit.payload.get("document_id"),
                                "score": hit.score,
                                "text_snippet": hit.payload.get("text", ""),
                            }
                        )

                    logger.info(
                        f"Search for '{query}' returned {len(results)} results from Qdrant"
                    )
                    return results
                except Exception as e:
                    logger.error(f"Error searching in Qdrant: {str(e)}")
                    if not self.fallback_enabled:
                        return []
                    logger.warning("Falling back to file-based search")

            # Fallback to simple keyword search
            results = []
            for doc_id, doc_data in self._documents.items():
                full_text = doc_data.get("full_text", "").lower()
                if query.lower() in full_text:
                    match_score = 0.5  # Placeholder score
                    results.append(
                        {
                            "document_id": doc_id,
                            "score": match_score,
                            "text_snippet": self._get_snippet(
                                full_text, query, max_length=200
                            ),
                        }
                    )

            # Sort by score (descending) and limit results
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
            logger.info(
                f"Search for '{query}' returned {len(results)} results from file-based search"
            )
            return results
        except Exception as e:
            logger.error(f"Error searching in vector store: {str(e)}")
            return []

    def _get_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """Extract a text snippet around the query match"""
        try:
            query_pos = text.lower().find(query.lower())
            if query_pos == -1:
                return text[:max_length] + "..."

            start = max(0, query_pos - max_length // 2)
            end = min(len(text), query_pos + len(query) + max_length // 2)

            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."

            return snippet
        except Exception:
            return text[:max_length] + "..."

    def _create_text_chunks(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks for better search

        Args:
            text: The text to split into chunks
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # Try to find a good breaking point (newline or space)
            if end < text_length:
                # Look for newline first
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start + chunk_size // 2:
                    end = newline_pos + 1
                else:
                    # Look for space
                    space_pos = text.rfind(" ", start + chunk_size // 2, end)
                    if space_pos > start + chunk_size // 2:
                        end = space_pos + 1

            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Calculate next start position with overlap
            start = end - overlap
            if start < 0 or start >= text_length:
                break

        return chunks

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store services"""
        status = {
            "qdrant_available": self._qdrant_available,
            "embedding_model_available": self.embedding_model is not None,
            "fallback_enabled": self.fallback_enabled,
            "fallback_documents_loaded": len(self._documents),
            "vectors_dir": str(self.vectors_dir),
        }

        # Try reconnecting to Qdrant if it was unavailable
        if not self._qdrant_available:
            status["reconnect_attempt"] = self._connect_to_qdrant(max_retries=1)
            status["qdrant_available"] = self._qdrant_available

        return status
