import logging
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional, Tuple
import os

from sutazai_agi.core.config_loader import get_setting, PROJECT_ROOT
from sutazai_agi.models.llm_interface import get_llm_interface # Import LLM interface for embeddings

logger = logging.getLogger(__name__)

# --- ChromaDB Client Initialization --- 

_chroma_client: Optional[chromadb.Client] = None
_embedding_function: Optional[Any] = None

def get_embedding_function():
    """Gets the embedding function based on settings."""
    global _embedding_function
    if _embedding_function is None:
        # Option 1: Use Ollama for embeddings via our LLMInterface
        ollama_embedding_model = get_setting("default_embedding_model")
        if ollama_embedding_model:
            logger.info(f"Using Ollama model '{ollama_embedding_model}' for embeddings via LLMInterface.")
            llm_interface = get_llm_interface() # Ensures LLM interface is initialized

            class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
                def __init__(self, ollama_model_name: str):
                    self._ollama_model_name = ollama_model_name
                    self._llm_interface = get_llm_interface() 

                def __call__(self, input: List[str]) -> List[List[float]]:
                    embeddings = []
                    for text in input:
                        emb = self._llm_interface.generate_embedding(text=text, model=self._ollama_model_name)
                        if emb:
                            embeddings.append(emb)
                        else:
                            # Handle embedding failure - maybe return zero vector or raise error
                            logger.error(f"Failed to generate embedding for text: '{text[:50]}...'")
                            # Using a zero vector of expected dimensionality might be problematic
                            # For now, let's skip adding this document if embedding fails, requires handling upstream
                            # Or raise an exception
                            raise ValueError(f"Embedding generation failed for input: {text[:50]}...")
                            # embeddings.append([0.0] * 768) # Example dim, needs actual model dim
                    if len(embeddings) != len(input):
                         logger.warning(f"Could not generate embeddings for all {len(input)} inputs. Only {len(embeddings)} generated.")
                         # This indicates potential data loss in indexing
                    return embeddings

            _embedding_function = OllamaEmbeddingFunction(ollama_embedding_model)
        else:
            # Option 2: Fallback to a default SentenceTransformer model (requires sentence-transformers installed)
            try:
                logger.info("Using default SentenceTransformer embedding function (all-MiniLM-L6-v2). Ensure 'sentence-transformers' is installed.")
                _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            except ImportError:
                logger.error("'sentence-transformers' library not found. Cannot use default embeddings. Please install it or configure an Ollama embedding model.")
                _embedding_function = None # Mark as unavailable
            except Exception as e:
                 logger.error(f"Failed to initialize SentenceTransformer embedding function: {e}", exc_info=True)
                 _embedding_function = None
        
        if _embedding_function is None:
             logger.critical("Failed to initialize any embedding function. Vector store operations will fail.")

    return _embedding_function

def get_chroma_client() -> chromadb.Client:
    """Initializes and returns the ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        persist_dir_relative = get_setting("vector_store.persist_directory", "data/chroma_db")
        persist_directory = os.path.join(PROJECT_ROOT, persist_dir_relative)
        
        if not os.path.exists(persist_directory):
            logger.info(f"Creating ChromaDB persistence directory: {persist_directory}")
            try:
                os.makedirs(persist_directory)
            except OSError as e:
                logger.error(f"Failed to create ChromaDB directory {persist_directory}: {e}", exc_info=True)
                raise

        logger.info(f"Initializing ChromaDB client. Persistence directory: {persist_directory}")
        
        try:
            # Chroma settings can be adjusted here if needed (e.g., for hostname/port if running server mode)
            chroma_settings = ChromaSettings(
                persist_directory=persist_directory,
                is_persistent=True,
                # Example: anonymized_telemetry=False # Disable telemetry if desired
            )
            _chroma_client = chromadb.Client(settings=chroma_settings)
            logger.info("ChromaDB client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            raise
    return _chroma_client

# --- Vector Store Interface --- 

class VectorStoreInterface:
    """Provides methods for interacting with the vector store (ChromaDB)."""

    def __init__(self):
        self.client = get_chroma_client()
        self.embedding_function = get_embedding_function()
        self.default_collection_name = get_setting("vector_store.default_collection", "sutazai_main_memory")
        logger.info(f"VectorStoreInterface initialized. Default collection: '{self.default_collection_name}'")

    def _get_or_create_collection(self, collection_name: Optional[str] = None) -> Optional[chromadb.Collection]:
        """Gets or creates a ChromaDB collection."""
        name = collection_name or self.default_collection_name
        if not self.embedding_function:
            logger.error(f"Cannot get/create collection '{name}': No embedding function available.")
            return None
        try:
            logger.debug(f"Getting or creating collection: {name}")
            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function
            )
            return collection
        except Exception as e:
            logger.error(f"Failed to get or create ChromaDB collection '{name}': {e}", exc_info=True)
            return None

    def add_documents(self, 
                      documents: List[str], 
                      metadatas: Optional[List[Dict[str, Any]]] = None, 
                      ids: Optional[List[str]] = None,
                      collection_name: Optional[str] = None) -> bool:
        """Adds documents to the specified collection.

        Args:
            documents: A list of text documents to add.
            metadatas: Optional list of metadata dictionaries corresponding to the documents.
            ids: Optional list of unique IDs for the documents. If None, IDs are generated.
            collection_name: The name of the collection. Defaults to the default collection.

        Returns:
            True if documents were added successfully (or partially), False otherwise.
        """
        collection = self._get_or_create_collection(collection_name)
        if not collection:
            return False
            
        if not documents:
            logger.warning(f"No documents provided to add to collection '{collection.name}'.")
            return True # No action needed

        # Ensure IDs are provided or generate them
        if ids is None:
            ids = [f"doc_{i}_{hash(doc[:50])}" for i, doc in enumerate(documents)] # Simple unique ID generation
        elif len(ids) != len(documents):
             logger.error("Number of IDs does not match number of documents.")
             return False

        # Ensure metadatas list matches documents length if provided
        if metadatas is None:
            metadatas = [{}] * len(documents) # Provide empty dicts if none
        elif len(metadatas) != len(documents):
             logger.error("Number of metadatas does not match number of documents.")
             return False

        logger.info(f"Adding {len(documents)} documents to collection '{collection.name}'")
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.debug(f"Successfully added documents to '{collection.name}'")
            # Consider calling self.client.persist() here if immediate persistence is critical, 
            # though ChromaDB might handle it automatically depending on config/version.
            return True
        except ValueError as e:
            # Catch embedding generation errors if OllamaEmbeddingFunction raises them
            logger.error(f"ValueError during document addition (likely embedding failure): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Failed to add documents to collection '{collection.name}': {e}", exc_info=True)
            return False

    def query(self, 
              query_texts: List[str],
              n_results: int = 3,
              where: Optional[Dict[str, Any]] = None, # For metadata filtering
              where_document: Optional[Dict[str, Any]] = None, # For document content filtering
              include: List[str] = ['documents', 'metadatas', 'distances'], # Default includes
              collection_name: Optional[str] = None
              ) -> Optional[Dict[str, List[Any]]]:
        """Queries the collection for documents similar to the query texts.

        Args:
            query_texts: A list of query strings.
            n_results: The number of results to return per query.
            where: Optional metadata filter dictionary (e.g., {"source": "wiki"}).
            where_document: Optional document content filter dictionary (e.g., {"$contains": "search term"}).
            include: List of fields to include in the result (e.g., ['documents', 'metadatas', 'distances']).
            collection_name: The name of the collection. Defaults to the default collection.
        
        Returns:
            A dictionary containing the query results or None if an error occurs.
        """
        collection = self._get_or_create_collection(collection_name)
        if not collection:
            return None

        logger.info(f"Querying collection '{collection.name}' with {len(query_texts)} queries, n_results={n_results}, where={where}, where_document={where_document}")
        try:
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            logger.debug(f"Query successful. Retrieved results for {len(query_texts)} queries.")
            return results
        except Exception as e:
            logger.error(f"Failed to query collection '{collection.name}': {e}", exc_info=True)
            return None

    def get(self, 
            ids: Optional[List[str]] = None, 
            where: Optional[Dict[str, Any]] = None, 
            where_document: Optional[Dict[str, Any]] = None, 
            limit: Optional[int] = None, 
            offset: Optional[int] = None, 
            include: List[str] = ['documents', 'metadatas'], # Default includes for get
            collection_name: Optional[str] = None
            ) -> Optional[Dict[str, Any]]:
        """Retrieves documents from the collection by ID or filters.

        Args:
            ids: Optional list of document IDs to retrieve.
            where: Optional metadata filter dictionary.
            where_document: Optional document content filter dictionary.
            limit: Optional maximum number of documents to return.
            offset: Optional offset for pagination.
            include: List of fields to include in the result (e.g., ['documents', 'metadatas']).
            collection_name: The name of the collection. Defaults to the default collection.

        Returns:
            A dictionary containing the retrieved documents and metadata, or None if an error occurs.
            Example ChromaDB get result format:
            {
                'ids': [id1, id2], 
                'metadatas': [m1, m2],
                'documents': [doc1, doc2],
                'embeddings': None # Unless included
            }
        """
        collection = self._get_or_create_collection(collection_name)
        if not collection:
            return None
            
        if not ids and not where and not where_document:
            logger.warning("'get' called without specific ids or filters. This might retrieve the entire collection (up to limit).")
            # Potentially add a safeguard here to prevent accidental large fetches
            # if collection.count() > (limit or 1000): 
            #     return {"error": "Please provide IDs or filters to avoid fetching too many documents."}

        logger.info(f"Getting documents from collection '{collection.name}' with ids={ids}, where={where}, limit={limit}")
        try:
            results = collection.get(
                ids=ids,
                where=where,
                where_document=where_document,
                limit=limit,
                offset=offset,
                include=include
            )
            logger.debug(f"Get successful. Retrieved {len(results.get('ids', []))} documents.")
            return results
        except Exception as e:
            logger.error(f"Failed to get documents from collection '{collection.name}': {e}", exc_info=True)
            return None

    def delete_documents(self, ids: List[str], collection_name: Optional[str] = None) -> bool:
        """Deletes documents from the collection by their IDs."""
        collection = self._get_or_create_collection(collection_name)
        if not collection:
            return False
        
        if not ids:
            logger.warning("No IDs provided for deletion.")
            return True
            
        logger.info(f"Deleting {len(ids)} documents from collection '{collection.name}'")
        try:
            collection.delete(ids=ids)
            logger.debug(f"Successfully deleted documents with IDs: {ids}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from collection '{collection.name}': {e}", exc_info=True)
            return False
            
    def list_collections(self) -> List[str]:
        """Lists all available collections."""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
             logger.error(f"Failed to list collections: {e}", exc_info=True)
             return []

    def count_documents(self, collection_name: Optional[str] = None) -> Optional[int]:
        """Counts the number of documents in the collection."""
        collection = self._get_or_create_collection(collection_name)
        if not collection:
            return None
        try:
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to count documents in collection '{collection.name}': {e}", exc_info=True)
            return None

# --- Global Vector Store Instance --- 
_vector_store_interface: Optional[VectorStoreInterface] = None

def get_vector_store() -> VectorStoreInterface:
    """Returns a singleton instance of the VectorStoreInterface."""
    global _vector_store_interface
    if _vector_store_interface is None:
        try:
            _vector_store_interface = VectorStoreInterface()
        except Exception as e:
            # Catch initialization errors (ChromaDB client, embedding function)
            logger.critical(f"VectorStoreInterface could not be initialized: {e}", exc_info=True)
            _vector_store_interface = None
            raise # Re-raise to signal failure
    
    if _vector_store_interface is None:
        raise RuntimeError("VectorStoreInterface initialization failed previously.")
        
    return _vector_store_interface

# Example Usage:
# if __name__ == '__main__':
#     try:
#         vector_store = get_vector_store()
#         print("Available collections:", vector_store.list_collections())

#         # Add some data
#         docs = ["This is document one", "This is document two about AI", "A third document mentioning ChromaDB"]
#         metadatas = [{"source": "manual"}, {"source": "manual", "topic": "AI"}, {"source": "test"}]
#         ids = ["doc1", "doc2", "doc3"]
#         added = vector_store.add_documents(documents=docs, metadatas=metadatas, ids=ids)
#         if added:
#             print("Documents added successfully.")

#             # Query the data
#             query = "Tell me about AI and ChromaDB"
#             results = vector_store.query(query_texts=[query], n_results=2)
#             if results:
#                 print(f"\nQuery results for: '{query}'")
#                 for i, doc in enumerate(results['documents'][0]):
#                     print(f"  - Doc: {doc}")
#                     print(f"    Meta: {results['metadatas'][0][i]}")
#                     print(f"    Dist: {results['distances'][0][i]:.4f}")
            
#             # Query with filter
#             results_filtered = vector_store.query(query_texts=[query], n_results=2, where={"topic": "AI"})
#             if results_filtered:
#                 print(f"\nFiltered query results (topic=AI):")
#                 for i, doc in enumerate(results_filtered['documents'][0]):
#                      print(f"  - Doc: {doc}")

#             # Delete a document
#             # deleted = vector_store.delete_documents(ids=["doc3"])
#             # print(f"\nDeletion status for doc3: {deleted}")

#     except Exception as e:
#         print(f"An error occurred: {e}") 