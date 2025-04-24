import logging
import os
from typing import Optional, Dict, Any, List
import uuid
from pathlib import Path
from datetime import datetime

# --- Add required imports for parsing and chunking ---
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ----------------------------------------------------

# Assume necessary imports from other modules
from sutazai_agi.core.config_loader import get_setting
from sutazai_agi.models.llm_interface import get_llm_interface, LLMInterface
from sutazai_agi.memory.vector_store import get_vector_store, VectorStoreInterface
from sutazai_agi.agents.agent_manager import AgentManager
from sutazai_agi.tools.search_local_docs import search_local_docs

logger = logging.getLogger(__name__)

# Define storage path based on workspace settings
UPLOAD_DIR_NAME = "document_uploads"
WORKSPACE_PATH = Path(get_setting("agent_workspace", "/opt/v3/workspace"))
UPLOAD_PATH = WORKSPACE_PATH / UPLOAD_DIR_NAME

# Ensure upload directory exists
try:
    UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured document upload directory exists: {UPLOAD_PATH}")
except OSError as e:
    logger.error(f"Failed to create document upload directory {UPLOAD_PATH}: {e}. Document processing may fail.")
    UPLOAD_PATH = None # Indicate failure

# Define chunking parameters (consider making these configurable in settings.yaml)
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400 # 10% overlap

class DocumentService:
    """Handles document uploading, processing, indexing, and querying."""

    def __init__(self, vector_store: VectorStoreInterface, agent_manager: AgentManager):
        if UPLOAD_PATH is None:
             raise RuntimeError("Document upload path is not configured or could not be created.")
        self.vector_store = vector_store
        self.agent_manager = agent_manager
        self.upload_dir = os.path.abspath(get_setting("document_processing.upload_directory", "./workspace/document_uploads"))
        self.allowed_extensions = get_setting("document_processing.allowed_upload_extensions", [".pdf", ".docx", ".txt", ".md"])
        self.collection_name = get_setting("document_processing.vector_collection", "sutazai_documents")
        
        # Ensure upload directory exists
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False, # Keep default separators
        )
        logger.info(f"DocumentService initialized. Upload dir: {self.upload_dir}, Allowed extensions: {self.allowed_extensions}")

    async def upload_and_index_document(self, file: UploadFile) -> Dict[str, Any]:
        """Saves an uploaded document, extracts text, chunks, and indexes it."""
        logger.info(f"Received request to upload and index document: {file.filename}")
        
        if not self.upload_dir:
            return {"status": "error", "message": "Upload directory not available."}

        doc_id = f"doc_{uuid.uuid4()}"
        file_path = os.path.join(self.upload_dir, file.filename)

        try:
            # 1. Save the file
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            logger.info(f"Saved uploaded file to: {file_path}")

            # 2. Extract text using unstructured
            logger.info(f"Attempting to extract text from {file_path}")
            try:
                # TODO: Future - Enhance partitioning to handle images/diagrams (Multimodal Input)
                # Could involve checking file type and using different unstructured strategies
                # or passing to a vision model tool.
                elements = partition(filename=file_path)
                text_content = "\n\n".join([el.text for el in elements if hasattr(el, 'text')])
                if not text_content.strip():
                     logger.warning(f"Unstructured extracted no text content from {file.filename}. Indexing skipped.")
                     # Clean up saved file? Or keep it for reference?
                     # Keep file for now, but return success indicating no indexing.
                     return {"status": "success", "message": f"Document '{file.filename}' saved, but no text content was extracted for indexing.", "doc_id": doc_id, "num_chunks": 0}
                logger.info(f"Extracted text from {file.filename}. Length: {len(text_content)}")
            except Exception as parse_err:
                 logger.error(f"Failed to parse {file.filename} with unstructured: {parse_err}", exc_info=True)
                 # Clean up saved file
                 if os.path.exists(file_path):
                     try: os.remove(file_path)
                     except OSError: pass
                 return {"status": "error", "message": f"Failed to parse document: {parse_err}"}
            
            # 3. Chunk the text using LangChain splitter
            logger.info(f"Chunking extracted text with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
            chunks = self.text_splitter.split_text(text_content)
            logger.info(f"Split text into {len(chunks)} chunks.")

            # 4. Embed and Index the chunks
            if chunks:
                metadata = {"source": file.filename, "doc_id": doc_id, "original_path": file_path, "upload_time": datetime.now().isoformat()}
                metadatas = [dict(metadata, chunk_index=i) for i in range(len(chunks))]
                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                
                # Use vector_store's add method 
                added_ok = self.vector_store.add_documents(
                    documents=chunks, 
                    ids=chunk_ids, 
                    metadatas=metadatas, 
                    collection_name=self.collection_name
                )
                
                if added_ok:
                    logger.info(f"Successfully indexed {len(chunks)} chunks for document {doc_id} ({file.filename}) into collection '{self.collection_name}'.")
                    logger.info(f"Successfully indexed {len(chunks)} chunks for document {doc_id} ({filename}) into collection '{self.collection_name}'.")
                    return {"status": "success", "message": f"Document '{filename}' processed and indexed.", "doc_id": doc_id, "num_chunks": len(chunks)}
                else:
                     # Error during embedding/adding handled by add_documents logging
                     # Clean up saved file? Maybe not, as indexing failed but file/parsing was ok.
                     return {"status": "error", "message": "Failed to add document chunks to vector store. Check logs for details."}
            else:
                 # This case should be handled by the earlier check for empty extracted_text
                 logger.warning(f"No chunks generated from {filename} despite non-empty text extraction. Indexing skipped.")
                 return {"status": "success", "message": f"Document '{filename}' saved and parsed, but no indexable chunks were generated.", "doc_id": doc_id, "num_chunks": 0}

        except Exception as e:
            logger.error(f"Unexpected error during document processing for {filename}: {e}", exc_info=True)
            if file_path.exists():
                 try: file_path.unlink()
                 except OSError: pass
            return {"status": "error", "message": f"Failed to process document: {e}"}

    async def analyze_document(self, document_id: str, analysis_prompt: str, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyzes an already indexed document using an agent."""
        # TODO: Agent Selection - Choose agent based on analysis type (summary, Q&A, financial?) or use default.
        # TODO: Context Retrieval - Fetch relevant chunks from vector store using document_id as filter.
        # TODO: Prompt Construction - Combine retrieved chunks with analysis_prompt for the agent.
        logger.info(f"Analyzing document '{document_id}' with prompt: {analysis_prompt[:50]}...")
        
        # Placeholder: Retrieve full text (inefficient) or use AgentManager with a query
        # Example using AgentManager directly (needs agent configured for document Q&A):
        selected_agent = agent_name or "LangChain Chat Agent" # Example default
        
        # Construct a user input that includes the document context
        # This requires fetching relevant chunks first
        try:
            vector_store_results = await self.vector_store.query(
                query_texts=[analysis_prompt], 
                n_results=5, # Get top 5 relevant chunks
                where={"source": document_id}, # Filter by document source
                collection_name=self.collection_name,
                include=["documents"] # Only need the document content
            )
            
            context_str = "\n---\n".join(vector_store_results["documents"][0]) if vector_store_results and vector_store_results["documents"] else "No relevant context found in document."
            
        except Exception as e:
             logger.error(f"Failed to retrieve context for document '{document_id}': {e}", exc_info=True)
             context_str = "Error retrieving context from document."
        
        # Combine prompt and context
        combined_input = f"Based on the following context from the document '{document_id}', please respond to the request:\n\nContext:\n{context_str}\n\nRequest: {analysis_prompt}"
        
        result = await self.agent_manager.execute_task(
            agent_name=selected_agent,
            user_input=combined_input 
        )
        return result

    async def query_documents(self, query: str, top_k: int = 5, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Performs a vector search across indexed documents."""
        # This largely duplicates the search_local_docs tool logic
        # Consider refactoring or using the tool directly if appropriate
        logger.info(f"Querying documents with: {query[:50]}...")
        try:
            search_result = await search_local_docs(query=query, top_k=top_k, collection_name=collection_name or self.collection_name)
            return search_result # Return the dict from the tool function
        except Exception as e:
            logger.error(f"Error during document query: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to query documents: {e}"}

# --- Dependency Injection ---

_document_service: Optional[DocumentService] = None

def get_document_service() -> DocumentService:
    """Provides a singleton instance of the DocumentService."""
    global _document_service
    if _document_service is None:
        logger.info("Creating DocumentService instance.")
        try:
            # Ensure dependencies are available
            llm_interface = get_llm_interface()
            vector_store = get_vector_store()
            _document_service = DocumentService(llm_interface, vector_store)
        except Exception as e:
            logger.critical(f"Failed to initialize DocumentService: {e}", exc_info=True)
            _document_service = None # Indicate failure
            raise RuntimeError(f"Could not initialize DocumentService: {e}") from e
            
    if _document_service is None:
         raise RuntimeError("DocumentService initialization failed previously.")
         
    return _document_service 