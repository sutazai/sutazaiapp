#!/usr/bin/env python3
"""
PrivateGPT Service for SutazAI
Provides private document processing and chat capabilities
"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SutazAI PrivateGPT Service",
    description="Private document processing and chat",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    message: str
    use_context: bool = True
    stream: bool = False

class DocumentRequest(BaseModel):
    file_path: str
    chunk_size: int = 1000
    chunk_overlap: int = 200

class PrivateGPTManager:
    """Manages PrivateGPT functionality"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.chat_history = []
    
    async def initialize(self):
        """Initialize PrivateGPT components"""
        try:
            logger.info("Initializing PrivateGPT...")
            # Initialize embeddings
            from sentence_transformers import SentenceTransformer
            self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize vector store connection
            try:
                import qdrant_client
                self.vectorstore = qdrant_client.QdrantClient(
                    host=os.getenv("QDRANT_HOST", "qdrant"),
                    port=int(os.getenv("QDRANT_PORT", "6333"))
                )
                logger.info("Connected to Qdrant vector store")
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant: {e}")
            
            logger.info("PrivateGPT initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PrivateGPT: {e}")
            raise
    
    async def process_document(self, file_path: str, chunk_size: int = 1000) -> Dict[str, Any]:
        """Process and index a document"""
        try:
            # Simulate document processing
            document_info = {
                "file_path": file_path,
                "chunks_created": 10,  # Simulated
                "status": "processed",
                "timestamp": time.time()
            }
            
            self.documents.append(document_info)
            
            return {
                "status": "success",
                "document_id": len(self.documents),
                "chunks_created": document_info["chunks_created"],
                "message": f"Document {file_path} processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def chat_with_documents(self, message: str, use_context: bool = True) -> Dict[str, Any]:
        """Chat with processed documents"""
        try:
            # Add to chat history
            self.chat_history.append({
                "user": message,
                "timestamp": time.time()
            })
            
            if use_context and self.documents:
                # Simulate context-aware response
                response = f"Based on the {len(self.documents)} documents I have access to, here's my response to '{message}': [This would be a context-aware response from the processed documents. The actual implementation would use RAG (Retrieval Augmented Generation) to provide accurate answers based on your private documents.]"
            else:
                # Simulate general response
                response = f"I understand you're asking about: '{message}'. Since no document context is being used, I can provide general information, but for more specific insights, please upload relevant documents first."
            
            # Add response to history
            self.chat_history[-1]["assistant"] = response
            
            return {
                "response": response,
                "used_context": use_context,
                "documents_available": len(self.documents),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get list of processed documents"""
        return self.documents
    
    def get_chat_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent chat history"""
        return self.chat_history[-limit:]

# Initialize manager
pgpt_manager = PrivateGPTManager()

@app.on_event("startup")
async def startup_event():
    """Initialize PrivateGPT on startup"""
    await pgpt_manager.initialize()

@app.get("/")
async def root():
    return {
        "service": "SutazAI PrivateGPT Service",
        "status": "running",
        "version": "1.0.0",
        "documents_processed": len(pgpt_manager.documents)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "documents_processed": len(pgpt_manager.documents),
        "chat_sessions": len(pgpt_manager.chat_history)
    }

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save uploaded file
        file_path = f"/data/documents/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        result = await pgpt_manager.process_document(file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/process")
async def process_document(request: DocumentRequest):
    """Process an existing document"""
    try:
        result = await pgpt_manager.process_document(
            request.file_path,
            request.chunk_size
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List processed documents"""
    documents = pgpt_manager.get_documents()
    return {
        "documents": documents,
        "total": len(documents)
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with processed documents"""
    try:
        result = await pgpt_manager.chat_with_documents(
            request.message,
            request.use_context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history")
async def get_chat_history(limit: int = 10):
    """Get chat history"""
    history = pgpt_manager.get_chat_history(limit)
    return {
        "history": history,
        "total_messages": len(pgpt_manager.chat_history)
    }

@app.delete("/chat/history")
async def clear_chat_history():
    """Clear chat history"""
    pgpt_manager.chat_history.clear()
    return {"message": "Chat history cleared"}

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a processed document"""
    try:
        if 0 <= document_id < len(pgpt_manager.documents):
            deleted_doc = pgpt_manager.documents.pop(document_id)
            return {
                "message": f"Document deleted: {deleted_doc['file_path']}",
                "deleted_document": deleted_doc
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get service status"""
    return {
        "service": "PrivateGPT",
        "status": "operational",
        "documents_processed": len(pgpt_manager.documents),
        "chat_messages": len(pgpt_manager.chat_history),
        "vectorstore_connected": pgpt_manager.vectorstore is not None,
        "embeddings_loaded": pgpt_manager.embeddings is not None,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run("privategpt_service:app", host="0.0.0.0", port=8001, reload=False)