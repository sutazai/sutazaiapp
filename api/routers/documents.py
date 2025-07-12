from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os
import shutil
from pathlib import Path
import mimetypes
from uuid import uuid4

from memory import vector_memory
from api.auth import get_current_user, require_admin
from api.database import db_manager
from config import config

logger = logging.getLogger(__name__)
router = APIRouter()

# Ensure upload directory exists
UPLOADS_DIR = Path(config.storage.uploads_path)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/")
async def list_documents(
    file_type: Optional[str] = None,
    processed: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """List uploaded documents with optional filtering."""
    try:
        # Get documents from database
        documents = await db_manager.get_documents(
            file_type=file_type,
            processed=processed,
            limit=limit,
            offset=offset
        )
        
        await db_manager.log_system_event(
            "info", "documents", "Listed documents",
            {"user": current_user.get("username"), "count": len(documents)}
        )
        
        return {
            "documents": documents,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}")
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get details about a specific document."""
    try:
        document = await db_manager.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    auto_process: bool = Form(True),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload a new document."""
    try:
        # Validate file type
        allowed_types = {'.txt', '.pdf', '.docx', '.doc', '.md', '.json', '.csv'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {', '.join(allowed_types)}"
            )
        
        # Generate unique filename
        document_id = str(uuid4())
        safe_filename = f"{document_id}_{file.filename}"
        file_path = UPLOADS_DIR / safe_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file info
        file_size = file_path.stat().st_size
        mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        
        # Store in database
        document_data = {
            "id": document_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_type": file_extension,
            "file_size": file_size,
            "mime_type": mime_type,
            "description": description,
            "uploaded_by": current_user.get("username"),
            "uploaded_at": datetime.utcnow(),
            "processed": False,
            "embedding_generated": False
        }
        
        await db_manager.store_document(document_data)
        
        # Auto-process if requested
        if auto_process:
            try:
                await process_document_content(document_id, file_path, file_extension)
                document_data["processed"] = True
                await db_manager.update_document(document_id, {"processed": True})
            except Exception as e:
                logger.warning(f"Auto-processing failed for {document_id}: {e}")
        
        await db_manager.log_system_event(
            "info", "documents", "Document uploaded",
            {"user": current_user.get("username"), "document_id": document_id, "filename": file.filename}
        )
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "file_size": file_size,
            "status": "uploaded",
            "processed": document_data["processed"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{document_id}/process")
async def process_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Process a document to extract content and generate embeddings."""
    try:
        document = await db_manager.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        file_path = Path(document["file_path"])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document file not found on disk")
        
        # Process document
        content = await process_document_content(document_id, file_path, document["file_type"])
        
        # Update database
        await db_manager.update_document(document_id, {
            "processed": True,
            "content": content[:10000],  # Store first 10k chars
            "processed_at": datetime.utcnow()
        })
        
        await db_manager.log_system_event(
            "info", "documents", "Document processed",
            {"user": current_user.get("username"), "document_id": document_id}
        )
        
        return {
            "document_id": document_id,
            "status": "processed",
            "content_length": len(content),
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{document_id}/embed")
async def generate_embeddings(
    document_id: str,
    chunk_size: int = 1000,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Generate embeddings for a document and store in vector database."""
    try:
        document = await db_manager.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        if not document.get("processed"):
            raise HTTPException(status_code=400, detail="Document must be processed first")
        
        # Get full content
        file_path = Path(document["file_path"])
        if file_path.exists():
            content = await read_document_content(file_path, document["file_type"])
        else:
            content = document.get("content", "")
        
        if not content:
            raise HTTPException(status_code=400, detail="No content available for embedding")
        
        # Generate embeddings
        embedding_count = await vector_memory.add_document(
            document_id=document_id,
            content=content,
            metadata={
                "filename": document["filename"],
                "file_type": document["file_type"],
                "uploaded_by": document.get("uploaded_by"),
                "uploaded_at": document.get("uploaded_at", datetime.utcnow()).isoformat()
            },
            chunk_size=chunk_size
        )
        
        # Update database
        await db_manager.update_document(document_id, {
            "embedding_generated": True,
            "embedding_count": embedding_count,
            "embedded_at": datetime.utcnow()
        })
        
        await db_manager.log_system_event(
            "info", "documents", "Embeddings generated",
            {"user": current_user.get("username"), "document_id": document_id, "embedding_count": embedding_count}
        )
        
        return {
            "document_id": document_id,
            "status": "embedded",
            "embedding_count": embedding_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete a document and its associated data."""
    try:
        document = await db_manager.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Delete file from disk
        file_path = Path(document["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Delete from vector database
        if document.get("embedding_generated"):
            await vector_memory.delete_document(document_id)
        
        # Delete from database
        await db_manager.delete_document(document_id)
        
        await db_manager.log_system_event(
            "info", "documents", "Document deleted",
            {"user": current_user.get("username"), "document_id": document_id}
        )
        
        return {
            "document_id": document_id,
            "status": "deleted",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_documents(
    query: str,
    limit: int = 10,
    similarity_threshold: float = 0.7,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Search documents using vector similarity."""
    try:
        results = await vector_memory.search(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        await db_manager.log_system_event(
            "info", "documents", "Document search",
            {"user": current_user.get("username"), "query": query[:100], "results_count": len(results)}
        )
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def process_document_content(document_id: str, file_path: Path, file_type: str) -> str:
    """Process document content based on file type."""
    try:
        if file_type in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_type == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)
                return json.dumps(data, indent=2)
        elif file_type == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_type in ['.pdf', '.docx', '.doc']:
            # For now, return placeholder - would need proper PDF/DOCX parsers
            return f"[{file_type.upper()} file - content extraction not yet implemented]"
        else:
            return "[Unsupported file type for content extraction]"
    except Exception as e:
        logger.error(f"Error processing content for {document_id}: {e}")
        return f"[Error processing content: {str(e)}]"

async def read_document_content(file_path: Path, file_type: str) -> str:
    """Read document content for embedding generation."""
    return await process_document_content(str(file_path), file_path, file_type)
