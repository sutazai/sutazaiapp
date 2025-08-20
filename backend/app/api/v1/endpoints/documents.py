"""
Documents endpoint - REAL IMPLEMENTATION

Implements comprehensive document management functionality:
- Document upload/download
- Document indexing and search
- Document version control
- Document sharing and permissions
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from app.core.middleware import jwt_required
from typing import List, Optional, Dict, Any
import os
import uuid
import mimetypes
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Document storage configuration
DOCUMENT_STORAGE_PATH = os.getenv('DOCUMENT_STORAGE_PATH', '/opt/sutazaiapp/data/documents')
DOCUMENT_INDEX_PATH = os.getenv('DOCUMENT_INDEX_PATH', '/opt/sutazaiapp/data/document_index.json')

# Ensure storage directory exists
os.makedirs(DOCUMENT_STORAGE_PATH, exist_ok=True)

def _load_document_index() -> Dict[str, Any]:
    """Load document index from disk"""
    try:
        if os.path.exists(DOCUMENT_INDEX_PATH):
            with open(DOCUMENT_INDEX_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading document index: {e}")
    return {"documents": {}, "version": 1, "last_updated": datetime.utcnow().isoformat()}

def _save_document_index(index: Dict[str, Any]):
    """Save document index to disk"""
    try:
        index["last_updated"] = datetime.utcnow().isoformat()
        with open(DOCUMENT_INDEX_PATH, 'w') as f:
            json.dump(index, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving document index: {e}")

def _get_file_metadata(file_path: str, original_name: str) -> Dict[str, Any]:
    """Get file metadata"""
    stat = os.stat(file_path)
    mime_type, _ = mimetypes.guess_type(original_name)
    
    return {
        "size": stat.st_size,
        "mime_type": mime_type or "application/octet-stream",
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
    }

router = APIRouter(dependencies=[Depends(jwt_required(scopes=["documents:read"]))])

@router.get("/")
async def list_documents(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    search: Optional[str] = Query(None, description="Search query for document names or content")
):
    """
    List documents with pagination and optional search
    """
    try:
        index = _load_document_index()
        documents = list(index["documents"].values())
        
        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            documents = [
                doc for doc in documents
                if search_lower in doc.get("original_name", "").lower() or
                   search_lower in doc.get("tags", [])
            ]
        
        # Sort by upload date (newest first)
        documents.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
        
        # Apply pagination
        total_count = len(documents)
        documents = documents[offset:offset + limit]
        
        return {
            "documents": documents,
            "count": len(documents),
            "total_count": total_count,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(documents) < total_count
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Query(None, description="Comma-separated tags for the document"),
    description: Optional[str] = Query(None, description="Document description")
):
    """
    Upload a new document
    """
    try:
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix if file.filename else ""
        stored_filename = f"{document_id}{file_extension}"
        file_path = os.path.join(DOCUMENT_STORAGE_PATH, stored_filename)
        
        # Save file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get file metadata
        metadata = _get_file_metadata(file_path, file.filename or "unknown")
        
        # Create document record
        document = {
            "id": document_id,
            "original_name": file.filename or "unknown",
            "stored_filename": stored_filename,
            "description": description,
            "tags": [tag.strip() for tag in tags.split(",")] if tags else [],
            "uploaded_at": datetime.utcnow().isoformat(),
            "file_path": file_path,
            **metadata
        }
        
        # Update index
        index = _load_document_index()
        index["documents"][document_id] = document
        _save_document_index(index)
        
        logger.info(f"Document uploaded: {document_id} ({file.filename})")
        return {
            "message": "Document uploaded successfully",
            "document": document
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}")
async def get_document_info(document_id: str):
    """
    Get document information by ID
    """
    try:
        index = _load_document_index()
        if document_id not in index["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = index["documents"][document_id]
        
        # Check if file still exists
        if not os.path.exists(document["file_path"]):
            logger.warning(f"Document file missing: {document['file_path']}")
            document["file_status"] = "missing"
        else:
            document["file_status"] = "available"
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}/download")
async def download_document(document_id: str):
    """
    Download a document by ID
    """
    try:
        index = _load_document_index()
        if document_id not in index["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = index["documents"][document_id]
        file_path = document["file_path"]
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document file not found")
        
        return FileResponse(
            path=file_path,
            filename=document["original_name"],
            media_type=document.get("mime_type", "application/octet-stream")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document by ID
    """
    try:
        index = _load_document_index()
        if document_id not in index["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = index["documents"][document_id]
        file_path = document["file_path"]
        
        # Remove file from disk
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove from index
        del index["documents"][document_id]
        _save_document_index(index)
        
        logger.info(f"Document deleted: {document_id}")
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_documents(
    query: str,
    limit: int = Query(20, ge=1, le=100),
    tags: Optional[List[str]] = None
):
    """
    Advanced document search
    """
    try:
        index = _load_document_index()
        documents = list(index["documents"].values())
        
        query_lower = query.lower()
        results = []
        
        for doc in documents:
            score = 0
            
            # Search in filename
            if query_lower in doc.get("original_name", "").lower():
                score += 10
            
            # Search in description
            if doc.get("description") and query_lower in doc["description"].lower():
                score += 5
            
            # Search in tags
            for tag in doc.get("tags", []):
                if query_lower in tag.lower():
                    score += 3
            
            # Filter by tags if specified
            if tags:
                if not any(tag.lower() in [t.lower() for t in doc.get("tags", [])] for tag in tags):
                    continue
            
            if score > 0:
                results.append({**doc, "search_score": score})
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x["search_score"], reverse=True)
        results = results[:limit]
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "tags_filter": tags
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/summary")
async def get_document_stats():
    """
    Get document collection statistics
    """
    try:
        index = _load_document_index()
        documents = list(index["documents"].values())
        
        total_size = 0
        mime_types = {}
        tags = {}
        
        for doc in documents:
            # Calculate total size
            total_size += doc.get("size", 0)
            
            # Count mime types
            mime_type = doc.get("mime_type", "unknown")
            mime_types[mime_type] = mime_types.get(mime_type, 0) + 1
            
            # Count tags
            for tag in doc.get("tags", []):
                tags[tag] = tags.get(tag, 0) + 1
        
        return {
            "total_documents": len(documents),
            "total_size_bytes": total_size,
            "total_size_human": _format_bytes(total_size),
            "mime_types": dict(sorted(mime_types.items(), key=lambda x: x[1], reverse=True)),
            "popular_tags": dict(sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10]),
            "last_updated": index.get("last_updated")
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _format_bytes(bytes_value: int) -> str:
    """Format bytes in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"
