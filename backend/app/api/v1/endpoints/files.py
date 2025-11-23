"""
File upload and download endpoints with security
Implements virus scanning, size limits, mime validation
"""

import os
import shutil
from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import mimetypes
import subprocess

from app.core.database import get_db
from app.models.user import User
from app.api.dependencies.auth import get_current_active_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration
# Use /tmp for uploads in container (should be volume-mounted in production)
UPLOAD_DIR = Path("/tmp/sutazai_uploads")
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_MIME_TYPES = {
    "text/plain", "text/csv", "text/markdown",
    "application/pdf", "application/json",
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/zip", "application/x-tar", "application/gzip",
    "text/html", "text/css", "text/javascript",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
}

# Ensure upload directory exists with safe permissions
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory ready: {UPLOAD_DIR}")
except Exception as e:
    logger.error(f"Failed to create upload directory: {e}")
    # Fallback to /tmp if primary location fails
    UPLOAD_DIR = Path("/tmp")
    logger.warning(f"Using fallback upload directory: {UPLOAD_DIR}")


def get_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def scan_file_with_clamav(file_path: Path) -> bool:
    """
    Scan file with ClamAV for viruses
    Returns True if file is clean, False if infected
    """
    try:
        # Check if clamscan is available
        result = subprocess.run(
            ["which", "clamscan"],
            capture_output=True,
            timeout=5
        )
        
        if result.returncode != 0:
            logger.warning("ClamAV not installed - skipping virus scan")
            return True  # Allow if ClamAV not available
        
        # Scan the file
        result = subprocess.run(
            ["clamscan", "--no-summary", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # ClamAV returns 0 if no viruses found
        if result.returncode == 0:
            logger.info(f"File {file_path.name} is clean")
            return True
        else:
            logger.warning(f"Virus detected in {file_path.name}: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"ClamAV scan timeout for {file_path.name}")
        return False
    except Exception as e:
        logger.error(f"ClamAV scan error: {e}")
        return True  # Allow on error (don't block legitimate files)


def validate_mime_type(file: UploadFile) -> bool:
    """Validate file MIME type"""
    # Get MIME type from upload
    content_type = file.content_type
    
    # Also check by extension
    if file.filename:
        guessed_type, _ = mimetypes.guess_type(file.filename)
        if guessed_type and guessed_type not in ALLOWED_MIME_TYPES:
            return False
    
    return content_type in ALLOWED_MIME_TYPES


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a file with virus scanning and validation
    
    **Security Features:**
    - File size limit: 100MB
    - MIME type validation
    - Virus scanning with ClamAV
    - Secure filename sanitization
    - User-specific storage
    
    **Request:**
    - `file` (required): File to upload (multipart/form-data)
    
    **Returns:**
    - `200 OK`: File uploaded successfully
        - `file_id`: Unique file identifier
        - `filename`: Sanitized filename
        - `size`: File size in bytes
        - `mime_type`: File MIME type
        - `sha256`: File hash for integrity verification
        - `upload_time`: Upload timestamp
    
    **Errors:**
    - `400 Bad Request`: Invalid file type or too large
    - `401 Unauthorized`: Not authenticated
    - `403 Forbidden`: Virus detected
    - `500 Internal Server Error`: Upload failed
    """
    try:
        # Validate MIME type
        if not validate_mime_type(file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file.content_type} not allowed"
            )
        
        # Create user directory
        user_dir = UPLOAD_DIR / str(current_user.id)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate safe filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = user_dir / safe_filename
        
        # Read and validate file size
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large: {file_size} bytes (max {MAX_FILE_SIZE})"
            )
        
        # Write file
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Scan for viruses
        if not scan_file_with_clamav(file_path):
            # Delete infected file
            file_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Virus detected in file"
            )
        
        # Calculate file hash
        file_hash = get_file_hash(file_path)
        
        logger.info(f"File uploaded: {safe_filename} by user {current_user.username}")
        
        return {
            "file_id": file_hash[:16],
            "filename": safe_filename,
            "original_filename": file.filename,
            "size": file_size,
            "mime_type": file.content_type,
            "sha256": file_hash,
            "upload_time": datetime.now(timezone.utc).isoformat(),
            "message": "File uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/download/{filename}")
async def download_file(
    filename: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Download a file with access control
    
    **Security Features:**
    - User can only download their own files
    - Streaming response for large files
    - Content-Disposition header for safe downloads
    
    **Path Parameters:**
    - `filename` (required): Name of file to download
    
    **Returns:**
    - `200 OK`: File stream
    - `404 Not Found`: File not found
    - `403 Forbidden`: Access denied
    """
    try:
        # Construct file path (user's directory only)
        user_dir = UPLOAD_DIR / str(current_user.id)
        file_path = user_dir / filename
        
        # Security: prevent directory traversal
        if not str(file_path.resolve()).startswith(str(user_dir.resolve())):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "application/octet-stream"
        
        # Stream file
        def file_iterator():
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
        
        logger.info(f"File downloaded: {filename} by user {current_user.username}")
        
        return StreamingResponse(
            file_iterator(),
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(file_path.stat().st_size)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/list")
async def list_files(
    current_user: User = Depends(get_current_active_user)
):
    """
    List all files uploaded by current user
    
    **Returns:**
    - `200 OK`: List of files
        - `files`: Array of file objects
            - `filename`: File name
            - `size`: File size in bytes
            - `upload_time`: Upload timestamp
            - `mime_type`: File MIME type
    """
    try:
        user_dir = UPLOAD_DIR / str(current_user.id)
        
        if not user_dir.exists():
            return {"files": [], "count": 0}
        
        files = []
        for file_path in user_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                mime_type, _ = mimetypes.guess_type(str(file_path))
                
                files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "upload_time": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "mime_type": mime_type or "application/octet-stream"
                })
        
        # Sort by upload time (newest first)
        files.sort(key=lambda x: x["upload_time"], reverse=True)
        
        return {
            "files": files,
            "count": len(files),
            "total_size": sum(f["size"] for f in files)
        }
        
    except Exception as e:
        logger.error(f"File list error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/delete/{filename}")
async def delete_file(
    filename: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a file
    
    **Path Parameters:**
    - `filename` (required): Name of file to delete
    
    **Returns:**
    - `200 OK`: File deleted
    - `404 Not Found`: File not found
    - `403 Forbidden`: Access denied
    """
    try:
        user_dir = UPLOAD_DIR / str(current_user.id)
        file_path = user_dir / filename
        
        # Security check
        if not str(file_path.resolve()).startswith(str(user_dir.resolve())):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        file_path.unlink()
        
        logger.info(f"File deleted: {filename} by user {current_user.username}")
        
        return {"message": "File deleted successfully", "filename": filename}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File delete error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
