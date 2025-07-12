from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os
import shutil
from pathlib import Path
import mimetypes
from uuid import uuid4
import zipfile
import io

from api.auth import get_current_user, require_admin
from api.database import db_manager
from config import config

logger = logging.getLogger(__name__)
router = APIRouter()

# File storage directories
UPLOADS_DIR = Path(config.storage.uploads_path)
TEMP_DIR = Path(config.storage.temp_path)
REPORTS_DIR = Path(config.storage.reports_path)

# Ensure directories exist
for directory in [UPLOADS_DIR, TEMP_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@router.get("/")
async def list_files(
    file_type: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """List files accessible to the current user."""
    try:
        files_info = []
        
        # Scan upload directory
        for file_path in UPLOADS_DIR.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                file_info = {
                    "name": file_path.name,
                    "path": str(file_path.relative_to(UPLOADS_DIR)),
                    "size": stat.st_size,
                    "type": mimetypes.guess_type(str(file_path))[0] or "application/octet-stream",
                    "category": "upload",
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                files_info.append(file_info)
        
        # Scan reports directory
        for file_path in REPORTS_DIR.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                file_info = {
                    "name": file_path.name,
                    "path": str(file_path.relative_to(REPORTS_DIR)),
                    "size": stat.st_size,
                    "type": mimetypes.guess_type(str(file_path))[0] or "application/octet-stream",
                    "category": "report",
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                files_info.append(file_info)
        
        # Apply filters
        if file_type:
            files_info = [f for f in files_info if file_type.lower() in f["type"].lower()]
        if category:
            files_info = [f for f in files_info if f["category"] == category]
        
        # Apply pagination
        total = len(files_info)
        files_info = files_info[offset:offset + limit]
        
        await db_manager.log_system_event(
            "info", "files", "Listed files",
            {"user": current_user.get("username"), "count": len(files_info)}
        )
        
        return {
            "files": files_info,
            "total": total,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form("upload"),
    description: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload a file to the server."""
    try:
        # Validate file size (e.g., 100MB limit)
        max_size = 100 * 1024 * 1024  # 100MB
        if hasattr(file, 'size') and file.size > max_size:
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")
        
        # Determine upload directory based on category
        if category == "upload":
            upload_dir = UPLOADS_DIR
        elif category == "temp":
            upload_dir = TEMP_DIR
        elif category == "report":
            upload_dir = REPORTS_DIR
        else:
            upload_dir = UPLOADS_DIR
        
        # Generate unique filename
        file_id = str(uuid4())
        file_extension = Path(file.filename).suffix
        safe_filename = f"{file_id}_{file.filename}"
        file_path = upload_dir / safe_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file info
        stat = file_path.stat()
        file_info = {
            "id": file_id,
            "name": file.filename,
            "safe_name": safe_filename,
            "path": str(file_path),
            "relative_path": str(file_path.relative_to(upload_dir)),
            "size": stat.st_size,
            "type": mimetypes.guess_type(str(file_path))[0] or "application/octet-stream",
            "category": category,
            "description": description,
            "uploaded_by": current_user.get("username"),
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        await db_manager.log_system_event(
            "info", "files", "File uploaded",
            {
                "user": current_user.get("username"),
                "file_id": file_id,
                "filename": file.filename,
                "category": category
            }
        )
        
        return {
            "file": file_info,
            "status": "uploaded",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{category}/{filename}")
async def download_file(
    category: str,
    filename: str,
    current_user: dict = Depends(get_current_user)
) -> FileResponse:
    """Download a file from the server."""
    try:
        # Determine directory based on category
        if category == "upload":
            base_dir = UPLOADS_DIR
        elif category == "temp":
            base_dir = TEMP_DIR
        elif category == "report":
            base_dir = REPORTS_DIR
        else:
            raise HTTPException(status_code=400, detail="Invalid file category")
        
        file_path = base_dir / filename
        
        # Security check - ensure file is within the allowed directory
        if not file_path.resolve().is_relative_to(base_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        await db_manager.log_system_event(
            "info", "files", "File downloaded",
            {
                "user": current_user.get("username"),
                "filename": filename,
                "category": category
            }
        )
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{category}/{filename}")
async def delete_file(
    category: str,
    filename: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete a file from the server."""
    try:
        # Determine directory based on category
        if category == "upload":
            base_dir = UPLOADS_DIR
        elif category == "temp":
            base_dir = TEMP_DIR
        elif category == "report":
            base_dir = REPORTS_DIR
        else:
            raise HTTPException(status_code=400, detail="Invalid file category")
        
        file_path = base_dir / filename
        
        # Security check
        if not file_path.resolve().is_relative_to(base_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete file
        file_path.unlink()
        
        await db_manager.log_system_event(
            "info", "files", "File deleted",
            {
                "user": current_user.get("username"),
                "filename": filename,
                "category": category
            }
        )
        
        return {
            "filename": filename,
            "category": category,
            "status": "deleted",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info/{category}/{filename}")
async def get_file_info(
    category: str,
    filename: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed information about a file."""
    try:
        # Determine directory based on category
        if category == "upload":
            base_dir = UPLOADS_DIR
        elif category == "temp":
            base_dir = TEMP_DIR
        elif category == "report":
            base_dir = REPORTS_DIR
        else:
            raise HTTPException(status_code=400, detail="Invalid file category")
        
        file_path = base_dir / filename
        
        # Security check
        if not file_path.resolve().is_relative_to(base_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        stat = file_path.stat()
        
        file_info = {
            "name": filename,
            "size": stat.st_size,
            "type": mimetypes.guess_type(str(file_path))[0] or "application/octet-stream",
            "category": category,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "path": str(file_path.relative_to(base_dir)),
            "permissions": oct(stat.st_mode)[-3:],
            "is_readable": os.access(file_path, os.R_OK),
            "is_writable": os.access(file_path, os.W_OK)
        }
        
        return {
            "file": file_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-upload")
async def batch_upload(
    files: List[UploadFile] = File(...),
    category: str = Form("upload"),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload multiple files at once."""
    try:
        uploaded_files = []
        errors = []
        
        for file in files:
            try:
                # Use the single file upload logic
                result = await upload_single_file(file, category, current_user)
                uploaded_files.append(result)
            except Exception as e:
                errors.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        await db_manager.log_system_event(
            "info", "files", "Batch upload completed",
            {
                "user": current_user.get("username"),
                "uploaded_count": len(uploaded_files),
                "error_count": len(errors)
            }
        )
        
        return {
            "uploaded_files": uploaded_files,
            "errors": errors,
            "summary": {
                "total": len(files),
                "successful": len(uploaded_files),
                "failed": len(errors)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/archive")
async def create_archive(
    archive_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create a ZIP archive from selected files."""
    try:
        files_to_archive = archive_data.get("files", [])
        archive_name = archive_data.get("name", f"archive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip")
        
        if not files_to_archive:
            raise HTTPException(status_code=400, detail="No files specified for archiving")
        
        # Create archive in temp directory
        archive_path = TEMP_DIR / archive_name
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_info in files_to_archive:
                category = file_info.get("category", "upload")
                filename = file_info.get("filename", "")
                
                # Determine source directory
                if category == "upload":
                    source_dir = UPLOADS_DIR
                elif category == "report":
                    source_dir = REPORTS_DIR
                else:
                    continue
                
                source_path = source_dir / filename
                
                if source_path.exists() and source_path.is_file():
                    # Add file to archive with relative path
                    arcname = f"{category}/{filename}"
                    zipf.write(source_path, arcname)
        
        # Get archive info
        stat = archive_path.stat()
        
        await db_manager.log_system_event(
            "info", "files", "Archive created",
            {
                "user": current_user.get("username"),
                "archive_name": archive_name,
                "files_count": len(files_to_archive)
            }
        )
        
        return {
            "archive_name": archive_name,
            "archive_size": stat.st_size,
            "files_included": len(files_to_archive),
            "download_url": f"/api/v1/files/download/temp/{archive_name}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating archive: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/storage/stats")
async def get_storage_stats(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get storage usage statistics."""
    try:
        stats = {}
        
        for category, directory in [("upload", UPLOADS_DIR), ("temp", TEMP_DIR), ("report", REPORTS_DIR)]:
            total_size = 0
            file_count = 0
            
            if directory.exists():
                for file_path in directory.glob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1
            
            stats[category] = {
                "file_count": file_count,
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        
        # Get disk usage for the storage directory
        disk_usage = shutil.disk_usage(UPLOADS_DIR)
        
        return {
            "categories": stats,
            "disk_usage": {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "total_gb": round(disk_usage.total / (1024 ** 3), 2),
                "used_gb": round(disk_usage.used / (1024 ** 3), 2),
                "free_gb": round(disk_usage.free / (1024 ** 3), 2)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup_temp_files(
    max_age_hours: int = 24,
    current_user: dict = Depends(require_admin)
) -> Dict[str, Any]:
    """Clean up old temporary files (admin only)."""
    try:
        cleaned_files = []
        current_time = datetime.utcnow().timestamp()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    cleaned_files.append({
                        "name": file_path.name,
                        "size": file_size,
                        "age_hours": round(file_age / 3600, 2)
                    })
        
        total_size_cleaned = sum(f["size"] for f in cleaned_files)
        
        await db_manager.log_system_event(
            "info", "files", "Temp files cleaned",
            {
                "user": current_user.get("username"),
                "files_cleaned": len(cleaned_files),
                "size_cleaned": total_size_cleaned
            }
        )
        
        return {
            "cleaned_files": cleaned_files,
            "total_files": len(cleaned_files),
            "total_size_cleaned": total_size_cleaned,
            "total_size_cleaned_mb": round(total_size_cleaned / (1024 * 1024), 2),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function for single file upload
async def upload_single_file(file: UploadFile, category: str, current_user: dict) -> Dict[str, Any]:
    """Helper function to upload a single file."""
    # Determine upload directory
    if category == "upload":
        upload_dir = UPLOADS_DIR
    elif category == "temp":
        upload_dir = TEMP_DIR
    elif category == "report":
        upload_dir = REPORTS_DIR
    else:
        upload_dir = UPLOADS_DIR
    
    # Generate unique filename
    file_id = str(uuid4())
    safe_filename = f"{file_id}_{file.filename}"
    file_path = upload_dir / safe_filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get file info
    stat = file_path.stat()
    
    return {
        "id": file_id,
        "name": file.filename,
        "safe_name": safe_filename,
        "size": stat.st_size,
        "category": category,
        "uploaded_at": datetime.utcnow().isoformat()
    }
