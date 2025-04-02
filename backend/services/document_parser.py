"""
SutazAI Document Parser Service
Provides API endpoints for parsing and analyzing documents
"""

import os
import tempfile
import logging
import base64
import hashlib
import uuid
import time
from pathlib import Path
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
    Request,
    Depends,
)
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger("document_parser_service")

# Max file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Set of allowed file extensions
ALLOWED_FILE_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".rtf"}

# Set of allowed MIME types
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "application/rtf",
}

# Global flag to track service availability
DOCUMENT_SERVICE_AVAILABLE = False
TESSERACT_AVAILABLE = False
PDF2IMAGE_AVAILABLE = False
TABULA_AVAILABLE = False
OCR_CAPABILITY = False

# Try to import optional dependencies
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    logger.warning("pytesseract not installed - OCR functionality will be limited")

try:
    # F401: Removed unused imports convert_from_path, convert_from_bytes
    # from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    # F401: Removed unused import tabula
    # import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

# Check if Tesseract is actually available
if TESSERACT_AVAILABLE:
    try:
        pytesseract.get_tesseract_version()
        OCR_CAPABILITY = True
    except Exception as e:  # E722: Replaced bare except
        logger.warning(
            f"Tesseract not found in PATH or error during check: {e}. OCR functionality will be disabled"
        )
        TESSERACT_AVAILABLE = False
        OCR_CAPABILITY = False

# Create temp directory with secure permissions
TEMP_DIR = Path("/opt/sutazaiapp/tmp/document_parser")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.chmod(TEMP_DIR, 0o700)  # Only owner can access

# Wrap document service import to catch any missing dependencies
try:
    # Import document parsing service with absolute import
    from backend.services.document_parsing.document_service import document_service

    # Verify service is working properly
    if (
        hasattr(document_service, "pdf_parser")
        and document_service.pdf_parser is not None
    ):
        DOCUMENT_SERVICE_AVAILABLE = True
    else:
        logger.warning("Document service initialized but PDF parser is not available")
except ImportError as e:
    logger.error(f"Failed to import document_service: {str(e)}")
except Exception as e:
    logger.error(f"Error initializing document_service: {str(e)}")

# Create API router
router = APIRouter(prefix="/documents", tags=["Document Parsing"])


# Dependency for file validation
async def validate_file_upload(file: UploadFile = File(...)) -> UploadFile:
    """
    Validate that the uploaded file meets security requirements.

    Args:
        file: The uploaded file

    Returns:
        The validated file or raises an exception
    """
    # Check if filename is provided
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file extension
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_FILE_EXTENSIONS)}",
        )

    # Check content type if available
    content_type = file.content_type
    if content_type and content_type not in ALLOWED_MIME_TYPES:
        logger.warning(
            f"Suspicious content type: {content_type} for file {file.filename}"
        )
        raise HTTPException(status_code=400, detail="Unsupported content type")

    return file


# Data models
class DocumentParseRequest(BaseModel):
    """Request model for parsing documents from base64 data"""

    file_content: str = Field(..., description="Base64 encoded file content")
    filename: Optional[str] = Field(
        None, description="Original filename with extension"
    )
    extract_tables: bool = Field(True, description="Whether to extract tables")
    extract_images: bool = Field(True, description="Whether to extract images")
    enable_ocr: bool = Field(
        False, description="Whether to enable OCR for scanned documents"
    )

    @validator("filename")
    def validate_filename(cls, v):
        if v is not None:
            # Validate file extension
            file_ext = os.path.splitext(v.lower())[1]
            if file_ext not in ALLOWED_FILE_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type. Allowed types: {', '.join(ALLOWED_FILE_EXTENSIONS)}"
                )

            # Sanitize filename to prevent path traversal
            return os.path.basename(v)
        return v

    @validator("file_content")
    def validate_file_size(cls, v):
        # Check that the base64 content isn't too large
        # Base64 encoding increases size by ~33%, so we adjust accordingly
        if len(v) * 0.75 > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB"
            )
        return v


class DocumentTextRequest(BaseModel):
    """Request model for extracting text from documents"""

    file_content: str = Field(..., description="Base64 encoded file content")
    filename: Optional[str] = Field(
        None, description="Original filename with extension"
    )
    enable_ocr: bool = Field(
        False, description="Whether to enable OCR for scanned documents"
    )

    @validator("filename")
    def validate_filename(cls, v):
        if v is not None:
            # Validate file extension
            file_ext = os.path.splitext(v.lower())[1]
            if file_ext not in ALLOWED_FILE_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type. Allowed types: {', '.join(ALLOWED_FILE_EXTENSIONS)}"
                )

            # Sanitize filename to prevent path traversal
            return os.path.basename(v)
        return v

    @validator("file_content")
    def validate_file_size(cls, v):
        # Check that the base64 content isn't too large
        if len(v) * 0.75 > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB"
            )
        return v


class DocumentResponse(BaseModel):
    """Response model for document parsing results"""

    success: bool = Field(..., description="Whether the parsing was successful")
    message: Optional[str] = Field(None, description="Error message if unsuccessful")
    filename: Optional[str] = Field(None, description="Original filename")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    text: Optional[str] = Field(None, description="Extracted full text")
    pages: Optional[int] = Field(None, description="Number of pages")
    structure: Optional[Dict[str, Any]] = Field(None, description="Document structure")
    tables: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted tables")
    images: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted images")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class TextExtractionResponse(BaseModel):
    """Response model for text extraction"""

    success: bool = Field(..., description="Whether the extraction was successful")
    message: Optional[str] = Field(None, description="Error message if unsuccessful")
    filename: Optional[str] = Field(None, description="Original filename")
    text: Optional[str] = Field(None, description="Extracted text")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class MetadataResponse(BaseModel):
    """Response model for metadata extraction"""

    success: bool = Field(..., description="Whether the extraction was successful")
    message: Optional[str] = Field(None, description="Error message if unsuccessful")
    filename: Optional[str] = Field(None, description="Original filename")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


# Helper functions
async def save_upload_file_streaming(file: UploadFile) -> Path:
    """
    Save the uploaded file to a secure temporary location using streaming
    to support large files efficiently.

    Args:
        file: The uploaded file

    Returns:
        Path to the saved temporary file
    """
    # Generate secure random filename
    file_id = uuid.uuid4().hex
    file_ext = os.path.splitext(file.filename)[1]
    secure_filename = f"{file_id}{file_ext}"
    temp_path = TEMP_DIR / secure_filename

    # Stream file content to disk in chunks
    try:
        with open(temp_path, "wb") as buffer:
            # Read in 1MB chunks
            chunk_size = 1024 * 1024
            while chunk := await file.read(chunk_size):
                # Check file size as we read
                if temp_path.stat().st_size > MAX_FILE_SIZE:
                    # Remove partial file
                    os.unlink(temp_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB",
                    )
                buffer.write(chunk)
    except Exception as e:
        # Ensure temp file is removed on error
        if temp_path.exists():
            os.unlink(temp_path)
        raise e

    return temp_path


def secure_cleanup_file(temp_path: Path) -> None:
    """
    Securely clean up a temporary file by overwriting and removing it.

    Args:
        temp_path: Path to the file to clean up
    """
    try:
        if temp_path.exists():
            # Overwrite the file with zeros before deletion for sensitive files
            with open(temp_path, "wb") as f:
                f.write(b"\0" * min(1024 * 1024, temp_path.stat().st_size))
            # Delete the file
            os.unlink(temp_path)
    except Exception as e:
        logger.error(f"Error cleaning up temporary file {temp_path}: {str(e)}")


# API Endpoints
@router.post("/parse", response_model=DocumentResponse, summary="Parse document")
async def parse_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(validate_file_upload),
):
    """
    Parse a document file and extract its content

    - **file**: Document file to parse (PDF, DOCX)

    Returns document metadata, content, structure and extracted elements
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    if not DOCUMENT_SERVICE_AVAILABLE:
        return DocumentResponse(
            success=False,
            message="Document parsing service is not available",
            filename=file.filename,
            request_id=request_id,
            processing_time=time.time() - start_time,
        )

    try:
        # Save file using streaming for large files
        temp_path = await save_upload_file_streaming(file)

        # Add cleanup task
        background_tasks.add_task(secure_cleanup_file, temp_path)

        # Calculate file hash for tracking
        file_hash = hashlib.md5(open(temp_path, "rb").read()).hexdigest()
        logger.info(
            f"Processing document: {file.filename} (size: {temp_path.stat().st_size} bytes, hash: {file_hash})"
        )

        # Parse the document
        result = document_service.parse_document(
            file_path=str(temp_path), filename=file.filename
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        if "error" in result:
            return DocumentResponse(
                success=False,
                message=result["error"],
                filename=file.filename,
                request_id=request_id,
                processing_time=processing_time,
            )

        # Process successful result
        response = DocumentResponse(
            success=True,
            filename=file.filename,
            metadata=result.get("metadata", {}),
            text=result.get("full_text", ""),
            pages=result.get("num_pages", None),
            structure=result.get("structure", {}),
            tables=result.get("tables", []) if "tables" in result else None,
            images=result.get("images", []) if "images" in result else None,
            request_id=request_id,
            processing_time=processing_time,
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error parsing document: {str(e)}")
        return DocumentResponse(
            success=False,
            message=f"Error parsing document: {str(e)}",
            filename=file.filename,
            request_id=request_id,
            processing_time=time.time() - start_time,
        )


@router.post(
    "/parse/base64",
    response_model=DocumentResponse,
    summary="Parse document from base64",
)
async def parse_document_base64(
    request: Request, document_request: DocumentParseRequest
):
    """
    Parse a document from base64-encoded content

    - **request**: Request containing file content and parsing options

    Returns document metadata, content, structure and extracted elements
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    if not DOCUMENT_SERVICE_AVAILABLE:
        return DocumentResponse(
            success=False,
            message="Document parsing service is not available",
            filename=document_request.filename,
            request_id=request_id,
            processing_time=time.time() - start_time,
        )

    try:
        # Decode base64 content
        try:
            file_content = base64.b64decode(document_request.file_content)
        except Exception as e:
            return DocumentResponse(
                success=False,
                message=f"Invalid base64 encoding: {str(e)}",
                filename=document_request.filename,
                request_id=request_id,
                processing_time=time.time() - start_time,
            )

        # Use sanitized filename
        safe_filename = (
            document_request.filename
            if document_request.filename
            else "unnamed_document"
        )

        # Parse the document
        result = document_service.parse_document(
            file_content=file_content, filename=safe_filename
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        if "error" in result:
            return DocumentResponse(
                success=False,
                message=result["error"],
                filename=safe_filename,
                request_id=request_id,
                processing_time=processing_time,
            )

        # Process successful result
        response = DocumentResponse(
            success=True,
            filename=safe_filename,
            metadata=result.get("metadata", {}),
            text=result.get("full_text", ""),
            pages=result.get("num_pages", None),
            structure=result.get("structure", {}),
            tables=result.get("tables", []) if "tables" in result else None,
            images=result.get("images", []) if "images" in result else None,
            request_id=request_id,
            processing_time=processing_time,
        )

        return response

    except Exception as e:
        logger.error(f"Error parsing document: {str(e)}")
        return DocumentResponse(
            success=False,
            message=f"Error parsing document: {str(e)}",
            filename=document_request.filename,
            request_id=request_id,
            processing_time=time.time() - start_time,
        )


@router.post(
    "/extract-text",
    response_model=TextExtractionResponse,
    summary="Extract text from document",
)
async def extract_text(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(validate_file_upload),
    enable_ocr: bool = Form(False),
):
    """
    Extract text from a document file

    - **file**: Document file to extract text from (PDF, DOCX)
    - **enable_ocr**: Whether to use OCR for scanned documents

    Returns extracted text content
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    if not DOCUMENT_SERVICE_AVAILABLE:
        return TextExtractionResponse(
            success=False,
            message="Document parsing service is not available",
            filename=file.filename,
            request_id=request_id,
            processing_time=time.time() - start_time,
        )

    try:
        # Save file using streaming for large files
        temp_path = await save_upload_file_streaming(file)

        # Add cleanup task
        background_tasks.add_task(secure_cleanup_file, temp_path)

        # Extract text from document using document service
        text = document_service.extract_text(file_path=str(temp_path))

        # Calculate processing time
        processing_time = time.time() - start_time

        return TextExtractionResponse(
            success=True,
            filename=file.filename,
            text=text,
            request_id=request_id,
            processing_time=processing_time,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return TextExtractionResponse(
            success=False,
            message=f"Error extracting text: {str(e)}",
            filename=file.filename,
            request_id=request_id,
            processing_time=time.time() - start_time,
        )


@router.post(
    "/extract-metadata",
    response_model=MetadataResponse,
    summary="Extract metadata from document",
)
async def extract_metadata(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    """
    Extract metadata from a document file

    - **file**: Document file to extract metadata from (PDF, DOCX)

    Returns document metadata
    """
    if not DOCUMENT_SERVICE_AVAILABLE:
        return MetadataResponse(
            success=False,
            message="Document parsing service is not available",
            filename=file.filename,
        )

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as temp:
            # Write uploaded file to temp file
            content = await file.read()
            temp.write(content)
            temp_path = temp.name

        # Add cleanup task
        background_tasks.add_task(os.unlink, temp_path)

        # Extract metadata
        metadata = document_service.extract_metadata(file_path=temp_path)

        return MetadataResponse(success=True, filename=file.filename, metadata=metadata)

    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error extracting metadata: {str(e)}"
        )


@router.get("/health", summary="Document parser health check")
async def health_check():
    """
    Check the health and status of the document parser service

    Returns information about available parsing features and dependencies
    """
    return {
        "status": "healthy" if DOCUMENT_SERVICE_AVAILABLE else "limited",
        "document_service_available": DOCUMENT_SERVICE_AVAILABLE,
        "pdf_parser_available": DOCUMENT_SERVICE_AVAILABLE
        and hasattr(document_service, "pdf_parser"),
        "docx_parser_available": DOCUMENT_SERVICE_AVAILABLE
        and hasattr(document_service, "docx_parser"),
        "ocr_available": DOCUMENT_SERVICE_AVAILABLE
        and hasattr(document_service, "pdf_parser")
        and getattr(document_service.pdf_parser, "enable_ocr", False),
        "table_extraction_available": DOCUMENT_SERVICE_AVAILABLE
        and hasattr(document_service, "pdf_parser")
        and getattr(document_service.pdf_parser, "extract_tables", False),
    }


# Export the router
document_parser_router = router

"""
SutazAI Document Parser Service Module
Re-exports the document parser router for easier imports
"""

__all__ = ["document_parser_router"]
