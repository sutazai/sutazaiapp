"""
Diagram Router

This module provides REST API endpoints for diagram processing.
"""

import os
import logging
import uuid
import tempfile
import mimetypes
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path

from ..core.exceptions import ServiceError
from ..core.config import settings
from ..services.diagram_parser import DiagramParser

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize diagram parser
diagram_parser = DiagramParser()

# Safe MIME types for diagrams
ALLOWED_MIME_TYPES = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/svg+xml",
    "application/pdf",
    "application/vnd.visio",
    "application/vnd.ms-visio.drawing",
    "application/vnd.openxmlformats-officedocument.drawingml.drawing",
]

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


def safely_remove_file(file_path: str) -> None:
    """
    Safely remove a file with error handling.

    Args:
        file_path: Path to the file to remove
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Successfully removed temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error removing temporary file {file_path}: {str(e)}")


def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file size and type.

    Args:
        file: Uploaded file

    Raises:
        HTTPException: If validation fails
    """
    # Check file existence
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided"
        )

    # Sanitize filename
    filename = os.path.basename(file.filename).lower()

    # Check file type
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type or mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {mime_type or 'unknown'}. Supported types: {', '.join(ALLOWED_MIME_TYPES)}",
        )


@router.post("/parse", status_code=status.HTTP_200_OK)
async def parse_diagram(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extract_elements: bool = True,
    analyze_relationships: bool = True,
) -> JSONResponse:
    """
    Parse a diagram file and extract its elements.

    Args:
        background_tasks: FastAPI BackgroundTasks for cleanup
        file: Uploaded diagram file
        extract_elements: Whether to extract diagram elements
        analyze_relationships: Whether to analyze element relationships

    Returns:
        JSONResponse containing parsed diagram information

    Raises:
        HTTPException: If parsing fails
    """
    temp_file_path = None
    request_id = str(uuid.uuid4())

    try:
        # Validate file
        validate_file(file)

        # Create a temporary file with a secure random name
        with tempfile.NamedTemporaryFile(
            delete=False, suffix="_" + os.path.basename(file.filename)
        ) as tmp:
            temp_file_path = tmp.name
            content = await file.read()

            # Check file size
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE / 1024 / 1024}MB",
                )

            # Write content to temporary file
            tmp.write(content)

        logger.info(
            f"Request {request_id}: Processing diagram {file.filename}, size: {len(content) / 1024:.2f}KB"
        )

        # Parse diagram
        result = diagram_parser.parse_diagram(
            file_path=temp_file_path,
            extract_elements=extract_elements,
            analyze_relationships=analyze_relationships,
        )

        # Add request ID to result
        result["request_id"] = request_id

        # Schedule file cleanup
        background_tasks.add_task(safely_remove_file, temp_file_path)

        logger.info(
            f"Request {request_id}: Successfully processed diagram {file.filename}"
        )
        return JSONResponse(content=result)

    except ServiceError as e:
        if temp_file_path:
            safely_remove_file(temp_file_path)
        logger.error(
            f"Request {request_id}: Service error processing diagram: {str(e)}"
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        if temp_file_path:
            safely_remove_file(temp_file_path)
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        if temp_file_path:
            safely_remove_file(temp_file_path)
        logger.error(
            f"Request {request_id}: Error processing diagram: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/supported-formats", status_code=status.HTTP_200_OK)
async def get_supported_formats() -> Dict[str, Any]:
    """
    Get list of supported diagram formats.

    Returns:
        Dictionary containing supported formats
    """
    return {
        "formats": diagram_parser.supported_formats,
        "max_file_size": MAX_FILE_SIZE,
        "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
        "allowed_mime_types": ALLOWED_MIME_TYPES,
    }


@router.post("/validate", status_code=status.HTTP_200_OK)
async def validate_diagram(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Validate a diagram file.

    Args:
        background_tasks: FastAPI BackgroundTasks for cleanup
        file: Uploaded diagram file

    Returns:
        Dictionary containing validation result

    Raises:
        HTTPException: If validation fails
    """
    temp_file_path = None
    request_id = str(uuid.uuid4())

    try:
        # Validate file
        validate_file(file)

        # Create a temporary file with a secure random name
        with tempfile.NamedTemporaryFile(
            delete=False, suffix="_" + os.path.basename(file.filename)
        ) as tmp:
            temp_file_path = tmp.name
            content = await file.read()

            # Check file size
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE / 1024 / 1024}MB",
                )

            # Write content to temporary file
            tmp.write(content)

        logger.info(
            f"Request {request_id}: Validating diagram {file.filename}, size: {len(content) / 1024:.2f}KB"
        )

        # Validate file
        diagram_parser._validate_file(temp_file_path)

        # Get metadata
        metadata = diagram_parser._get_metadata(temp_file_path)

        # Schedule file cleanup
        background_tasks.add_task(safely_remove_file, temp_file_path)

        logger.info(
            f"Request {request_id}: Successfully validated diagram {file.filename}"
        )

        return {"valid": True, "metadata": metadata, "request_id": request_id}

    except ServiceError as e:
        if temp_file_path:
            safely_remove_file(temp_file_path)
        logger.error(f"Request {request_id}: Validation error for diagram: {str(e)}")
        return {"valid": False, "error": str(e), "request_id": request_id}
    except HTTPException:
        if temp_file_path:
            safely_remove_file(temp_file_path)
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        if temp_file_path:
            safely_remove_file(temp_file_path)
        logger.error(
            f"Request {request_id}: Error validating diagram: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/analyze")
async def analyze_diagram(
    file: UploadFile = File(...), include_metadata: bool = True
) -> JSONResponse:
    """
    Analyze a diagram and extract detailed information.

    Args:
        file: Uploaded diagram file
        include_metadata: Whether to include file metadata

    Returns:
        JSONResponse containing analysis results

    Raises:
        HTTPException: If analysis fails
    """
    try:
        # Create upload directory if it doesn't exist
        upload_dir = Path(settings.UPLOAD_DIR) / "diagrams"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Parse diagram
        result = diagram_parser.parse_diagram(str(file_path))

        # Add additional analysis if needed
        analysis = {
            "elements": result["elements"],
            "relationships": result["relationships"],
        }

        if include_metadata:
            analysis["metadata"] = result["metadata"]

        # Clean up uploaded file
        os.remove(file_path)

        return JSONResponse(content=analysis)

    except ServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing diagram: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
