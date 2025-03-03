#!/usr/bin/env python3.11
"""Document Routes Module

This module provides routes for document parsing and diagram analysis.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles  # type: ignore # noqa
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from backend.services.diagram_parser import DiagramParser
from backend.services.doc_processing import DocumentParser

# Create router
router = APIRouter(prefix="/doc", tags=["document-processing"])

# Initialize parsers
doc_parser = DocumentParser(output_dir="/opt/sutazaiapp/data/documents")
diagram_parser = DiagramParser(output_dir="/opt/sutazaiapp/data/diagrams")

# Configuration
UPLOAD_DIR = "/opt/sutazaiapp/doc_data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ParseResponse(BaseModel):
    """Standardized parsing response model."""
    success: bool = Field(description="Whether the parsing was successful")
    message: str = Field(description="Status message or error description")
    result: Dict[str, Any] = Field(
        default_factory=dict, description="Parsing result data"
    )

@router.post("/parse", response_model=ParseResponse)
async def parse_document(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> ParseResponse:
    """Parse uploaded documents (PDF/DOCX).
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded file
        
    Returns:
        Parsing response with status and result
    """
    # Validate file extension
    filename: Optional[str] = file.filename
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided in the upload."
        )
        
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in [".pdf", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Use PDF or DOCX.",
        )
    
    # Save file to disk
    file_path = f"/opt/sutazaiapp/data/documents/{filename}"
    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
        
    # Create background task for parsing
    def parse_file() -> Dict[str, Any]:
        """Parse the uploaded document in the background."""
        try:
            if file_ext == ".pdf":
                result = doc_parser.parse_pdf(file_path)
            else:
                result = doc_parser.parse_docx(file_path)
                
            # Optional: Remove uploaded file after processing
            os.remove(file_path)
            return result
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            return {"error": str(e)}
            
    # Add task to background tasks
    background_tasks.add_task(parse_file)
    
    return ParseResponse(
        success=True,
        message=f"Document uploaded and being processed: {filename}",
    )

@router.post("/diagram/analyze", response_model=ParseResponse)
async def analyze_diagram(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> ParseResponse:
    """Analyze uploaded diagram images.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded diagram image
        
    Returns:
        Analysis response with status and result
    """
    # Validate file extension
    filename: Optional[str] = file.filename
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided in the upload."
        )
        
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Use PNG or JPG/JPEG.",
        )
        
    # Save file to disk
    file_path = f"/opt/sutazaiapp/data/diagrams/{filename}"
    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # Create background task for analysis
    def analyze_file() -> Dict[str, Any]:
        """Analyze the uploaded diagram in the background."""
        try:
            result = diagram_parser.analyze_diagram(Path(file_path))
            # Optional: Remove uploaded file after processing
            os.remove(file_path)
            return result
        except Exception as e:
            logger.error(f"Error analyzing diagram: {e}")
            return {"error": str(e)}
            
    # Add task to background tasks
    background_tasks.add_task(analyze_file)
    
    return ParseResponse(
        success=True,
        message=f"Diagram uploaded and being analyzed: {filename}",
    )

def setup_routes(app) -> None:
    """Include router in the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(router)

