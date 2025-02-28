import os
from typing import List

import aiofiles
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend.services.doc_processing import DocumentParser
from backend.services.diagram_parser import DiagramParser

router = APIRouter(prefix="/doc", tags=["Document Processing"])

# Initialize parsers
doc_parser = DocumentParser()
diagram_parser = DiagramParser()

# Configuration
UPLOAD_DIR = "/opt/sutazaiapp/doc_data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ParseResponse(BaseModel):
    """
    Standardized parsing response model
    """

    success: bool
    message: str
    result: dict = {}


@router.post("/parse", response_model=ParseResponse)
async def parse_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Parse uploaded document (PDF/DOCX)

    Args:
        file (UploadFile): Uploaded document file

    Returns:
        Parsing result with extracted information
    """
    # Validate file extension
    filename = file.filename.lower()
    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or DOCX.")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, filename)

    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # Parse document in background
    def parse_file():
        try:
            if filename.endswith(".pdf"):
                result = doc_parser.parse_pdf(file_path)
            else:
                result = doc_parser.parse_docx(file_path)

            # Optional: Remove uploaded file after processing
            os.remove(file_path)

            return result
        except Exception as e:
            return {"error": str(e)}

    background_tasks.add_task(parse_file)

    return ParseResponse(success=True, message="Document queued for parsing", result={"filename": filename})


@router.post("/diagram/analyze", response_model=ParseResponse)
async def analyze_diagram(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Analyze uploaded diagram image

    Args:
        file (UploadFile): Uploaded diagram image

    Returns:
        Diagram analysis result
    """
    # Validate image extensions
    filename = file.filename.lower()
    valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    if not any(filename.endswith(ext) for ext in valid_extensions):
        raise HTTPException(status_code=400, detail="Unsupported image type. Use PNG, JPG, JPEG, BMP, or TIFF.")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, filename)

    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # Analyze diagram in background
    def analyze_file():
        try:
            result = diagram_parser.analyze_diagram(file_path)

            # Optional: Remove uploaded file after processing
            os.remove(file_path)

            return result
        except Exception as e:
            return {"error": str(e)}

    background_tasks.add_task(analyze_file)

    return ParseResponse(success=True, message="Diagram queued for analysis", result={"filename": filename})


def setup_routes(app):
    """
    Setup document processing routes

    Args:
        app (FastAPI): FastAPI application instance
    """
    app.include_router(router)
