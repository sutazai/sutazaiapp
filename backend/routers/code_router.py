"""
Code Router

This module provides API routes for code generation, snippets, and management.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..core.database import get_db
from ..models.code_model import CodeSnippet
from ..core.config import get_settings

# Set up logging
logger = logging.getLogger("code_router")

# Create router
router = APIRouter()

# Get application settings
settings = get_settings()


# Pydantic models for request/response
class CodeSnippetCreate(BaseModel):
    title: str
    description: Optional[str] = None
    language: str
    code_content: str
    is_public: bool = False
    tags: Optional[str] = None


class CodeSnippetResponse(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    language: str
    code_content: str
    user_id: Optional[int] = None
    is_public: bool
    tags: List[str]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.get(
    "/snippets",
    summary="List all code snippets",
    response_model=List[CodeSnippetResponse],
)
async def list_snippets(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    language: Optional[str] = None,
    tag: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Retrieve a list of code snippets with optional filtering.
    """
    query = db.query(CodeSnippet)

    # Apply filters if specified
    if language:
        query = query.filter(CodeSnippet.language == language)

    if tag:
        # Filter by tag (simple contains)
        query = query.filter(CodeSnippet.tags.contains(tag))

    # Get results with pagination
    snippets = query.offset(skip).limit(limit).all()
    return [snippet.as_dict for snippet in snippets]


@router.post(
    "/snippets",
    summary="Create a code snippet",
    response_model=CodeSnippetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_snippet(snippet: CodeSnippetCreate, db: Session = Depends(get_db)):
    """
    Create a new code snippet.
    """
    db_snippet = CodeSnippet(
        title=snippet.title,
        description=snippet.description,
        language=snippet.language,
        code_content=snippet.code_content,
        is_public=snippet.is_public,
        tags=snippet.tags,
    )

    try:
        db.add(db_snippet)
        db.commit()
        db.refresh(db_snippet)
    except Exception as e:
        logger.error(f"Error creating code snippet: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating code snippet",
        )

    return db_snippet.as_dict


@router.get(
    "/snippets/{snippet_id}",
    summary="Get code snippet details",
    response_model=CodeSnippetResponse,
)
async def get_snippet(snippet_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a specific code snippet by ID.
    """
    snippet = db.query(CodeSnippet).filter(CodeSnippet.id == snippet_id).first()
    if not snippet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Code snippet not found"
        )

    return snippet.as_dict


@router.put(
    "/snippets/{snippet_id}",
    summary="Update a code snippet",
    response_model=CodeSnippetResponse,
)
async def update_snippet(
    snippet_id: int, snippet_update: CodeSnippetCreate, db: Session = Depends(get_db)
):
    """
    Update an existing code snippet.
    """
    db_snippet = db.query(CodeSnippet).filter(CodeSnippet.id == snippet_id).first()
    if not db_snippet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Code snippet not found"
        )

    # Update fields
    db_snippet.title = snippet_update.title
    db_snippet.description = snippet_update.description
    db_snippet.language = snippet_update.language
    db_snippet.code_content = snippet_update.code_content
    db_snippet.is_public = snippet_update.is_public
    db_snippet.tags = snippet_update.tags

    try:
        db.commit()
        db.refresh(db_snippet)
    except Exception as e:
        logger.error(f"Error updating code snippet: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating code snippet",
        )

    return db_snippet.as_dict


@router.delete("/snippets/{snippet_id}", summary="Delete a code snippet")
async def delete_snippet(snippet_id: int, db: Session = Depends(get_db)):
    """
    Delete a code snippet.
    """
    db_snippet = db.query(CodeSnippet).filter(CodeSnippet.id == snippet_id).first()
    if not db_snippet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Code snippet not found"
        )

    try:
        db.delete(db_snippet)
        db.commit()
    except Exception as e:
        logger.error(f"Error deleting code snippet: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting code snippet",
        )

    return {"message": "Code snippet deleted successfully"}


@router.post("/generate", summary="Generate code from description")
async def generate_code(
    prompt: str = Body(..., embed=True),
    language: str = Body(..., embed=True),
    max_tokens: int = Body(500, embed=True),
):
    """
    Generate code from a natural language description using AI models.

    This is a placeholder - in a real implementation, it would call the
    appropriate AI service to generate code based on the prompt.
    """
    try:
        # In a real implementation, this would call an AI model
        # For now, we return a placeholder response
        sample_code = f"// Generated code in {language} based on: {prompt}\n\n"

        if language.lower() == "python":
            sample_code += "def main():\n    print('Hello, World!')\n    # TODO: Implement functionality\n\nif __name__ == '__main__':\n    main()"
        elif language.lower() == "javascript":
            sample_code += "function main() {\n    console.log('Hello, World!');\n    // TODO: Implement functionality\n}\n\nmain();"
        else:
            sample_code += f"// Sample code for {language}\n// TODO: Implement functionality based on: {prompt}"

        return {"generated_code": sample_code, "language": language, "prompt": prompt}
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating code",
        )
