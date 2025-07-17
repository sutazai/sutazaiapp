"""
Code Router

This module provides API routes for code generation, snippets, and management.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from sqlmodel import Session as SQLModelSession

from backend.core.database import get_db
from backend.models.base_models import User
from backend.core.config import get_settings
from backend.dependencies import get_current_active_user
from backend.schemas import CodeSnippetCreate, CodeSnippetUpdate, CodeSnippetRead
from backend.crud import code_crud

# Set up logging
logger = logging.getLogger("code_router")

# Create router
router = APIRouter()

# Get application settings
settings = get_settings()


@router.get(
    "/snippets",
    summary="List all code snippets",
    response_model=List[CodeSnippetRead],
)
async def list_snippets(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    language: Optional[str] = None,
    tag: Optional[str] = None,
    db: SQLModelSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_active_user),
):
    """
    Retrieve a list of code snippets with optional filtering.
    Currently lists public snippets and snippets owned by the current user (if authenticated).
    """
    # TODO: Refine permission logic (e.g., admin sees all?)
    # TODO: Move visibility filtering logic into CRUD function?
    owner_id = current_user.id if current_user else None
    
    # Use CRUD function
    snippets = code_crud.list_snippets(
        db=db, 
        skip=skip, 
        limit=limit, 
        # owner_id=owner_id, # Filter by owner in CRUD?
        language=language,
        tag=tag
        # is_public=None # Fetch both initially, filter visibility below
    )
    
    # Filter for visibility (public or owned by current user)
    visible_snippets = []
    for snippet in snippets:
        if snippet.is_public or (current_user and snippet.owner_id == current_user.id):
            visible_snippets.append(snippet)
            
    return visible_snippets # FastAPI handles validation via response_model


@router.post(
    "/snippets",
    summary="Create a code snippet",
    response_model=CodeSnippetRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_snippet(
    snippet_in: CodeSnippetCreate, 
    db: SQLModelSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user) # Require logged-in user
):
    """
    Create a new code snippet, associated with the current user.
    """
    try:
        # Use CRUD function, passing owner_id
        created_snippet = code_crud.create_snippet(
            db=db, snippet_in=snippet_in, owner_id=current_user.id
        )
    except Exception as e:
        logger.error(f"Error creating code snippet: {str(e)}", exc_info=True)
        # db.rollback() should happen in CRUD function or get_db dependency
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating code snippet",
        )
    
    return created_snippet # Return ORM object for automatic validation


@router.get(
    "/snippets/{snippet_id}",
    summary="Get code snippet details",
    response_model=CodeSnippetRead,
)
async def get_snippet(
    snippet_id: int, 
    db: SQLModelSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_active_user) # Allow potential anonymous access to public
):
    """
    Retrieve a specific code snippet by ID.
    Returns the snippet if it's public or owned by the current user.
    """
    snippet = code_crud.get_snippet(db=db, snippet_id=snippet_id)
    if not snippet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Code snippet not found"
        )
    
    # Check permissions
    # TODO: Consider admin override
    if not snippet.is_public and (not current_user or snippet.owner_id != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this snippet"
        )
        
    return snippet # Return ORM object


@router.put(
    "/snippets/{snippet_id}",
    summary="Update a code snippet",
    response_model=CodeSnippetRead,
)
async def update_snippet(
    snippet_id: int, 
    snippet_in: CodeSnippetUpdate, # Use Update schema
    db: SQLModelSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user) # Require user
):
    """
    Update an existing code snippet. Only the owner can update.
    """
    db_snippet = code_crud.get_snippet(db=db, snippet_id=snippet_id)
    if not db_snippet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Code snippet not found"
        )
        
    # Check ownership
    # TODO: Consider admin override
    if db_snippet.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this snippet"
        )

    try:
        # Use CRUD function
        updated_snippet = code_crud.update_snippet(
            db=db, db_snippet=db_snippet, snippet_in=snippet_in
        )
    except Exception as e:
        logger.error(f"Error updating code snippet: {str(e)}")
        # Rollback should happen in CRUD/dependency
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating code snippet",
        )
    
    return updated_snippet # Return ORM object


@router.delete("/snippets/{snippet_id}", summary="Delete a code snippet", status_code=status.HTTP_204_NO_CONTENT)
async def delete_snippet(
    snippet_id: int, 
    db: SQLModelSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user) # Require user
):
    """
    Delete a code snippet. Only the owner can delete.
    """
    db_snippet = code_crud.get_snippet(db=db, snippet_id=snippet_id)
    if not db_snippet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Code snippet not found"
        )
    
    # Check ownership
    # TODO: Consider admin override
    if db_snippet.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this snippet"
        )

    try:
        # Use CRUD function
        deleted = code_crud.delete_snippet(db=db, snippet_id=snippet_id)
        if not deleted:
             # Should not happen if get_snippet succeeded, but handle defensively
             raise HTTPException(status_code=404, detail="Snippet not found during delete attempt")
    except Exception as e:
        logger.error(f"Error deleting code snippet: {str(e)}")
        # Rollback should happen in CRUD/dependency
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting code snippet",
        )
    
    # Return None for 204 No Content status
    return None 


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
