"""
CRUD operations for CodeSnippet model.
"""

import logging
from typing import List, Optional

from sqlmodel import Session, select # Use SQLModel components

from backend.models.base_models import CodeSnippet # Import model
from backend.schemas import CodeSnippetCreate, CodeSnippetUpdate # Import schemas

logger = logging.getLogger(__name__)

def create_snippet(db: Session, snippet_in: CodeSnippetCreate, owner_id: int) -> CodeSnippet:
    """Create a new code snippet."""
    logger.info(f"Creating code snippet titled '{snippet_in.title}' for owner {owner_id}")
    # TODO: Implement actual DB insertion
    db_snippet = CodeSnippet.model_validate(snippet_in) # Basic validation
    db_snippet.owner_id = owner_id
    # Placeholder implementation:
    db.add(db_snippet)
    db.commit()
    db.refresh(db_snippet)
    logger.info(f"Code snippet created with ID: {db_snippet.id}")
    return db_snippet

def get_snippet(db: Session, snippet_id: int) -> Optional[CodeSnippet]:
    """Get a code snippet by ID."""
    logger.debug(f"Retrieving code snippet with ID: {snippet_id}")
    # TODO: Implement actual DB query
    snippet = db.get(CodeSnippet, snippet_id)
    return snippet

def list_snippets(
    db: Session, 
    skip: int = 0, 
    limit: int = 100, 
    owner_id: Optional[int] = None, 
    language: Optional[str] = None,
    tag: Optional[str] = None, # TODO: Implement tag filtering logic (JSON contains?)
    is_public: Optional[bool] = None
) -> List[CodeSnippet]:
    """List code snippets with filtering and pagination."""
    logger.debug("Listing code snippets with filters")
    statement = select(CodeSnippet)
    if owner_id is not None:
        statement = statement.where(CodeSnippet.owner_id == owner_id)
    if language is not None:
        statement = statement.where(CodeSnippet.language == language)
    if is_public is not None:
        statement = statement.where(CodeSnippet.is_public == is_public)
    # TODO: Add tag filtering logic here if needed
        
    statement = statement.offset(skip).limit(limit)
    snippets = db.exec(statement).all()
    return list(snippets)

def update_snippet(
    db: Session, db_snippet: CodeSnippet, snippet_in: CodeSnippetUpdate
) -> CodeSnippet:
    """Update an existing code snippet."""
    logger.info(f"Updating code snippet with ID: {db_snippet.id}")
    update_data = snippet_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_snippet, key, value)
    # TODO: Implement actual DB update
    db.add(db_snippet)
    db.commit()
    db.refresh(db_snippet)
    logger.info(f"Code snippet {db_snippet.id} updated.")
    return db_snippet

def delete_snippet(db: Session, snippet_id: int) -> Optional[CodeSnippet]:
    """Delete a code snippet by ID."""
    logger.info(f"Deleting code snippet with ID: {snippet_id}")
    db_snippet = db.get(CodeSnippet, snippet_id)
    if db_snippet:
        # TODO: Implement actual DB deletion
        db.delete(db_snippet)
        db.commit()
        logger.info(f"Code snippet {snippet_id} deleted.")
        return db_snippet # Return deleted object
    logger.warning(f"Code snippet {snippet_id} not found for deletion.")
    return None 