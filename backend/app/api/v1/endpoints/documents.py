"""
Documents endpoint - NOT IMPLEMENTED

This is a placeholder endpoint for the documents API.
TODO: Implement real document management functionality:
- Document upload/download
- Document indexing and search
- Document version control
- Document sharing and permissions
"""
from fastapi import APIRouter, Depends, HTTPException
from app.core.middleware import jwt_required

router = APIRouter(dependencies=[Depends(jwt_required(scopes=["documents:read"]))])

@router.get("/")
async def list_documents():
    """
    List documents - NOT IMPLEMENTED
    
    Returns empty list as placeholder.
    TODO: Implement real document listing from database
    """
    return {
        "documents": [],
        "count": 0,
        "message": "Document management not yet implemented"
    }
