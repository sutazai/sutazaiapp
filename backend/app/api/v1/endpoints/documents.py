"""Documents endpoint stub"""
from fastapi import APIRouter, Depends
from app.core.middleware import jwt_required

router = APIRouter(dependencies=[Depends(jwt_required(scopes=["documents:read"]))])

@router.get("/")
async def list_documents():
    return {"documents": [], "count": 0}
