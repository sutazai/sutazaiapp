"""
Pagination utilities for list endpoints
Provides cursor-based and offset-based pagination
"""

from typing import TypeVar, Generic, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy import Select, func
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar('T')


class PaginationParams(BaseModel):
    """Standard pagination parameters"""
    skip: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(50, ge=1, le=500, description="Maximum items to return")
    order_by: Optional[str] = Field(None, description="Field to order by")
    order_desc: bool = Field(True, description="Order descending (newest first)")


class CursorPaginationParams(BaseModel):
    """Cursor-based pagination parameters for large datasets"""
    cursor: Optional[str] = Field(None, description="Cursor for next page")
    limit: int = Field(50, ge=1, le=500, description="Maximum items to return")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper"""
    items: List[T]
    total: int
    skip: int
    limit: int
    has_more: bool = Field(description="Whether more items are available")
    
    class Config:
        from_attributes = True


class CursorPaginatedResponse(BaseModel, Generic[T]):
    """Cursor-based paginated response"""
    items: List[T]
    next_cursor: Optional[str] = Field(None, description="Cursor for next page")
    has_more: bool
    
    class Config:
        from_attributes = True


async def paginate_query(
    db: AsyncSession,
    query: Select,
    params: PaginationParams,
    model_class: type
) -> PaginatedResponse:
    """
    Apply pagination to SQLAlchemy query
    
    Args:
        db: Database session
        query: SQLAlchemy select statement
        params: Pagination parameters
        model_class: Model class for type checking
        
    Returns:
        PaginatedResponse with items and metadata
    """
    # Count total items
    count_query = query.with_only_columns(func.count()).order_by(None)
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()
    
    # Apply pagination
    paginated_query = query.offset(params.skip).limit(params.limit)
    
    # Execute query
    result = await db.execute(paginated_query)
    items = result.scalars().all()
    
    # Calculate if more items exist
    has_more = (params.skip + len(items)) < total
    
    return PaginatedResponse(
        items=items,
        total=total,
        skip=params.skip,
        limit=params.limit,
        has_more=has_more
    )


def paginate_list(
    items: List[T],
    skip: int = 0,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Paginate a Python list (for in-memory collections)
    
    Args:
        items: List of items to paginate
        skip: Number of items to skip
        limit: Maximum items to return
        
    Returns:
        Dict with paginated results and metadata
    """
    total = len(items)
    start = skip
    end = skip + limit
    
    paginated_items = items[start:end]
    has_more = end < total
    
    return {
        "items": paginated_items,
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": has_more,
        "page": (skip // limit) + 1 if limit > 0 else 1,
        "total_pages": (total + limit - 1) // limit if limit > 0 else 1
    }


def create_cursor(item_id: Any, timestamp: Optional[float] = None) -> str:
    """Create cursor from item ID and optional timestamp"""
    import base64
    import json
    
    cursor_data = {"id": str(item_id)}
    if timestamp:
        cursor_data["ts"] = timestamp
    
    cursor_json = json.dumps(cursor_data)
    return base64.urlsafe_b64encode(cursor_json.encode()).decode()


def parse_cursor(cursor: str) -> Dict[str, Any]:
    """Parse cursor to extract item ID and timestamp"""
    import base64
    import json
    
    try:
        cursor_json = base64.urlsafe_b64decode(cursor.encode()).decode()
        return json.loads(cursor_json)
    except Exception:
        raise ValueError("Invalid cursor format")


# Example usage decorators for common pagination patterns
def with_pagination(skip: int = 0, limit: int = 50):
    """Decorator to add pagination parameters to endpoint"""
    def decorator(func):
        func.__annotations__["skip"] = int
        func.__annotations__["limit"] = int
        return func
    return decorator
