"""
SutazAI API Routes

Secure API routes with comprehensive error handling and input validation.
"""

import logging
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar, cast

try:
    from flask import Flask, Response, jsonify, request
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    from marshmallow import Schema, ValidationError, fields
    from werkzeug.exceptions import HTTPException
except ImportError as e:
    raise ImportError(
        f"Required package not found. Please install required packages: {e}\n"
        "Run: pip install flask flask-limiter marshmallow werkzeug"
    )

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("/opt/sutazai_project/SutazAI/logs/api.log")
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

# Initialize Flask app with security headers
app = Flask(__name__)
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # Additional security settings
    PERMANENT_SESSION_LIFETIME=1800,  # 30 minutes
    SEND_FILE_MAX_AGE_DEFAULT=31536000,  # 1 year
)

# Configure rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379",
    strategy="fixed-window",  # or "moving-window"
)

# Type variables for routes
T = TypeVar('T')
RouteReturn = Union[Response, Tuple[Response, int]]
RouteDecorator = Callable[[Callable[..., RouteReturn]], Callable[..., RouteReturn]]

# Request validation schemas
class SearchSchema(Schema):
    """Validation schema for search requests."""
    q = fields.String(required=True, validate=lambda s: len(s.strip()) > 0)
    limit = fields.Integer(missing=10, validate=lambda n: 0 < n <= 100)
    offset = fields.Integer(missing=0, validate=lambda n: n >= 0)

    class Meta:
        """Schema metadata."""
        strict = True

class ResourceSchema(Schema):
    """Validation schema for resource requests."""
    id = fields.Integer(required=True, validate=lambda n: n > 0)
    fields = fields.Dict(
        keys=fields.String(),
        values=fields.Raw(allow_none=True),
        missing={"id": "string", "name": "string"},
        validate=lambda x: len(x) > 0
    )

    class Meta:
        """Schema metadata."""
        strict = True

def validate_schema(schema: Schema) -> RouteDecorator:
    """
    Decorator to validate request data against a schema.
    
    Args:
        schema: Marshmallow schema for validation
        
    Returns:
        Decorated function
    """
    def decorator(f: Callable[..., RouteReturn]) -> Callable[..., RouteReturn]:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> RouteReturn:
            try:
                # Validate request data
                if request.is_json:
                    data = schema.load(request.get_json())
                else:
                    data = schema.load(request.args)
                return f(*args, data=data, **kwargs)
            except ValidationError as err:
                logger.warning(f"Validation error: {err.messages}")
                return cast(RouteReturn, (jsonify({
                    "error": "Validation failed",
                    "messages": err.messages,
                    "status_code": 400
                }), 400))
        return decorated_function
    return decorator

def handle_error(error: Union[HTTPException, Exception]) -> RouteReturn:
    """
    Global error handler for all routes.
    
    Args:
        error: Exception that was raised
        
    Returns:
        Error response and status code
    """
    try:
        if isinstance(error, HTTPException):
            status_code = getattr(error, 'code', 500)
            error_message = getattr(error, 'description', str(error))
        else:
            status_code = 500
            error_message = "Internal server error"
        
        # Log the error with traceback for non-HTTP exceptions
        if not isinstance(error, HTTPException):
            logger.error(
                f"Unhandled error: {error}\n"
                f"Traceback: {traceback.format_exc()}"
            )
        else:
            logger.error(f"HTTP error: {error}")
        
        response = {
            "error": error_message,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat()
        }
        
        # Include debug information in development
        if app.debug and status_code == 500:
            response.update({
                "debug_info": {
                    "error_type": error.__class__.__name__,
                    "traceback": traceback.format_exc()
                }
            })
        
        return cast(RouteReturn, (jsonify(response), status_code))
        
    except Exception as e:
        # Fallback error handler
        logger.critical(
            f"Error in error handler: {e}\n"
            f"Original error: {error}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        return cast(RouteReturn, (jsonify({
            "error": "Critical server error",
            "status_code": 500
        }), 500))

# Register error handler
app.errorhandler(Exception)(handle_error)

@app.route("/search")
@limiter.limit("30/minute")
@validate_schema(SearchSchema())
def search(data: Dict[str, Any]) -> RouteReturn:
    """
    Search endpoint with validation and rate limiting.
    
    Args:
        data: Validated request data
        
    Returns:
        Search results response
    """
    try:
        query = str(data["q"])
        limit = int(data["limit"])
        offset = int(data["offset"])
        
        logger.info(f"Processing search request: query='{query}' limit={limit} offset={offset}")
        
        # TODO: Implement actual search logic here
        results: List[Dict[str, Any]] = [
            {"id": i, "title": f"Result {i} for '{query}'"}
            for i in range(offset, offset + limit)
        ]
        
        return cast(RouteReturn, jsonify({
            "query": query,
            "limit": limit,
            "offset": offset,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }))
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise

@app.route("/api/resource/<int:resource_id>")
@limiter.limit("10/minute")
@validate_schema(ResourceSchema())
def get_resource(resource_id: int, data: Dict[str, Any]) -> RouteReturn:
    """
    Get resource endpoint with validation and rate limiting.
    
    Args:
        resource_id: Resource ID from URL
        data: Validated request data
        
    Returns:
        Resource data response
    """
    try:
        fields = data.get("fields", {})
        logger.info(f"Fetching resource {resource_id} with fields {fields}")
        
        # TODO: Implement actual resource fetching logic here
        resource: Dict[str, Any] = {
            "id": resource_id,
            "name": f"Resource {resource_id}",
            "description": "Sample resource description",
            "created_at": datetime.now().isoformat()
        }
        
        # Filter fields
        filtered_resource = {
            k: v for k, v in resource.items()
            if k in fields
        }
        
        if not filtered_resource:
            logger.warning(f"No matching fields found for resource {resource_id}")
            return cast(RouteReturn, (jsonify({
                "error": "No matching fields found",
                "status_code": 404
            }), 404))
        
        return cast(RouteReturn, jsonify(filtered_resource))
        
    except Exception as e:
        logger.error(f"Failed to fetch resource {resource_id}: {str(e)}")
        raise

@app.before_request
def log_request_info() -> None:
    """Log information about incoming requests."""
    logger.info(
        f"Request: {request.method} {request.url} "
        f"(IP: {request.remote_addr}, User-Agent: {request.user_agent})"
    )

@app.after_request
def add_security_headers(response: Response) -> Response:
    """Add security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response
