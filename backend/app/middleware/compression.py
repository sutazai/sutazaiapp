"""Response Compression Middleware
Compress API responses to improve performance
"""

import gzip
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)


class GZipMiddleware(BaseHTTPMiddleware):
    """Compress responses with gzip if client supports it"""
    
    def __init__(self, app, minimum_size: int = 500, compression_level: int = 6):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
    
    async def dispatch(self, request: Request, call_next):
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        
        response = await call_next(request)
        
        # Don't compress if:
        # - Client doesn't accept gzip
        # - Response is streaming
        # - Content-Encoding already set
        # - Response too small
        if (
            "gzip" not in accept_encoding
            or isinstance(response, StreamingResponse)
            or "content-encoding" in response.headers
            or int(response.headers.get("content-length", self.minimum_size)) < self.minimum_size
        ):
            return response
        
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Only compress if worth it
        if len(body) < self.minimum_size:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        # Compress the response
        compressed_body = gzip.compress(body, compresslevel=self.compression_level)
        
        # Build new response with compressed content
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(compressed_body))
        response.headers["Vary"] = "Accept-Encoding"
        
        return Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
