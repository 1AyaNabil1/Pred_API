"""
Authentication and security middleware for the API.
"""
from typing import Optional
from fastapi import Header, HTTPException, Request
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config import settings
from logger import set_request_id, get_request_id


# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """
    Verify API key if authentication is enabled.
    
    Args:
        x_api_key: API key from request header
    
    Raises:
        HTTPException: If API key is invalid or missing
    """
    # Skip verification if API key is not configured
    if not settings.API_KEY:
        return
    
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required. Provide X-API-Key header."
        )
    
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Set request ID for logging
        request_id = request.headers.get("X-Request-ID") or set_request_id()
        
        # Call next middleware/endpoint
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
