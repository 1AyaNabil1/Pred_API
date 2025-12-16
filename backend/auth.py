"""
Authentication and security middleware for the MLflow Model Registry Proxy.
"""
import uuid
from typing import Optional
from fastapi import Request, HTTPException, Header
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config import settings
from logger import set_request_id


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4())[:8])
        set_request_id(request_id)
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers['X-Request-ID'] = request_id
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Remove server header
        if 'server' in response.headers:
            del response.headers['server']
        
        return response


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
) -> Optional[str]:
    """
    Verify API key authentication if configured.
    
    Args:
        x_api_key: API key from X-API-Key header
        authorization: Bearer token from Authorization header
        
    Returns:
        The verified API key or None if auth is disabled
        
    Raises:
        HTTPException: If API key is required but invalid
    """
    # If no API key is configured, allow all requests
    if not settings.API_KEY:
        return None
    
    # Check X-API-Key header
    if x_api_key and x_api_key == settings.API_KEY:
        return x_api_key
    
    # Check Authorization header for Bearer token
    if authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            if token == settings.API_KEY:
                return token
        elif authorization == settings.API_KEY:
            return authorization
    
    # API key required but not provided or invalid
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"}
    )
