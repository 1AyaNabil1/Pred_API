"""
Logging configuration for the MLflow Model Registry Proxy.
Provides structured logging with request correlation IDs.
"""
import logging
import sys
from datetime import datetime
from typing import Optional
import uuid
from contextvars import ContextVar

# Context variable for request ID
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class RequestIdFilter(logging.Filter):
    """Add request ID to log records."""
    
    def filter(self, record):
        record.request_id = request_id_var.get() or "N/A"
        return True


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("mlflow_proxy")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add filter and formatter
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID for current context.
    
    Args:
        request_id: Optional request ID, generates new UUID if not provided
    
    Returns:
        The request ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    request_id_var.set(request_id)
    return request_id


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()
