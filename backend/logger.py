"""
Logging configuration for the MLflow Model Registry Proxy.
"""
import logging
import sys
from typing import Optional
from contextvars import ContextVar
import uuid

# Context variable for request ID tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up application logging with structured format.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("mlflow_proxy")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - [%(request_id)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add filter to include request_id
    class RequestIdFilter(logging.Filter):
        def filter(self, record):
            record.request_id = get_request_id() or '-'
            return True
    
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set the request ID for the current context.
    
    Args:
        request_id: Optional request ID. If not provided, generates a new UUID.
        
    Returns:
        The request ID that was set
    """
    rid = request_id or str(uuid.uuid4())[:8]
    request_id_var.set(rid)
    return rid


def get_request_id() -> str:
    """
    Get the current request ID from context.
    
    Returns:
        Current request ID or empty string
    """
    return request_id_var.get()
