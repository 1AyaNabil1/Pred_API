"""
MLflow Model Registry Proxy API

A FastAPI backend that provides:
1. Proxy endpoints to MLflow Model Registry REST API
2. Model serving invocation
3. PKL file upload with proper MLflow model wrapping and signature inference

References:
- MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
- MLflow REST API: https://mlflow.org/docs/latest/rest-api.html
- MLflow Model Serving: https://mlflow.org/docs/latest/models.html#local-rest-server
- MLflow Model Signature: https://mlflow.org/docs/latest/models.html#model-signature-and-input-example
- MLflow PyFunc: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
"""

import os
import json
import time
import pickle
import tempfile
import shutil
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import httpx
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec

try:
    import joblib
except ImportError:
    joblib = None

from config import settings
from logger import setup_logging, set_request_id, get_request_id
from auth import verify_api_key, RequestIdMiddleware, SecurityHeadersMiddleware
from consistency_api import router as consistency_router

# Import hardening router for production-ready endpoints
try:
    from hardening import router as hardening_router
    HARDENING_AVAILABLE = True
except ImportError:
    HARDENING_AVAILABLE = False


# Initialize logging
logger = setup_logging(settings.LOG_LEVEL)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    description="""
    MLflow Model Registry Proxy API
    
    This API provides:
    - Model Registry operations (list, create, update, delete)
    - Model version management
    - Stage transitions (Production, Staging, Archived)
    - PKL file upload with automatic signature inference
    - Model serving invocation
    
    References:
    - [MLflow Documentation](https://mlflow.org/docs/latest/)
    - [Model Registry](https://mlflow.org/docs/latest/model-registry.html)
    - [Model Serving](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)
    """
)

# Add rate limit exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add security middlewares
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIdMiddleware)

# CORS configuration with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include Consistency API router for MLflow sync
app.include_router(consistency_router)

# Constants
MAX_FILE_SIZE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024

# Include hardening router if available
if HARDENING_AVAILABLE:
    app.include_router(hardening_router, prefix="/api/v1", tags=["Hardening"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SearchModelsRequest(BaseModel):
    filter: Optional[str] = Field(None, max_length=500)
    max_results: Optional[int] = Field(100, ge=1, le=1000)


class CreateModelVersionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    source: str = Field(..., min_length=1, max_length=1000)
    run_id: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=5000)
    tags: Optional[list] = None


class GetModelVersionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(..., min_length=1, max_length=50)


class TransitionStageRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(..., min_length=1, max_length=50)
    stage: str = Field(..., min_length=1, max_length=50)
    archive_existing_versions: Optional[bool] = False
    
    @validator('stage')
    def validate_stage(cls, v):
        allowed_stages = ['None', 'Staging', 'Production', 'Archived']
        if v not in allowed_stages:
            raise ValueError(f'Stage must be one of: {", ".join(allowed_stages)}')
        return v


class DeleteModelRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class UpdateModelRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=5000)


class UpdateModelVersionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=5000)


class InvokeModelRequest(BaseModel):
    """
    Request model for invoking a served model.
    
    Reference: https://mlflow.org/docs/latest/models.html#local-rest-server
    
    The model_url should point to the /invocations endpoint of a running
    MLflow model serving instance.
    """
    model_config = {"protected_namespaces": ()}
    
    model_url: str = Field(..., min_length=1, max_length=500, 
                          description="URL of the model serving endpoint (e.g., http://localhost:5002/invocations)")
    inputs: Dict[str, Any] = Field(..., description="Input data for prediction")
    auth_token: Optional[str] = Field(None, max_length=1000, description="Optional auth token")
    input_format: Optional[str] = Field("auto", description="Input format: auto, inputs, dataframe_split, instances")


class PredictRequest(BaseModel):
    """
    Request model for direct predictions (loads model in-process).
    
    CANONICAL CONTRACT (NON-NEGOTIABLE):
    ------------------------------------
    The request body MUST conform to this exact format:
    
    {
        "inputs": [
            [x1, x2, x3, x4],
            [y1, y2, y3, y4]
        ]
    }
    
    Where:
    - "inputs" is a 2D list (list of lists)
    - Each row has exactly 4 numeric values
    - No additional nesting
    - No objects
    - No strings
    - No alternate keys (data, body, instances, dataframe_split, etc.)
    
    REJECTED FORMATS:
    - {"inputs": {"feature1": 1.0, ...}} - Objects not allowed
    - {"inputs": [1, 2, 3, 4]} - 1D array not allowed  
    - {"inputs": [[[1, 2, 3, 4]]]} - 3D array not allowed
    - {"data": [[1, 2, 3, 4]]} - Wrong key name
    - {"inputs": [[1, 2, 3]]} - Wrong row length (must be 4)
    - {"inputs": [["a", "b", "c", "d"]]} - Non-numeric values not allowed
    
    Reference: https://mlflow.org/docs/latest/models.html#rest-api
    """
    inputs: List[List[float]] = Field(
        ..., 
        description="2D array of numeric values. Each row must have exactly 4 features.",
        min_length=1
    )
    version: Optional[str] = Field(None, description="Model version (default: latest Production)")
    
    @field_validator('inputs')
    def validate_inputs_structure(cls, v):
        """
        Strict validation of inputs to ensure canonical format.
        """
        if not isinstance(v, list):
            raise ValueError(
                "inputs must be a 2D list. "
                "Expected format: [[x1, x2, x3, x4], [y1, y2, y3, y4]]"
            )
        
        if len(v) == 0:
            raise ValueError("inputs cannot be empty. Provide at least one row.")
        
        for row_idx, row in enumerate(v):
            if not isinstance(row, list):
                raise ValueError(
                    f"Row {row_idx} is not a list. "
                    "Each row must be a list of 4 numeric values. "
                    f"Got type: {type(row).__name__}"
                )
            
            if len(row) != 4:
                raise ValueError(
                    f"Row {row_idx} has {len(row)} values, expected exactly 4. "
                    "Each row must contain exactly 4 numeric features."
                )
            
            for col_idx, value in enumerate(row):
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Value at row {row_idx}, column {col_idx} is not numeric. "
                        f"Got type: {type(value).__name__}, value: {value}. "
                        "All values must be numeric (int or float)."
                    )
                
                # Check for NaN or Infinity
                if isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf')):
                    raise ValueError(
                        f"Value at row {row_idx}, column {col_idx} is invalid. "
                        "NaN and Infinity values are not allowed."
                    )
        
        return v


class InvokeByNameRequest(BaseModel):
    """
    Request model for invoking a model by name through the configured serving URL.
    
    This uses MODEL_SERVING_URL from configuration.
    """
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(..., min_length=1, max_length=255, description="Name of the registered model")
    inputs: Any = Field(..., description="Input data for prediction")
    stage: Optional[str] = Field("Production", description="Model stage to invoke")
    version: Optional[str] = Field(None, description="Specific version (overrides stage)")


# =============================================================================
# Helper Functions
# =============================================================================

def get_headers(authorization: Optional[str] = None) -> Dict[str, str]:
    """Build headers for MLflow API requests."""
    headers = {"Content-Type": "application/json"}
    
    token = authorization or settings.MLFLOW_AUTH_TOKEN
    if token:
        if token.startswith("Bearer "):
            headers["Authorization"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"
    
    return headers


def sanitize_error_message(error_text: str) -> str:
    """Sanitize error messages to prevent token/credential leakage."""
    sanitized = error_text
    sanitized = re.sub(r'Bearer\s+[A-Za-z0-9\-._~+/]+', 'Bearer [REDACTED]', sanitized)
    sanitized = re.sub(r'(api[_-]?key|token|password|secret)["\s:=]+[^\s"]+', 
                       r'\1=[REDACTED]', sanitized, flags=re.IGNORECASE)
    return sanitized


async def validate_file_upload(file: UploadFile) -> None:
    """Validate uploaded file before processing."""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_FILE_EXTENSIONS)}"
        )
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")


def load_pickle_file(pkl_path: str) -> Any:
    """Safely load a pickle file with multiple fallback methods."""
    with open(pkl_path, "rb") as f:
        file_data = f.read()
    
    load_methods = []
    
    if joblib:
        load_methods.append(("joblib.load", lambda: joblib.load(pkl_path)))
    
    load_methods.extend([
        ("pickle.load (default)", lambda: pickle.loads(file_data)),
        ("pickle.load (latin1)", lambda: pickle.loads(file_data, encoding='latin1')),
        ("pickle.load (bytes)", lambda: pickle.loads(file_data, encoding='bytes')),
    ])
    
    load_error = None
    for method_name, method in load_methods:
        try:
            model = method()
            logger.info(f"Successfully loaded model using: {method_name}")
            return model
        except Exception as e:
            load_error = str(e)
            logger.debug(f"{method_name} failed: {load_error}")
            continue
    
    raise ValueError(
        f"Failed to load pickle file: {sanitize_error_message(load_error)}. "
        "Try re-saving with joblib.dump() or pickle.dump(protocol=4)."
    )


def format_mlflow_payload(input_data: Any) -> Dict[str, Any]:
    """
    Format input data for MLflow serving endpoint.
    
    Reference: https://mlflow.org/docs/latest/models.html#local-rest-server
    
    MLflow supports these input formats:
    - {"inputs": [...]} - For tensor-based models
    - {"dataframe_split": {"columns": [...], "data": [...]}} - For DataFrame models
    - {"instances": [...]} - For record-oriented data
    - {"dataframe_records": [...]} - Deprecated but still supported
    
    Args:
        input_data: Raw input data
        
    Returns:
        Properly formatted payload for MLflow serving
    """
    # If already formatted, return as-is
    if isinstance(input_data, dict):
        if any(k in input_data for k in ['inputs', 'dataframe_split', 'instances', 'dataframe_records']):
            return input_data
        
        # Single record dict - wrap in dataframe_records
        return {"dataframe_records": [input_data]}
    
    elif isinstance(input_data, list):
        if len(input_data) > 0 and isinstance(input_data[0], dict):
            # List of records
            return {"instances": input_data}
        else:
            # Array/tensor data
            return {"inputs": input_data}
    
    elif isinstance(input_data, pd.DataFrame):
        return {
            "dataframe_split": {
                "columns": input_data.columns.tolist(),
                "data": input_data.values.tolist()
            }
        }
    
    elif isinstance(input_data, np.ndarray):
        return {"inputs": input_data.tolist()}
    
    else:
        return {"inputs": input_data}


# =============================================================================
# PyFunc Model Wrapper for .pkl files
# =============================================================================

class PickleModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper class for pickle models to be used with MLflow PyFunc.
    
    Reference: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
    
    This wrapper allows any pickle model with a predict() method to be
    served through MLflow's model serving infrastructure.
    """
    
    def __init__(self, model):
        """
        Initialize the wrapper with a model.
        
        Args:
            model: A model object with a predict() method
        """
        self.model = model
    
    def predict(self, context, model_input: Union[pd.DataFrame, np.ndarray, List]) -> Any:
        """
        Make predictions using the wrapped model.
        
        Args:
            context: MLflow context (unused but required by interface)
            model_input: Input data (DataFrame, array, or list)
            
        Returns:
            Model predictions
        """
        # Convert to numpy array if needed for sklearn-like models
        if isinstance(model_input, pd.DataFrame):
            input_array = model_input.values
        elif isinstance(model_input, list):
            input_array = np.array(model_input)
        else:
            input_array = model_input
        
        return self.model.predict(input_array)


def infer_signature_from_model(
    model: Any,
    input_example: Optional[Any] = None
) -> Optional[ModelSignature]:
    """
    Infer model signature from model and optional input example.
    
    Reference: https://mlflow.org/docs/latest/models.html#model-signature-and-input-example
    
    Args:
        model: Model object with predict() method
        input_example: Optional sample input data
        
    Returns:
        ModelSignature if inference succeeds, None otherwise
    """
    if input_example is None:
        logger.info("No input example provided, skipping signature inference")
        return None
    
    try:
        # Convert input example to appropriate format
        if isinstance(input_example, dict):
            # Check if it's a single record or multiple
            if all(isinstance(v, (list, np.ndarray)) for v in input_example.values()):
                # Multiple records as dict of lists
                input_df = pd.DataFrame(input_example)
            else:
                # Single record
                input_df = pd.DataFrame([input_example])
        elif isinstance(input_example, list):
            if len(input_example) > 0 and isinstance(input_example[0], dict):
                input_df = pd.DataFrame(input_example)
            else:
                input_df = pd.DataFrame(input_example)
        elif isinstance(input_example, np.ndarray):
            input_df = pd.DataFrame(input_example)
        elif isinstance(input_example, pd.DataFrame):
            input_df = input_example
        else:
            logger.warning(f"Unsupported input example type: {type(input_example)}")
            return None
        
        # Make a prediction to infer output
        try:
            if isinstance(input_example, pd.DataFrame):
                predictions = model.predict(input_example)
            elif isinstance(input_example, np.ndarray):
                predictions = model.predict(input_example)
            else:
                predictions = model.predict(input_df.values)
        except Exception as e:
            logger.warning(f"Could not make prediction for signature inference: {e}")
            # Create signature with only input schema
            signature = infer_signature(input_df, None)
            return signature
        
        # Infer full signature
        signature = infer_signature(input_df, predictions)
        logger.info(f"Inferred signature: inputs={signature.inputs}, outputs={signature.outputs}")
        return signature
        
    except Exception as e:
        logger.warning(f"Failed to infer signature: {e}")
        return None


# =============================================================================
# Application Lifecycle
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Validate configuration and connectivity on startup."""
    logger.info("Starting MLflow Model Registry Proxy...")
    logger.info(f"API Version: {settings.API_VERSION}")
    logger.info(f"MLflow Tracking URL: {settings.MLFLOW_BASE_URL}")
    logger.info(f"Model Serving URL: {settings.MODEL_SERVING_URL}")
    logger.info(f"Model Serving Enabled: {settings.MODEL_SERVING_ENABLED}")
    logger.info(f"Allowed Origins: {settings.ALLOWED_ORIGINS}")
    logger.info(f"API Key Auth: {'Enabled' if settings.API_KEY else 'Disabled'}")
    
    # Test MLflow connectivity
    try:
        async with httpx.AsyncClient(timeout=settings.HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{settings.MLFLOW_BASE_URL}/health")
            if response.status_code == 200:
                logger.info("✓ MLflow tracking server is reachable")
            else:
                logger.warning(f"⚠ MLflow server returned status {response.status_code}")
    except Exception as e:
        logger.error(f"✗ Cannot connect to MLflow tracking server: {str(e)}")
        logger.warning("Application will start but MLflow operations may fail")
    
    # Test Model Serving connectivity (if enabled)
    if settings.MODEL_SERVING_ENABLED:
        try:
            async with httpx.AsyncClient(timeout=settings.HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(f"{settings.MODEL_SERVING_URL}/health")
                if response.status_code == 200:
                    logger.info("✓ Model serving is reachable")
                else:
                    logger.warning(f"⚠ Model serving returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠ Model serving not available: {str(e)}")
            logger.info("  Start serving with: mlflow models serve --model-uri 'models:/ModelName/Production' --port 5002")


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def health(request: Request):
    """Health check endpoint with MLflow connectivity test."""
    request_id = set_request_id()
    logger.info("Health check requested")
    
    mlflow_status = "unknown"
    serving_status = "unknown"
    
    try:
        async with httpx.AsyncClient(timeout=settings.HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{settings.MLFLOW_BASE_URL}/health")
            mlflow_status = "connected" if response.status_code == 200 else "error"
    except Exception as e:
        mlflow_status = "unreachable"
        logger.warning(f"MLflow health check failed: {str(e)}")
    
    if settings.MODEL_SERVING_ENABLED:
        try:
            async with httpx.AsyncClient(timeout=settings.HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(f"{settings.MODEL_SERVING_URL}/health")
                serving_status = "connected" if response.status_code == 200 else "error"
        except Exception:
            serving_status = "not running"
    else:
        serving_status = "disabled"
    
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "mlflow_tracking_url": settings.MLFLOW_BASE_URL,
        "mlflow_status": mlflow_status,
        "model_serving_url": settings.MODEL_SERVING_URL,
        "model_serving_status": serving_status,
        "request_id": request_id
    }


@app.get("/serve/health")
async def serve_health():
    """Check model serving health specifically."""
    if not settings.MODEL_SERVING_ENABLED:
        return {"status": "disabled", "message": "Model serving is disabled in configuration"}
    
    try:
        async with httpx.AsyncClient(timeout=settings.HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{settings.MODEL_SERVING_URL}/health")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "serving_url": settings.MODEL_SERVING_URL,
                "status_code": response.status_code
            }
    except Exception as e:
        return {
            "status": "unreachable",
            "serving_url": settings.MODEL_SERVING_URL,
            "error": str(e),
            "hint": "Start model serving with: mlflow models serve --model-uri 'models:/ModelName/Production' --port 5002 --no-conda"
        }


# =============================================================================
# Model Registry Endpoints (Proxy to MLflow)
# =============================================================================

@app.post("/mlflow/registered-models/search")
@app.get("/mlflow/registered-models/search")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def search_registered_models(
    request: Request,
    body: Optional[SearchModelsRequest] = None,
    filter: Optional[str] = None,
    max_results: Optional[int] = None,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Search for registered models.
    
    Reference: https://mlflow.org/docs/latest/rest-api.html#search-registered-models
    """
    request_id = set_request_id()
    logger.info(f"Searching registered models (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/search"
    headers = get_headers(authorization)
    
    params = {}
    if body:
        if body.filter:
            params["filter"] = body.filter
        if body.max_results:
            params["max_results"] = body.max_results
    else:
        if filter:
            params["filter"] = filter
        if max_results:
            params["max_results"] = max_results
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            logger.info(f"Successfully retrieved models (request_id: {request_id})")
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"MLflow API error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {sanitize_error_message(e.response.text)}"
        )
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Request failed. Please try again.")


@app.get("/mlflow/model-versions/search")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def search_model_versions(
    request: Request,
    filter: str,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Search for model versions.
    
    Reference: https://mlflow.org/docs/latest/rest-api.html#search-model-versions
    """
    request_id = set_request_id()
    logger.info(f"Searching model versions (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/model-versions/search"
    headers = get_headers(authorization)
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            response = await client.get(url, params={"filter": filter}, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"MLflow API error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {sanitize_error_message(e.response.text)}"
        )
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Request failed. Please try again.")


@app.post("/mlflow/model-versions/create")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def create_model_version(
    request: Request,
    body: CreateModelVersionRequest,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Create/register a new model version.
    
    Reference: https://mlflow.org/docs/latest/rest-api.html#create-model-version
    """
    request_id = set_request_id()
    logger.info(f"Creating model version for '{body.name}' (request_id: {request_id})")
    
    headers = get_headers(authorization)
    
    # First, ensure the registered model exists
    try:
        create_model_url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/create"
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            response = await client.post(
                create_model_url,
                json={"name": body.name, "description": body.description or ""},
                headers=headers
            )
            if response.status_code == 200:
                logger.info(f"Created new registered model: {body.name}")
    except Exception as e:
        logger.debug(f"Model creation skipped (may already exist): {str(e)}")
    
    # Create the model version
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/model-versions/create"
    
    payload = {
        "name": body.name,
        "source": body.source,
    }
    if body.run_id:
        payload["run_id"] = body.run_id
    if body.description:
        payload["description"] = body.description
    if body.tags:
        payload["tags"] = body.tags
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(f"Successfully created model version (request_id: {request_id})")
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"MLflow API error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {sanitize_error_message(e.response.text)}"
        )
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Request failed. Please try again.")


@app.get("/mlflow/model-versions/get")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def get_model_version(
    request: Request,
    name: str,
    version: str,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Get model version metadata including signature and input_example.
    
    Reference: https://mlflow.org/docs/latest/rest-api.html#get-model-version
    """
    request_id = set_request_id()
    logger.info(f"Getting model version '{name}' v{version} (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/model-versions/get"
    headers = get_headers(authorization)
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            response = await client.get(url, params={"name": name, "version": version}, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"MLflow API error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {sanitize_error_message(e.response.text)}"
        )
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Request failed. Please try again.")


@app.post("/mlflow/model-versions/transition-stage")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def transition_model_stage(
    request: Request,
    body: TransitionStageRequest,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Transition model version to a different stage.
    
    Reference: https://mlflow.org/docs/latest/rest-api.html#transition-model-version-stage
    
    Valid stages: None, Staging, Production, Archived
    """
    request_id = set_request_id()
    logger.info(f"Transitioning '{body.name}' v{body.version} to {body.stage} (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/model-versions/transition-stage"
    headers = get_headers(authorization)
    
    payload = {
        "name": body.name,
        "version": body.version,
        "stage": body.stage,
        "archive_existing_versions": body.archive_existing_versions
    }
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(f"Successfully transitioned stage (request_id: {request_id})")
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"MLflow API error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {sanitize_error_message(e.response.text)}"
        )
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Request failed. Please try again.")


@app.delete("/mlflow/registered-models/delete")
@app.post("/mlflow/registered-models/delete")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def delete_registered_model(
    request: Request,
    name: Optional[str] = None,
    body: Optional[DeleteModelRequest] = None,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Delete a registered model.
    
    Reference: https://mlflow.org/docs/latest/rest-api.html#delete-registered-model
    """
    request_id = set_request_id()
    
    model_name = name if name else (body.name if body else None)
    
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")
    
    logger.info(f"Deleting model '{model_name}' (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/delete"
    headers = get_headers(authorization)
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            # MLflow delete API requires DELETE method with JSON body containing model name
            # Reference: https://mlflow.org/docs/latest/rest-api.html#delete-registered-model
            response = await client.request(
                method="DELETE",
                url=url,
                json={"name": model_name},
                headers=headers
            )
            response.raise_for_status()
            logger.info(f"Successfully deleted model (request_id: {request_id})")
            return {"message": f"Model '{model_name}' deleted successfully"}
    except httpx.HTTPStatusError as e:
        logger.error(f"MLflow API error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {sanitize_error_message(e.response.text)}"
        )
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Request failed. Please try again.")


@app.patch("/mlflow/registered-models/update")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def update_registered_model(
    request: Request,
    body: UpdateModelRequest,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Update registered model metadata.
    
    Reference: https://mlflow.org/docs/latest/rest-api.html#update-registered-model
    """
    request_id = set_request_id()
    logger.info(f"Updating model '{body.name}' (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/update"
    headers = get_headers(authorization)
    
    payload = {"name": body.name}
    if body.description is not None:
        payload["description"] = body.description
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            response = await client.patch(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(f"Successfully updated model (request_id: {request_id})")
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"MLflow API error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {sanitize_error_message(e.response.text)}"
        )
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Request failed. Please try again.")


@app.patch("/mlflow/model-versions/update")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def update_model_version(
    request: Request,
    body: UpdateModelVersionRequest,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Update model version metadata."""
    request_id = set_request_id()
    logger.info(f"Updating model version '{body.name}' v{body.version} (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/model-versions/update"
    headers = get_headers(authorization)
    
    payload = {"name": body.name, "version": body.version}
    if body.description is not None:
        payload["description"] = body.description
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            response = await client.patch(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"MLflow API error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {sanitize_error_message(e.response.text)}"
        )
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Request failed. Please try again.")


# =============================================================================
# Model Serving / Invocation Endpoints
# =============================================================================

@app.post("/serve/invoke")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def invoke_model(
    request: Request,
    body: InvokeModelRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Invoke a model serving endpoint.
    
    Reference: https://mlflow.org/docs/latest/models.html#local-rest-server
    
    This endpoint forwards requests to a running MLflow model serving instance.
    The model_url should point to the /invocations endpoint.
    
    Supported input formats:
    - {"inputs": [...]} - For tensor-based models
    - {"dataframe_split": {"columns": [...], "data": [...]}} - For DataFrame models
    - {"instances": [...]} - For record-oriented data
    """
    request_id = set_request_id()
    logger.info(f"Invoking model endpoint (request_id: {request_id})")
    
    if not body.model_url:
        raise HTTPException(status_code=400, detail="model_url is required")
    
    headers = {"Content-Type": "application/json"}
    if body.auth_token:
        if body.auth_token.startswith("Bearer "):
            headers["Authorization"] = body.auth_token
        else:
            headers["Authorization"] = f"Bearer {body.auth_token}"
    
    # Format payload for MLflow serving
    payload = format_mlflow_payload(body.inputs)
    
    logger.info(f"Sending payload format: {list(payload.keys())[0]} (request_id: {request_id})")
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=settings.MODEL_INVOKE_TIMEOUT) as client:
            response = await client.post(body.model_url, json=payload, headers=headers)
            response.raise_for_status()
            latency_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Model invoked successfully, latency: {latency_ms:.2f}ms (request_id: {request_id})")
            
            return {
                "predictions": response.json(),
                "latency_ms": round(latency_ms, 2),
                "status_code": response.status_code,
                "request_id": request_id
            }
    except httpx.HTTPStatusError as e:
        logger.error(f"Model serving error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Model serving error: {sanitize_error_message(e.response.text)}"
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Model serving not available. Start serving with: mlflow models serve --model-uri 'models:/ModelName/Production' --port 5002 --no-conda"
        )
    except Exception as e:
        logger.error(f"Invocation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Invocation failed. Please try again.")


@app.post("/serve/invoke/{model_name}")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def invoke_model_by_name(
    request: Request,
    model_name: str,
    body: InvokeByNameRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Invoke a model by name using the configured MODEL_SERVING_URL.
    
    This is a convenience endpoint that constructs the invocation URL
    from the model name and the MODEL_SERVING_URL configuration.
    
    Note: This assumes the model serving instance is serving the requested model.
    If you need to serve different models, you need multiple serving instances.
    """
    request_id = set_request_id()
    
    if not settings.MODEL_SERVING_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Model serving is disabled. Enable MODEL_SERVING_ENABLED in configuration."
        )
    
    # Construct the invocation URL
    invoke_url = f"{settings.MODEL_SERVING_URL}/invocations"
    
    # Format payload
    payload = format_mlflow_payload(body.inputs)
    
    logger.info(f"Invoking model '{model_name}' at {invoke_url} (request_id: {request_id})")
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=settings.MODEL_INVOKE_TIMEOUT) as client:
            response = await client.post(
                invoke_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "predictions": response.json(),
                "model_name": model_name,
                "latency_ms": round(latency_ms, 2),
                "request_id": request_id
            }
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Model serving error: {sanitize_error_message(e.response.text)}"
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Model serving not available at {settings.MODEL_SERVING_URL}"
        )
    except Exception as e:
        logger.error(f"Invocation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Invocation failed.")


@app.post("/predict/{model_name}")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def predict(
    request: Request,
    model_name: str,
    body: PredictRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Make predictions using a registered MLflow model (loads model in-process).
    
    CANONICAL INPUT CONTRACT (NON-NEGOTIABLE):
    ==========================================
    
    Request body MUST be:
    {
        "inputs": [
            [x1, x2, x3, x4],
            [y1, y2, y3, y4]
        ]
    }
    
    Requirements:
    - "inputs" is a 2D list (list of lists)
    - Each row has exactly 4 numeric values
    - No objects, strings, or alternate keys
    
    Example (CORRECT):
    curl -X POST http://localhost:8000/predict/MyModel \
      -H "Content-Type: application/json" \
      -d '{"inputs": [[5.1, 3.5, 1.4, 0.2], [6.0, 2.7, 5.1, 1.6]]}'
    
    This endpoint loads the model directly using mlflow.pyfunc.load_model()
    and makes predictions in the current process.
    
    Reference: https://mlflow.org/docs/latest/models.html#rest-api
    """
    request_id = set_request_id()
    logger.info(f"Prediction request for model '{model_name}' (request_id: {request_id})")
    
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_BASE_URL)
        
        # Determine model URI
        if body.version:
            model_uri = f"models:/{model_name}/{body.version}"
        else:
            model_uri = f"models:/{model_name}/Production"
        
        logger.info(f"Loading model from: {model_uri}")
        
        # Load the model
        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            # Try latest version if Production fails
            logger.warning(f"Failed to load from {model_uri}: {str(e)}")
            
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            model_versions = client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                raise HTTPException(
                    status_code=404,
                    detail=f"No versions found for model '{model_name}'"
                )
            
            latest_version = max(model_versions, key=lambda x: int(x.version)).version
            model_uri = f"models:/{model_name}/{latest_version}"
            logger.info(f"Using latest version: {latest_version}")
            model = mlflow.pyfunc.load_model(model_uri)
        
        # Make prediction - inputs are pre-validated as 2D list by PredictRequest
        # CANONICAL FORMAT: inputs is guaranteed to be List[List[float]] with 4 values per row
        start_time = time.time()
        
        # Direct conversion to numpy array - no wrapping, no reshaping
        prediction_input = np.array(body.inputs)
        
        predictions = model.predict(prediction_input)
        latency_ms = (time.time() - start_time) * 1000
        
        # Convert numpy arrays to lists
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        
        logger.info(f"Prediction completed in {latency_ms:.2f}ms (request_id: {request_id})")
        
        return {
            "predictions": predictions,
            "model_name": model_name,
            "model_uri": model_uri,
            "latency_ms": round(latency_ms, 2),
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {sanitize_error_message(str(e))}"
        )


# =============================================================================
# PKL Upload Endpoint with Proper Signature Handling
# =============================================================================

@app.post("/mlflow/upload-pkl-model")
@limiter.limit("10/hour")
async def upload_pkl_model(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(..., min_length=1, max_length=255),
    description: Optional[str] = Form(None, max_length=5000),
    input_example: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Upload a .pkl file and register it as a model in MLflow.
    
    This endpoint properly wraps pickle files as MLflow PyFunc models with:
    - Model signature (inferred from input_example if provided)
    - Input example
    - Proper artifact storage
    
    Reference: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
    
    SECURITY WARNING: Pickle files can execute arbitrary code.
    Only upload pickle files from trusted sources.
    
    Args:
        file: The .pkl file to upload
        model_name: Name for the registered model
        description: Optional model description
        input_example: Optional JSON string with sample input data for signature inference
    """
    request_id = set_request_id()
    logger.info(f"Uploading model '{model_name}' (request_id: {request_id})")
    
    # Validate file
    await validate_file_upload(file)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        pkl_path = os.path.join(temp_dir, "model.pkl")
        with open(pkl_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved, attempting to load model (request_id: {request_id})")
        
        # Load the model
        try:
            model = load_pickle_file(pkl_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Verify model has predict method
        if not hasattr(model, 'predict'):
            raise HTTPException(
                status_code=400,
                detail="Model must have a predict() method"
            )
        
        # Parse input example if provided
        input_ex = None
        if input_example:
            try:
                input_ex = json.loads(input_example)
                logger.info(f"Parsed input example: {type(input_ex)}")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="input_example must be valid JSON")
        
        # Infer signature from input example
        signature = None
        if input_ex is not None:
            signature = infer_signature_from_model(model, input_ex)
            if signature:
                logger.info(f"Inferred model signature: {signature}")
            else:
                logger.warning("Could not infer signature from input example")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_BASE_URL)
        
        # Create or get experiment
        experiment_name = "model_uploads"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment("Default")
            else:
                experiment_id = experiment.experiment_id
        
        # Start an MLflow run and log the model
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"upload_{model_name}"):
            run_id = mlflow.active_run().info.run_id
            
            # Prepare input example for logging
            log_input_example = None
            if input_ex is not None:
                if isinstance(input_ex, dict):
                    if all(isinstance(v, (list, np.ndarray)) for v in input_ex.values()):
                        log_input_example = pd.DataFrame(input_ex)
                    else:
                        log_input_example = pd.DataFrame([input_ex])
                elif isinstance(input_ex, list):
                    if len(input_ex) > 0 and isinstance(input_ex[0], dict):
                        log_input_example = pd.DataFrame(input_ex)
                    else:
                        log_input_example = np.array(input_ex)
                else:
                    log_input_example = input_ex
            
            # Try to log as sklearn model first (preserves more functionality)
            try:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=model_name,
                    signature=signature,
                    input_example=log_input_example
                )
                model_flavor = "sklearn"
                logger.info("Model logged as sklearn model with signature")
            except Exception as e:
                logger.info(f"sklearn logging failed ({str(e)}), wrapping as pyfunc")
                
                # Wrap in PyFunc model
                # Reference: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
                wrapper = PickleModelWrapper(model)
                
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=wrapper,
                    registered_model_name=model_name,
                    signature=signature,
                    input_example=log_input_example,
                    pip_requirements=["numpy", "pandas", "scikit-learn"]
                )
                model_flavor = "pyfunc"
                logger.info("Model logged as pyfunc model with signature")
        
        # Update model description if provided
        if description:
            try:
                update_url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/update"
                headers = get_headers(authorization)
                async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
                    await client.patch(
                        update_url,
                        json={"name": model_name, "description": description},
                        headers=headers
                    )
                    logger.info("Model description updated")
            except Exception as e:
                logger.warning(f"Failed to update description: {str(e)}")
        
        logger.info(f"Model '{model_name}' uploaded successfully (request_id: {request_id})")
        
        return {
            "success": True,
            "model_name": model_name,
            "run_id": run_id,
            "model_flavor": model_flavor,
            "has_signature": signature is not None,
            "has_input_example": input_ex is not None,
            "message": f"Model '{model_name}' uploaded and registered successfully",
            "model_uri": f"models:/{model_name}/1",
            "request_id": request_id,
            "next_steps": [
                f"View model: {settings.MLFLOW_BASE_URL}/#/models/{model_name}",
                f"Promote to Production: POST /mlflow/model-versions/transition-stage",
                f"Serve model: mlflow models serve --model-uri 'models:/{model_name}/Production' --port 5002 --no-conda"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {sanitize_error_message(str(e))}"
        )
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            logger.debug("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {str(e)}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
