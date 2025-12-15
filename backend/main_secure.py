import os
import json
import time
import pickle
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import httpx
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

try:
    import joblib
except ImportError:
    joblib = None

from config import settings
from logger import setup_logging, set_request_id, get_request_id
from auth import verify_api_key, RequestIdMiddleware, SecurityHeadersMiddleware


# Initialize logging
logger = setup_logging(settings.LOG_LEVEL)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
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

# Constants
MAX_FILE_SIZE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


# Request/Response models with validation
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
    model_config = {"protected_namespaces": ()}
    
    model_url: str = Field(..., min_length=1, max_length=500)
    inputs: Dict[str, Any]
    auth_token: Optional[str] = Field(None, max_length=1000)


class PredictRequest(BaseModel):
    """Request model for predictions."""
    inputs: Any = Field(..., description="Input data for prediction (array, list, or dict)")
    version: Optional[str] = Field(None, description="Model version (default: latest Production)")


def get_headers(authorization: Optional[str] = None) -> Dict[str, str]:
    """Build headers for MLflow API requests."""
    headers = {"Content-Type": "application/json"}
    
    # Use provided auth token, fallback to env, or none
    token = authorization or settings.MLFLOW_AUTH_TOKEN
    if token:
        if token.startswith("Bearer "):
            headers["Authorization"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"
    
    return headers


def sanitize_error_message(error_text: str) -> str:
    """
    Sanitize error messages to prevent token/credential leakage.
    
    Args:
        error_text: Original error message
    
    Returns:
        Sanitized error message
    """
    # Remove potential tokens and sensitive information
    sanitized = error_text
    
    # Remove Bearer tokens
    import re
    sanitized = re.sub(r'Bearer\s+[A-Za-z0-9\-._~+/]+', 'Bearer [REDACTED]', sanitized)
    
    # Remove potential API keys
    sanitized = re.sub(r'(api[_-]?key|token|password|secret)["\s:=]+[^\s"]+', r'\1=[REDACTED]', sanitized, flags=re.IGNORECASE)
    
    return sanitized


async def validate_file_upload(file: UploadFile) -> None:
    """
    Validate uploaded file before processing.
    
    Args:
        file: Uploaded file
    
    Raises:
        HTTPException: If file is invalid
    """
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_FILE_EXTENSIONS)}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")


def load_pickle_file(pkl_path: str) -> Any:
    """
    Safely load a pickle file with multiple fallback methods.
    
    Args:
        pkl_path: Path to pickle file
    
    Returns:
        Loaded model object
    
    Raises:
        ValueError: If file cannot be loaded
    """
    # Read file data once
    with open(pkl_path, "rb") as f:
        file_data = f.read()
    
    # Try different pickle loading methods
    load_methods = []
    
    # Joblib is more robust for sklearn models
    if joblib:
        load_methods.append(("joblib.load", lambda: joblib.load(pkl_path)))
    
    # Standard pickle methods with different options
    load_methods.extend([
        ("pickle.load (default)", lambda: pickle.loads(file_data)),
        ("pickle.load (latin1)", lambda: pickle.loads(file_data, encoding='latin1')),
        ("pickle.load (bytes)", lambda: pickle.loads(file_data, encoding='bytes')),
        ("pickle.load (fix_imports)", lambda: pickle.loads(file_data, fix_imports=True, encoding='latin1')),
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
    
    # All methods failed
    raise ValueError(
        f"Failed to load pickle file: {sanitize_error_message(load_error)}. "
        "The file may be corrupted or created with an incompatible Python/pickle version. "
        "Try re-saving with joblib.dump() or pickle.dump(protocol=4)."
    )


@app.on_event("startup")
async def startup_event():
    """Validate configuration and connectivity on startup."""
    logger.info("Starting MLflow Model Registry Proxy...")
    logger.info(f"API Version: {settings.API_VERSION}")
    logger.info(f"MLflow Base URL: {settings.MLFLOW_BASE_URL}")
    logger.info(f"Allowed Origins: {settings.ALLOWED_ORIGINS}")
    logger.info(f"API Key Auth: {'Enabled' if settings.API_KEY else 'Disabled'}")
    
    # Test MLflow connectivity
    try:
        async with httpx.AsyncClient(timeout=settings.HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{settings.MLFLOW_BASE_URL}/health")
            if response.status_code == 200:
                logger.info("✓ MLflow server is reachable")
            else:
                logger.warning(f"⚠ MLflow server returned status {response.status_code}")
    except Exception as e:
        logger.error(f"✗ Cannot connect to MLflow server: {str(e)}")
        logger.warning("Application will start but MLflow operations may fail")


@app.get("/health")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def health(request: Request):
    """Health check endpoint with MLflow connectivity test."""
    request_id = set_request_id()
    logger.info("Health check requested")
    
    mlflow_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=settings.HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{settings.MLFLOW_BASE_URL}/health")
            mlflow_status = "connected" if response.status_code == 200 else "error"
    except Exception as e:
        mlflow_status = "unreachable"
        logger.warning(f"MLflow health check failed: {str(e)}")
    
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "mlflow_base_url": settings.MLFLOW_BASE_URL,
        "mlflow_status": mlflow_status,
        "request_id": request_id
    }


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
    Supports both GET and POST methods.
    """
    request_id = set_request_id()
    logger.info(f"Searching registered models (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/search"
    headers = get_headers(authorization)
    
    # Handle both GET and POST parameters
    params = {}
    if body:  # POST request with body
        if body.filter:
            params["filter"] = body.filter
        if body.max_results:
            params["max_results"] = body.max_results
    else:  # GET request with query params
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
    Automatically creates the registered model if it doesn't exist.
    """
    request_id = set_request_id()
    logger.info(f"Creating model version for '{body.name}' (request_id: {request_id})")
    
    headers = get_headers(authorization)
    
    # First, try to create the registered model if it doesn't exist
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
    except httpx.HTTPStatusError as e:
        # Model might already exist (409 or 400)
        if e.response.status_code in [400, 409]:
            logger.debug(f"Model '{body.name}' already exists, proceeding to create version")
        else:
            logger.warning(f"Unexpected error creating model: {e.response.status_code}")
    except Exception as e:
        logger.warning(f"Error creating model (will attempt version creation): {str(e)}")
    
    # Now create the model version
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
        logger.error(f"MLflow API error creating version: {e.response.status_code}")
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
    """Get model version metadata including signature and input_example."""
    request_id = set_request_id()
    logger.info(f"Getting model version '{name}' v{version} (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/model-versions/get"
    headers = get_headers(authorization)
    
    params = {"name": name, "version": version}
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            logger.info(f"Successfully retrieved model version (request_id: {request_id})")
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
    """Transition model version to a different stage (Production/Staging/Archived)."""
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
    """Delete a registered model."""
    request_id = set_request_id()
    
    # Get model name from either query param or request body
    model_name = name if name else (body.name if body else None)
    
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")
    
    logger.info(f"Deleting model '{model_name}' (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/delete"
    headers = get_headers(authorization)
    
    payload = {"name": model_name}
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            request_obj = client.build_request("DELETE", url, json=payload, headers=headers)
            response = await client.send(request_obj)
            response.raise_for_status()
            logger.info(f"Successfully deleted model (request_id: {request_id})")
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


@app.patch("/mlflow/registered-models/update")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def update_registered_model(
    request: Request,
    body: UpdateModelRequest,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Update registered model metadata."""
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
    """Update model version metadata (e.g., description)."""
    request_id = set_request_id()
    logger.info(f"Updating model version '{body.name}' v{body.version} (request_id: {request_id})")
    
    url = f"{settings.MLFLOW_BASE_URL}/api/2.0/mlflow/model-versions/update"
    headers = get_headers(authorization)
    
    payload = {
        "name": body.name,
        "version": body.version
    }
    if body.description is not None:
        payload["description"] = body.description
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
            response = await client.patch(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(f"Successfully updated model version (request_id: {request_id})")
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


@app.post("/serve/invoke")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def invoke_model(
    request: Request,
    body: InvokeModelRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Invoke model serving endpoint and return response with latency."""
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
    
    payload = {"inputs": body.inputs}
    
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
        logger.error(f"Model serving error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Model serving error: {sanitize_error_message(e.response.text)}"
        )
    except Exception as e:
        logger.error(f"Invocation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Invocation failed. Please try again.")


@app.post("/predict/{model_name}")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def predict(
    request: Request,
    model_name: str,
    body: PredictRequest,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Make predictions using a registered MLflow model.
    
    Args:
        model_name: Name of the registered model
        body: Prediction request with inputs and optional version
        
    Returns:
        Predictions from the model
    """
    request_id = set_request_id()
    logger.info(f"Prediction request for model '{model_name}' (request_id: {request_id})")
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_BASE_URL)
        
        # Determine which model version to use
        if body.version:
            model_uri = f"models:/{model_name}/{body.version}"
            logger.info(f"Using model version: {body.version}")
        else:
            # Use Production stage by default
            model_uri = f"models:/{model_name}/Production"
            logger.info("Using Production stage model")
        
        # Load the model
        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            # If Production not found, try latest version
            logger.warning(f"Failed to load from Production: {str(e)}")
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            model_versions = client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                raise HTTPException(
                    status_code=404,
                    detail=f"No versions found for model '{model_name}'"
                )
            
            latest_version = model_versions[0].version
            model_uri = f"models:/{model_name}/{latest_version}"
            logger.info(f"Using latest version: {latest_version}")
            model = mlflow.pyfunc.load_model(model_uri)
        
        # Make prediction
        start_time = time.time()
        
        # Convert inputs to numpy array if it's a list
        import numpy as np
        if isinstance(body.inputs, list):
            prediction_input = np.array(body.inputs)
        else:
            prediction_input = body.inputs
            
        predictions = model.predict(prediction_input)
        latency_ms = (time.time() - start_time) * 1000
        
        # Convert numpy arrays to lists for JSON serialization
        import numpy as np
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


@app.post("/mlflow/upload-pkl-model")
@limiter.limit("10/hour")  # More restrictive limit for uploads
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
    
    SECURITY WARNING: Pickle files can execute arbitrary code.
    Only upload pickle files from trusted sources.
    """
    request_id = set_request_id()
    logger.info(f"Uploading model '{model_name}' (request_id: {request_id})")
    
    # Validate file
    await validate_file_upload(file)
    
    # Create temporary directory for model artifacts
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        pkl_path = os.path.join(temp_dir, "model.pkl")
        with open(pkl_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved, attempting to load model (request_id: {request_id})")
        
        # Load the model with multiple fallback methods
        try:
            model = load_pickle_file(pkl_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Parse input example if provided
        input_ex = None
        if input_example:
            try:
                input_ex = json.loads(input_example)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="input_example must be valid JSON")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_BASE_URL)
        
        # Create or get an active experiment
        experiment_name = "model_uploads"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        except Exception:
            # Experiment already exists, get it
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment("Default")
            else:
                experiment_id = experiment.experiment_id
        
        # Start an MLflow run and log the model
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"upload_{model_name}"):
            # Try to log as sklearn model first, fallback to pyfunc
            try:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=model_name,
                    input_example=input_ex
                )
                logger.info("Model logged as sklearn model")
            except Exception as e:
                logger.info(f"sklearn logging failed, trying pyfunc: {str(e)}")
                # If sklearn logging fails, try pyfunc
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=model,
                    registered_model_name=model_name,
                    input_example=input_ex
                )
                logger.info("Model logged as pyfunc model")
            
            run_id = mlflow.active_run().info.run_id
        
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
            "message": f"Model '{model_name}' uploaded and registered successfully",
            "request_id": request_id
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
        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.debug("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
