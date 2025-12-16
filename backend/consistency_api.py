"""
Consistency Layer API Endpoints
================================

This module provides FastAPI endpoints that enforce the consistency contract
between the Pixonal frontend and MLflow Model Registry.

WRITE-PATH ENFORCEMENT (Critical):
All the following actions MUST go through these endpoints:
- Register model → /api/registry/models (POST)
- Delete model → /api/registry/models/{name} (DELETE)
- Upload .pkl → /api/registry/models/upload (POST)
- Transition stage → /api/registry/models/{name}/versions/{version}/stage (POST)
- Update model → /api/registry/models/{name} (PATCH)

FORBIDDEN:
- Writing via MLflow UI (eventual consistency applies)
- Writing via MLflow CLI (outside Pixonal)
- Direct database manipulation

RECONCILIATION LOGIC:
- Backend ALWAYS re-queries MLflow
- Backend diffs results in memory
- Returns FULL, CURRENT state to UI
- Frontend REPLACES local state (never merge)

References:
- MLflow REST API: https://mlflow.org/docs/latest/rest-api.html
- Model Registry: https://mlflow.org/docs/latest/model-registry.html
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

from registry_sync import (
    RegistrySyncService,
    get_registry_service,
    SyncMetadata,
    RegisteredModel,
    ModelVersion
)

logger = logging.getLogger(__name__)

# Configuration from environment
MLFLOW_BASE_URL = os.getenv("MLFLOW_BASE_URL", "http://localhost:5001")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "30.0"))
MLFLOW_AUTH_TOKEN = os.getenv("MLFLOW_AUTH_TOKEN", "")

# Create router
router = APIRouter(prefix="/mlflow/registry", tags=["Registry Sync"])


# ==============================================================================
# Pydantic Models for Request/Response
# ==============================================================================

class CreateModelRequest(BaseModel):
    """Request to create a new registered model."""
    name: str = Field(..., description="Unique model name")
    description: Optional[str] = Field(None, description="Model description")
    tags: Optional[List[Dict[str, str]]] = Field(None, description="Model tags")


class CreateModelVersionRequest(BaseModel):
    """Request to create a new model version."""
    name: str = Field(..., description="Registered model name")
    source: str = Field(..., description="URI of the model artifact")
    run_id: Optional[str] = Field(None, description="MLflow run ID")
    description: Optional[str] = Field(None, description="Version description")
    tags: Optional[List[Dict[str, str]]] = Field(None, description="Version tags")


class TransitionStageRequest(BaseModel):
    """Request to transition model version stage."""
    stage: str = Field(..., description="Target stage: None, Staging, Production, Archived")
    archive_existing_versions: bool = Field(
        False, 
        description="Archive other versions in the target stage"
    )


class UpdateModelRequest(BaseModel):
    """Request to update a model's metadata."""
    description: Optional[str] = Field(None, description="New description")


class UpdateVersionRequest(BaseModel):
    """Request to update a model version's metadata."""
    description: Optional[str] = Field(None, description="New description")


class SyncResponse(BaseModel):
    """Response containing models and sync metadata."""
    models: List[Dict[str, Any]]
    sync_status: Dict[str, Any]


class ModelResponse(BaseModel):
    """Response for a single model operation."""
    model: Dict[str, Any]
    sync_status: Dict[str, Any]


class VersionResponse(BaseModel):
    """Response for a model version operation."""
    version: Dict[str, Any]
    sync_status: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response with consistency information."""
    error: str
    detail: str
    sync_status: Optional[Dict[str, Any]] = None


# ==============================================================================
# Dependency Injection
# ==============================================================================

def get_sync_service() -> RegistrySyncService:
    """Get the singleton RegistrySyncService instance."""
    return get_registry_service(
        mlflow_base_url=MLFLOW_BASE_URL,
        timeout=API_TIMEOUT,
        auth_token=MLFLOW_AUTH_TOKEN
    )


# ==============================================================================
# Sync Endpoints (Read Operations)
# ==============================================================================

@router.get("/sync", response_model=SyncResponse)
async def sync_registry(
    force_refresh: bool = Query(
        False,
        description="Force refresh bypassing short-lived cache"
    ),
    authorization: Optional[str] = Header(None),
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Synchronize with MLflow Model Registry.
    
    This endpoint:
    1. Fetches ALL registered models from MLflow
    2. Fetches ALL model versions from MLflow
    3. Returns COMPLETE, AUTHORITATIVE state
    4. Includes sync metadata for transparency
    
    The frontend should REPLACE its local state with this response.
    
    MLflow API: POST /api/2.0/mlflow/registered-models/search
    Reference: https://mlflow.org/docs/latest/rest-api.html#search-registered-models
    """
    try:
        models, sync_metadata = await sync_service.fetch_registered_models(
            authorization=authorization,
            force_refresh=force_refresh
        )
        
        return SyncResponse(
            models=[m.to_dict() for m in models],
            sync_status=sync_metadata.to_dict()
        )
        
    except Exception as e:
        logger.error(f"Registry sync failed: {str(e)}")
        sync_status = sync_service.get_sync_status()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Registry sync failed",
                "detail": str(e),
                "sync_status": sync_status.to_dict() if sync_status else None
            }
        )


@router.get("/sync/status")
async def get_sync_status(
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Get the current sync status without fetching models.
    
    Useful for monitoring data freshness.
    """
    sync_status = sync_service.get_sync_status()
    
    if sync_status is None:
        return {
            "synced": False,
            "message": "No sync has been performed yet"
        }
    
    return {
        "synced": True,
        "sync_status": sync_status.to_dict(),
        "is_stale": sync_service.is_data_stale()
    }


@router.get("/models/{model_name}/versions")
async def get_model_versions(
    model_name: str,
    authorization: Optional[str] = Header(None),
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Fetch all versions for a specific model.
    
    MLflow API: POST /api/2.0/mlflow/model-versions/search
    Reference: https://mlflow.org/docs/latest/rest-api.html#search-model-versions
    """
    try:
        versions = await sync_service.fetch_model_versions(
            model_name=model_name,
            authorization=authorization
        )
        
        return {
            "model_name": model_name,
            "versions": [v.to_dict() for v in versions],
            "total_versions": len(versions),
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch versions for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/versions/{version}")
async def get_model_version_details(
    model_name: str,
    version: str,
    authorization: Optional[str] = Header(None),
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Get detailed metadata for a specific model version.
    
    MLflow API: GET /api/2.0/mlflow/model-versions/get
    Reference: https://mlflow.org/docs/latest/rest-api.html#get-model-version
    """
    try:
        details = await sync_service.get_model_version_details(
            model_name=model_name,
            version=version,
            authorization=authorization
        )
        
        return details
        
    except Exception as e:
        logger.error(f"Failed to fetch version {version} for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Write Endpoints (Must go through backend)
# ==============================================================================

@router.post("/models")
async def create_model(
    request: CreateModelRequest,
    authorization: Optional[str] = Header(None),
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Create a new registered model in MLflow.
    
    WRITE-PATH ENFORCEMENT: All model creation MUST use this endpoint.
    
    MLflow API: POST /api/2.0/mlflow/registered-models/create
    Reference: https://mlflow.org/docs/latest/rest-api.html#create-registered-model
    """
    try:
        result = await sync_service.create_registered_model(
            name=request.name,
            description=request.description,
            tags=request.tags,
            authorization=authorization
        )
        
        # Fetch updated sync status
        sync_status = sync_service.get_sync_status()
        
        logger.info(f"Created registered model: {request.name}")
        
        return {
            "created": True,
            "model": result,
            "sync_status": sync_status.to_dict() if sync_status else None
        }
        
    except Exception as e:
        logger.error(f"Failed to create model {request.name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    authorization: Optional[str] = Header(None),
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Delete a registered model from MLflow.
    
    WRITE-PATH ENFORCEMENT: All model deletion MUST use this endpoint.
    
    MLflow API: DELETE /api/2.0/mlflow/registered-models/delete
    Reference: https://mlflow.org/docs/latest/rest-api.html#delete-registered-model
    """
    try:
        await sync_service.delete_registered_model(
            name=model_name,
            authorization=authorization
        )
        
        # Fetch updated sync status
        sync_status = sync_service.get_sync_status()
        
        logger.info(f"Deleted registered model: {model_name}")
        
        return {
            "deleted": True,
            "model_name": model_name,
            "sync_status": sync_status.to_dict() if sync_status else None
        }
        
    except Exception as e:
        logger.error(f"Failed to delete model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/versions")
async def create_model_version(
    model_name: str,
    request: CreateModelVersionRequest,
    authorization: Optional[str] = Header(None),
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Create a new model version in MLflow.
    
    WRITE-PATH ENFORCEMENT: All version creation MUST use this endpoint.
    
    MLflow API: POST /api/2.0/mlflow/model-versions/create
    Reference: https://mlflow.org/docs/latest/rest-api.html#create-model-version
    """
    try:
        result = await sync_service.create_model_version(
            name=model_name,
            source=request.source,
            run_id=request.run_id,
            description=request.description,
            tags=request.tags,
            authorization=authorization
        )
        
        sync_status = sync_service.get_sync_status()
        
        logger.info(f"Created version for model: {model_name}")
        
        return {
            "created": True,
            "version": result,
            "sync_status": sync_status.to_dict() if sync_status else None
        }
        
    except Exception as e:
        logger.error(f"Failed to create version for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/versions/{version}/stage")
async def transition_stage(
    model_name: str,
    version: str,
    request: TransitionStageRequest,
    authorization: Optional[str] = Header(None),
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Transition a model version to a new stage.
    
    WRITE-PATH ENFORCEMENT: All stage transitions MUST use this endpoint.
    
    Valid stages: None, Staging, Production, Archived
    
    MLflow API: POST /api/2.0/mlflow/model-versions/transition-stage
    Reference: https://mlflow.org/docs/latest/rest-api.html#transition-model-version-stage
    
    NOTE: MLflow does NOT auto-reload serving on stage change.
    If using MLflow Model Serving, you must restart the serving container.
    """
    valid_stages = {"None", "Staging", "Production", "Archived"}
    if request.stage not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage. Must be one of: {valid_stages}"
        )
    
    try:
        result = await sync_service.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=request.stage,
            archive_existing_versions=request.archive_existing_versions,
            authorization=authorization
        )
        
        sync_status = sync_service.get_sync_status()
        
        logger.info(
            f"Transitioned {model_name} v{version} to stage: {request.stage}"
        )
        
        return {
            "transitioned": True,
            "model_name": model_name,
            "version": version,
            "new_stage": request.stage,
            "result": result,
            "sync_status": sync_status.to_dict() if sync_status else None,
            # Important warning about serving
            "warning": (
                "If using MLflow Model Serving, you may need to restart the "
                "serving container to pick up the new stage. MLflow does NOT "
                "auto-reload on stage changes."
            )
        }
        
    except Exception as e:
        logger.error(
            f"Failed to transition {model_name} v{version} to {request.stage}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/models/{model_name}")
async def update_model(
    model_name: str,
    request: UpdateModelRequest,
    authorization: Optional[str] = Header(None),
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Update a registered model's metadata.
    
    WRITE-PATH ENFORCEMENT: All model updates MUST use this endpoint.
    
    MLflow API: PATCH /api/2.0/mlflow/registered-models/update
    Reference: https://mlflow.org/docs/latest/rest-api.html#update-registered-model
    """
    try:
        result = await sync_service.update_registered_model(
            name=model_name,
            description=request.description,
            authorization=authorization
        )
        
        sync_status = sync_service.get_sync_status()
        
        logger.info(f"Updated model: {model_name}")
        
        return {
            "updated": True,
            "model": result,
            "sync_status": sync_status.to_dict() if sync_status else None
        }
        
    except Exception as e:
        logger.error(f"Failed to update model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/models/{model_name}/versions/{version}")
async def update_model_version(
    model_name: str,
    version: str,
    request: UpdateVersionRequest,
    authorization: Optional[str] = Header(None),
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Update a model version's metadata.
    
    WRITE-PATH ENFORCEMENT: All version updates MUST use this endpoint.
    
    MLflow API: PATCH /api/2.0/mlflow/model-versions/update
    Reference: https://mlflow.org/docs/latest/rest-api.html#update-model-version
    """
    try:
        result = await sync_service.update_model_version(
            name=model_name,
            version=version,
            description=request.description,
            authorization=authorization
        )
        
        sync_status = sync_service.get_sync_status()
        
        logger.info(f"Updated version {version} for model: {model_name}")
        
        return {
            "updated": True,
            "version": result,
            "sync_status": sync_status.to_dict() if sync_status else None
        }
        
    except Exception as e:
        logger.error(f"Failed to update version {version} for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Upload Endpoints
# ==============================================================================

@router.post("/models/upload")
async def upload_pkl_model(
    file: UploadFile = File(...),
    model_name: str = Query(..., description="Name for the registered model"),
    description: Optional[str] = Query(None, description="Model description"),
    authorization: Optional[str] = Header(None),
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Upload a .pkl file and register it as a new model in MLflow.
    
    WRITE-PATH ENFORCEMENT: All model uploads MUST use this endpoint.
    
    This endpoint:
    1. Saves the uploaded .pkl file temporarily
    2. Logs the model using mlflow.pyfunc.log_model
    3. Registers the model version
    4. Cleans up temporary files
    
    Reference: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.log_model
    """
    import mlflow
    import mlflow.pyfunc
    import pickle
    
    # Validate file extension
    if not file.filename.endswith('.pkl'):
        raise HTTPException(
            status_code=400,
            detail="Only .pkl files are supported"
        )
    
    temp_dir = tempfile.mkdtemp()
    pkl_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(pkl_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Load and validate the pickle file
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        
        # Check if it's a valid model (has predict method)
        if not hasattr(model, 'predict'):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file does not contain a valid model (no 'predict' method)"
            )
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_BASE_URL)
        
        # Create or get experiment
        experiment_name = "model_uploads"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Log the model
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=None,
                artifacts={"model": pkl_path},
                registered_model_name=model_name
            )
            
            run_id = run.info.run_id
        
        # Get the created version
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(versions, key=lambda v: int(v.version)) if versions else None
        
        # Update description if provided
        if description and latest_version:
            await sync_service.update_model_version(
                name=model_name,
                version=latest_version.version,
                description=description,
                authorization=authorization
            )
        
        sync_status = sync_service.get_sync_status()
        
        logger.info(f"Uploaded and registered model: {model_name}")
        
        return {
            "uploaded": True,
            "model_name": model_name,
            "run_id": run_id,
            "version": latest_version.version if latest_version else None,
            "sync_status": sync_status.to_dict() if sync_status else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# Health & Diagnostics
# ==============================================================================

@router.get("/health")
async def registry_health(
    sync_service: RegistrySyncService = Depends(get_sync_service)
):
    """
    Check the health of the registry sync service.
    """
    sync_status = sync_service.get_sync_status()
    
    return {
        "healthy": sync_status is not None and sync_status.mlflow_reachable,
        "mlflow_url": MLFLOW_BASE_URL,
        "sync_status": sync_status.to_dict() if sync_status else None,
        "is_stale": sync_service.is_data_stale()
    }
