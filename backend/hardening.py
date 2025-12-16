"""
MLflow Model Serving Hardening Extensions

This module provides additional endpoints and utilities for:
1. Model schema verification (checking signature is stored)
2. Model serving health checks
3. Stage change notifications for serving restart
4. Enhanced observability logging

Reference: https://mlflow.org/docs/latest/models.html#model-signature-and-input-example
"""

import asyncio
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from pydantic import BaseModel, Field
import httpx
import mlflow
from mlflow.tracking import MlflowClient

from config import settings
from logger import setup_logging, set_request_id, get_request_id
from auth import verify_api_key

logger = setup_logging(settings.LOG_LEVEL)

router = APIRouter(tags=["Model Serving Hardening"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SchemaVerificationResponse(BaseModel):
    """Response for schema verification endpoint."""
    model_name: str
    version: str
    has_signature: bool
    signature: Optional[Dict[str, Any]] = None
    has_input_example: bool
    input_example: Optional[Any] = None
    schema_source: str  # "mlflow_signature", "manual", "none"
    warnings: List[str] = []


class ServingHealthResponse(BaseModel):
    """Response for serving health check."""
    serving_url: str
    is_healthy: bool
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None


class StageChangeNotification(BaseModel):
    """Notification payload for stage changes."""
    model_name: str
    version: str
    old_stage: str
    new_stage: str
    requires_serving_restart: bool
    restart_command: str


# =============================================================================
# Schema Verification Endpoint
# Reference: https://mlflow.org/docs/latest/models.html#model-signature-and-input-example
# =============================================================================

@router.get("/models/{model_name}/versions/{version}/schema")
async def verify_model_schema(
    request: Request,
    model_name: str,
    version: str,
    api_key: Optional[str] = Depends(verify_api_key)
) -> SchemaVerificationResponse:
    """
    Verify that a model has a proper signature stored in MLflow registry.
    
    Reference: https://mlflow.org/docs/latest/models.html#model-signature-and-input-example
    
    This endpoint checks:
    1. Whether the model version has a signature
    2. Whether it has an input_example
    3. The schema source (MLflow signature vs manual)
    
    Why this matters:
    - MLflow does NOT infer schema retroactively
    - If signature is missing at logging time, UI auto-schema must be disabled
    - Manual schema fallback is required for legacy models
    """
    request_id = set_request_id()
    logger.info(f"Verifying schema for '{model_name}' v{version} (request_id: {request_id})")
    
    warnings = []
    
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_BASE_URL)
        client = MlflowClient()
        
        # Get model version metadata
        model_version = client.get_model_version(model_name, version)
        
        # Get run data if available
        run_id = model_version.run_id
        signature = None
        input_example = None
        has_signature = False
        has_input_example = False
        
        if run_id:
            try:
                # Try to load the model to check signature
                model_uri = f"models:/{model_name}/{version}"
                model_info = mlflow.models.get_model_info(model_uri)
                
                if model_info.signature:
                    has_signature = True
                    signature = {
                        "inputs": str(model_info.signature.inputs) if model_info.signature.inputs else None,
                        "outputs": str(model_info.signature.outputs) if model_info.signature.outputs else None,
                        "params": str(model_info.signature.params) if hasattr(model_info.signature, 'params') and model_info.signature.params else None
                    }
                else:
                    warnings.append("Model has no signature. UI auto-schema disabled.")
                
                # Check for input example
                if hasattr(model_info, 'saved_input_example_info') and model_info.saved_input_example_info:
                    has_input_example = True
                    # Try to load the input example
                    try:
                        run = client.get_run(run_id)
                        artifact_uri = run.info.artifact_uri
                        # Input example is typically at input_example.json
                        input_example = {"stored": True, "artifact_path": f"{artifact_uri}/input_example.json"}
                    except Exception:
                        input_example = {"stored": True, "path": "unable to retrieve"}
                else:
                    warnings.append("No input_example stored. Manual testing required.")
                    
            except Exception as e:
                logger.warning(f"Could not load model info: {e}")
                warnings.append(f"Could not verify signature: {str(e)}")
        else:
            warnings.append("Model has no associated run_id. Cannot verify signature.")
        
        # Determine schema source
        if has_signature:
            schema_source = "mlflow_signature"
        else:
            schema_source = "none"
        
        return SchemaVerificationResponse(
            model_name=model_name,
            version=version,
            has_signature=has_signature,
            signature=signature,
            has_input_example=has_input_example,
            input_example=input_example,
            schema_source=schema_source,
            warnings=warnings
        )
        
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow error: {e}")
        raise HTTPException(status_code=404, detail=f"Model version not found: {model_name} v{version}")
    except Exception as e:
        logger.error(f"Schema verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Schema verification failed: {str(e)}")


# =============================================================================
# Model Serving Health Check
# Reference: https://mlflow.org/docs/latest/models.html#local-rest-server
# =============================================================================

@router.get("/serving/health")
async def check_serving_health(
    request: Request,
    serving_url: Optional[str] = None,
    api_key: Optional[str] = Depends(verify_api_key)
) -> ServingHealthResponse:
    """
    Check if MLflow model serving is running and healthy.
    
    Reference: https://mlflow.org/docs/latest/models.html#local-rest-server
    
    This endpoint verifies:
    1. The serving process is running
    2. The /invocations endpoint is accessible
    3. Response latency
    
    Why this matters:
    - MLflow does NOT auto-start serving when models are registered
    - Serving is a separate runtime that must be explicitly started
    - This check confirms the serving process is live
    """
    request_id = set_request_id()
    url = serving_url or settings.MODEL_SERVING_URL
    
    logger.info(f"Checking serving health at {url} (request_id: {request_id})")
    
    import time
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=10.0) as client:
            # First try /health endpoint (if available)
            health_url = f"{url}/health"
            try:
                response = await client.get(health_url)
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    return ServingHealthResponse(
                        serving_url=url,
                        is_healthy=True,
                        status_code=response.status_code,
                        latency_ms=round(latency_ms, 2),
                        model_info={"health_endpoint": "available"}
                    )
            except Exception:
                pass
            
            # Fall back to checking /invocations with OPTIONS
            invocations_url = f"{url}/invocations"
            try:
                response = await client.options(invocations_url)
                latency_ms = (time.time() - start_time) * 1000
                
                # 200, 405 (Method Not Allowed for OPTIONS), or 400 all indicate serving is up
                if response.status_code in [200, 400, 405]:
                    return ServingHealthResponse(
                        serving_url=url,
                        is_healthy=True,
                        status_code=response.status_code,
                        latency_ms=round(latency_ms, 2),
                        model_info={"invocations_endpoint": "available"}
                    )
                else:
                    return ServingHealthResponse(
                        serving_url=url,
                        is_healthy=False,
                        status_code=response.status_code,
                        latency_ms=round(latency_ms, 2),
                        error=f"Unexpected status: {response.status_code}"
                    )
            except Exception as e:
                return ServingHealthResponse(
                    serving_url=url,
                    is_healthy=False,
                    error=f"Connection failed: {str(e)}"
                )
                
    except httpx.ConnectError:
        return ServingHealthResponse(
            serving_url=url,
            is_healthy=False,
            error="Connection refused. Model serving is not running."
        )
    except Exception as e:
        return ServingHealthResponse(
            serving_url=url,
            is_healthy=False,
            error=f"Health check failed: {str(e)}"
        )


# =============================================================================
# Stage Change with Serving Restart Notification
# Reference: https://mlflow.org/docs/latest/models.html#model-uri
# =============================================================================

@router.post("/models/{model_name}/versions/{version}/transition-stage")
async def transition_stage_with_notification(
    request: Request,
    model_name: str,
    version: str,
    new_stage: str,
    background_tasks: BackgroundTasks,
    archive_existing: bool = False,
    api_key: Optional[str] = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Transition model stage with serving restart notification.
    
    Reference: https://mlflow.org/docs/latest/models.html#model-uri
    
    CRITICAL: MLflow does NOT automatically reload a served model when stage changes.
    
    If you are serving:
        models:/MyModel/Production
    
    And you transition a new version to Production:
        - The old version KEEPS running
        - You MUST restart the serving process
    
    This endpoint:
    1. Performs the stage transition
    2. Returns a notification with restart instructions
    3. Optionally triggers a background task for logging
    """
    request_id = set_request_id()
    logger.info(f"Stage transition with notification: {model_name} v{version} -> {new_stage}")
    
    # Validate stage
    valid_stages = ['None', 'Staging', 'Production', 'Archived']
    if new_stage not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage. Must be one of: {', '.join(valid_stages)}"
        )
    
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_BASE_URL)
        client = MlflowClient()
        
        # Get current stage
        model_version = client.get_model_version(model_name, version)
        old_stage = model_version.current_stage
        
        # Perform transition
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=new_stage,
            archive_existing_versions=archive_existing
        )
        
        # Determine if serving restart is needed
        requires_restart = (
            old_stage in ['Production', 'Staging'] or 
            new_stage in ['Production', 'Staging']
        )
        
        # Generate restart command
        if new_stage == 'Production':
            restart_command = (
                f"# If serving models:/{model_name}/Production, restart with:\n"
                f"docker-compose restart model-serving\n"
                f"# Or for local:\n"
                f"pkill -f 'mlflow models serve' && "
                f"mlflow models serve --model-uri 'models:/{model_name}/Production' "
                f"--port 5002 --no-conda &"
            )
        else:
            restart_command = "# No restart needed if not serving this stage"
        
        notification = StageChangeNotification(
            model_name=model_name,
            version=version,
            old_stage=old_stage,
            new_stage=new_stage,
            requires_serving_restart=requires_restart,
            restart_command=restart_command
        )
        
        # Log the stage change
        background_tasks.add_task(
            log_stage_change,
            request_id,
            model_name,
            version,
            old_stage,
            new_stage
        )
        
        logger.info(f"Stage transition complete: {old_stage} -> {new_stage} (request_id: {request_id})")
        
        return {
            "success": True,
            "model_version": {
                "name": model_name,
                "version": version,
                "current_stage": new_stage,
                "previous_stage": old_stage
            },
            "notification": notification.model_dump(),
            "warning": (
                "If you are serving this model, you MUST restart the serving process "
                "for the new version to take effect. MLflow does NOT auto-reload models."
            ) if requires_restart else None
        }
        
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow transition failed: {e}")
        raise HTTPException(status_code=400, detail=f"Stage transition failed: {str(e)}")
    except Exception as e:
        logger.error(f"Transition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transition failed: {str(e)}")


async def log_stage_change(
    request_id: str,
    model_name: str,
    version: str,
    old_stage: str,
    new_stage: str
):
    """Background task to log stage changes for observability."""
    logger.info(
        f"STAGE_CHANGE | request_id={request_id} | "
        f"model={model_name} | version={version} | "
        f"from={old_stage} | to={new_stage}"
    )


# =============================================================================
# Enhanced Invocation with Full Observability
# Reference: https://mlflow.org/docs/latest/concepts.html#mlflow-components
# =============================================================================

@router.post("/serve/invoke/observed")
async def invoke_with_full_observability(
    request: Request,
    model_url: str,
    inputs: Dict[str, Any],
    auth_token: Optional[str] = None,
    api_key: Optional[str] = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Invoke model with full three-layer observability logging.
    
    This endpoint logs:
    1. UI/Client request ID
    2. Backend request ID
    3. MLflow response/error body
    
    Reference: https://mlflow.org/docs/latest/concepts.html#mlflow-components
    """
    import time
    import uuid
    
    # Generate/capture request IDs
    request_id = set_request_id()
    client_request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    logger.info(
        f"INVOKE_START | client_request_id={client_request_id} | "
        f"backend_request_id={request_id} | model_url={model_url}"
    )
    
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}" if not auth_token.startswith("Bearer ") else auth_token
    
    # Format payload using MLflow-compatible format
    if isinstance(inputs, dict):
        if any(k in inputs for k in ['inputs', 'dataframe_split', 'instances', 'dataframe_records']):
            payload = inputs
        else:
            payload = {"dataframe_records": [inputs]}
    elif isinstance(inputs, list):
        payload = {"instances": inputs}
    else:
        payload = {"inputs": inputs}
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(model_url, json=payload, headers=headers)
            latency_ms = (time.time() - start_time) * 1000
            
            # Log response
            logger.info(
                f"INVOKE_RESPONSE | client_request_id={client_request_id} | "
                f"backend_request_id={request_id} | status={response.status_code} | "
                f"latency_ms={latency_ms:.2f}"
            )
            
            if response.status_code != 200:
                logger.error(
                    f"INVOKE_ERROR | client_request_id={client_request_id} | "
                    f"mlflow_response={response.text[:500]}"
                )
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Model serving error: {response.text[:200]}"
                )
            
            result = response.json()
            
            return {
                "predictions": result,
                "observability": {
                    "client_request_id": client_request_id,
                    "backend_request_id": request_id,
                    "latency_ms": round(latency_ms, 2),
                    "status_code": response.status_code,
                    "payload_format": list(payload.keys())[0]
                }
            }
            
    except httpx.ConnectError as e:
        logger.error(
            f"INVOKE_FAILED | client_request_id={client_request_id} | "
            f"backend_request_id={request_id} | error=connection_refused"
        )
        raise HTTPException(
            status_code=503,
            detail="Model serving not available. Ensure serving is running."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"INVOKE_FAILED | client_request_id={client_request_id} | "
            f"backend_request_id={request_id} | error={str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Invocation failed: {str(e)}")


# =============================================================================
# Serving Process Status
# =============================================================================

@router.get("/serving/status")
async def get_serving_status(
    request: Request,
    api_key: Optional[str] = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Get comprehensive serving status including configuration.
    """
    request_id = set_request_id()
    
    # Check health
    health = await check_serving_health(request, None, api_key)
    
    return {
        "configured_url": settings.MODEL_SERVING_URL,
        "enabled": settings.MODEL_SERVING_ENABLED,
        "health": health.model_dump(),
        "restart_instructions": {
            "docker": "docker-compose restart model-serving",
            "local": "pkill -f 'mlflow models serve' && mlflow models serve --model-uri 'models:/ModelName/Production' --port 5002 --no-conda &"
        },
        "mlflow_docs": {
            "serving": "https://mlflow.org/docs/latest/deployment/deploy-model-locally.html",
            "model_uri": "https://mlflow.org/docs/latest/model-registry.html#loading-registered-models"
        }
    }
