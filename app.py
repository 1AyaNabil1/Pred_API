import os
import sys
import numpy as np
import random
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import mlflow
import mlflow.pyfunc
from typing import List, Optional, Dict, Any
import httpx

app = FastAPI(title="ML Prediction API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URI = os.getenv("MODEL_URI", "models:/MyModel/Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_BASE_URL = os.getenv("MLFLOW_BASE_URL", "http://localhost:5001")
MLFLOW_AUTH_TOKEN = os.getenv("MLFLOW_AUTH_TOKEN", "")
INFERENCE_LOGGING_SAMPLE_RATE = float(os.getenv("INFERENCE_LOGGING_SAMPLE_RATE", "0.1"))
INFERENCE_EXPERIMENT_NAME = "inference_tracking"

model = None
prediction_count = 0

class PredictionRequest(BaseModel):
    data: List[List[float]] = Field(..., description="Input features as 2D array")
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("data cannot be empty")
        if not all(isinstance(row, list) for row in v):
            raise ValueError("data must be a list of lists")
        return v

class PredictionResponse(BaseModel):
    predictions: List[float]
    probabilities: Optional[List[List[float]]] = None

# MLflow Model Registry request/response models
class SearchModelsRequest(BaseModel):
    filter: Optional[str] = None
    max_results: Optional[int] = 100

class CreateModelVersionRequest(BaseModel):
    name: str
    source: str
    run_id: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list] = None

class GetModelVersionRequest(BaseModel):
    name: str
    version: str

class TransitionStageRequest(BaseModel):
    name: str
    version: str
    stage: str
    archive_existing_versions: Optional[bool] = False

class DeleteModelRequest(BaseModel):
    name: str

class UpdateModelRequest(BaseModel):
    name: str
    description: Optional[str] = None

class InvokeModelRequest(BaseModel):
    model_url: str
    inputs: Dict[str, Any]
    auth_token: Optional[str] = None

def get_headers(authorization: Optional[str] = None) -> Dict[str, str]:
    """Build headers for MLflow API requests."""
    headers = {"Content-Type": "application/json"}
    
    # Use provided auth token, fallback to env, or none
    token = authorization or MLFLOW_AUTH_TOKEN
    if token:
        if token.startswith("Bearer "):
            headers["Authorization"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"
    
    return headers

@app.on_event("startup")
async def load_model():
    global model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        experiment = mlflow.get_experiment_by_name(INFERENCE_EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(INFERENCE_EXPERIMENT_NAME)
            print(f"Created MLflow experiment: {INFERENCE_EXPERIMENT_NAME}")
        
        try:
            model = mlflow.pyfunc.load_model(MODEL_URI)
            print(f"Model loaded from: {MODEL_URI}")
            print(f"Inference logging enabled with {INFERENCE_LOGGING_SAMPLE_RATE*100}% sampling rate")
        except Exception as model_error:
            print(f"No model found at {MODEL_URI}: {model_error}")
            print("API started WITHOUT a model. Please register a model in MLflow UI:")
            print("    1. Go to http://localhost:5001")
            print("    2. Train and log a model")
            print("    3. Register it in the Model Registry")
            print("    4. Set MODEL_URI environment variable or register as 'MyModel/Production'")
            print("    5. Restart the API")
            model = None
    except Exception as e:
        print(f"Critical error during startup: {e}", file=sys.stderr)
        sys.exit(1)

@app.get("/")
async def root():
    return {
        "message": "ML Prediction API",
        "model_uri": MODEL_URI,
        "endpoints": {
            "predict": "/predict [POST]",
            "health": "/health [GET]"
        }
    }

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return {"status": "healthy", "model_uri": MODEL_URI}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global prediction_count
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please register a model in MLflow Registry first."
        )
    
    try:
        input_data = np.array(request.data)
        
        predictions = model.predict(input_data)
        predictions_list = predictions.tolist()
        
        probabilities_list = None
        try:
            if hasattr(model._model_impl.python_model, 'predict_proba'):
                probabilities = model._model_impl.python_model.predict_proba(input_data)
                probabilities_list = probabilities.tolist()
        except Exception:
            pass
        
        prediction_count += 1
        
        if random.random() < INFERENCE_LOGGING_SAMPLE_RATE:
            try:
                mlflow.set_experiment(INFERENCE_EXPERIMENT_NAME)
                with mlflow.start_run(run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_param("num_samples", len(request.data))
                    mlflow.log_param("model_uri", MODEL_URI)
                    mlflow.log_param("timestamp", datetime.now().isoformat())
                    mlflow.log_metric("prediction_count", prediction_count)
                    
                    if len(request.data) > 0:
                        mlflow.log_param("sample_input", str(request.data[0]))
                    
                    for i, pred in enumerate(predictions_list):
                        mlflow.log_metric(f"prediction_{i}", float(pred))
                        if probabilities_list and i < len(probabilities_list):
                            for class_idx, prob in enumerate(probabilities_list[i]):
                                mlflow.log_metric(f"prob_class_{class_idx}_sample_{i}", float(prob))

            except Exception as log_error:
                print(f"Warning: Failed to log inference to MLflow: {log_error}", file=sys.stderr)
        
        return PredictionResponse(
            predictions=predictions_list,
            probabilities=probabilities_list
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input shape or data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

# MLflow Model Registry Endpoints
@app.post("/mlflow/registered-models/search")
async def search_registered_models(
    request: SearchModelsRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Search for registered models.
    Forwards to GET /api/2.0/mlflow/registered-models/search
    """
    url = f"{MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/search"
    headers = get_headers(authorization)
    
    params = {}
    if request.filter:
        params["filter"] = request.filter
    if request.max_results:
        params["max_results"] = request.max_results
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

@app.post("/mlflow/model-versions/create")
async def create_model_version(
    request: CreateModelVersionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Create/register a new model version.
    Forwards to POST /api/2.0/mlflow/model-versions/create
    """
    url = f"{MLFLOW_BASE_URL}/api/2.0/mlflow/model-versions/create"
    headers = get_headers(authorization)
    
    payload = {
        "name": request.name,
        "source": request.source,
    }
    
    if request.run_id:
        payload["run_id"] = request.run_id
    if request.description:
        payload["description"] = request.description
    if request.tags:
        payload["tags"] = request.tags
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

@app.get("/mlflow/model-versions/get")
async def get_model_version(
    name: str,
    version: str,
    authorization: Optional[str] = Header(None)
):
    """
    Get model version metadata including signature and input_example.
    Forwards to GET /api/2.0/mlflow/model-versions/get
    """
    url = f"{MLFLOW_BASE_URL}/api/2.0/mlflow/model-versions/get"
    headers = get_headers(authorization)
    
    params = {"name": name, "version": version}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

@app.post("/mlflow/model-versions/transition-stage")
async def transition_model_stage(
    request: TransitionStageRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Transition model version to a different stage (Production/Staging/Archived).
    Forwards to POST /api/2.0/mlflow/model-versions/transition-stage
    """
    url = f"{MLFLOW_BASE_URL}/api/2.0/mlflow/model-versions/transition-stage"
    headers = get_headers(authorization)
    
    payload = {
        "name": request.name,
        "version": request.version,
        "stage": request.stage,
        "archive_existing_versions": request.archive_existing_versions
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

@app.post("/mlflow/registered-models/delete")
async def delete_registered_model(
    request: DeleteModelRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Delete a registered model.
    Forwards to POST /api/2.0/mlflow/registered-models/delete
    """
    url = f"{MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/delete"
    headers = get_headers(authorization)
    
    payload = {"name": request.name}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

@app.post("/mlflow/registered-models/update")
async def update_registered_model(
    request: UpdateModelRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Update registered model metadata.
    Forwards to POST /api/2.0/mlflow/registered-models/update
    """
    url = f"{MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/update"
    headers = get_headers(authorization)
    
    payload = {"name": request.name}
    if request.description is not None:
        payload["description"] = request.description
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"MLflow API error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

@app.post("/serve/invoke")
async def invoke_model(request: InvokeModelRequest):
    """
    Invoke model serving endpoint and return response with latency.
    Forwards to model serving endpoint (e.g., /invocations or /predict).
    """
    if not request.model_url:
        raise HTTPException(status_code=400, detail="model_url is required")
    
    headers = {"Content-Type": "application/json"}
    if request.auth_token:
        if request.auth_token.startswith("Bearer "):
            headers["Authorization"] = request.auth_token
        else:
            headers["Authorization"] = f"Bearer {request.auth_token}"
    
    payload = {"inputs": request.inputs}
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                request.model_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "predictions": response.json(),
                "latency_ms": round(latency_ms, 2),
                "status_code": response.status_code
            }
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Model serving error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invocation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)