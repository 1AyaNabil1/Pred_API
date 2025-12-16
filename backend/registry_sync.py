"""
MLflow Registry Sync Service
============================

This module implements a Registry Sync Service that ensures strong consistency
between MLflow Model Registry and the Pixonal platform.

ARCHITECTURAL PRINCIPLES (Non-Negotiable):
1. MLflow is the SINGLE SOURCE OF TRUTH
2. The website NEVER stores model state independently
3. All write operations go through the backend to MLflow
4. The website reflects MLflow state via ACTIVE synchronization
5. Eventual consistency is acceptable; silent divergence is NOT

MLFLOW LIMITATIONS (Documented):
- MLflow does NOT push updates (no webhooks, no event streams)
- MLflow does NOT notify on changes
- MLflow does NOT auto-reload serving on stage change
- All sync must be PULL-BASED via REST API polling

References:
- MLflow REST API: https://mlflow.org/docs/latest/rest-api.html
- Model Registry: https://mlflow.org/docs/latest/model-registry.html
- Concepts: https://mlflow.org/docs/latest/concepts.html

Author: Pixonal Platform Team
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    """Sync operation status codes."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    STALE = "stale"


@dataclass
class SyncMetadata:
    """
    Metadata about the last synchronization with MLflow.
    
    This provides transparency about sync state to the frontend,
    enabling users to understand data freshness.
    """
    last_sync_timestamp: datetime
    last_sync_status: SyncStatus
    total_models: int
    total_versions: int
    sync_duration_ms: float
    mlflow_reachable: bool
    error_message: Optional[str] = None
    # Hash of the model registry state for change detection
    state_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_sync_timestamp": self.last_sync_timestamp.isoformat(),
            "last_sync_status": self.last_sync_status.value,
            "total_models": self.total_models,
            "total_versions": self.total_versions,
            "sync_duration_ms": round(self.sync_duration_ms, 2),
            "mlflow_reachable": self.mlflow_reachable,
            "error_message": self.error_message,
            "state_hash": self.state_hash,
            "age_seconds": (datetime.now(timezone.utc) - self.last_sync_timestamp).total_seconds()
        }


@dataclass
class ModelVersion:
    """Represents a model version from MLflow registry."""
    name: str
    version: str
    creation_timestamp: int
    last_updated_timestamp: int
    current_stage: str
    description: Optional[str] = None
    source: Optional[str] = None
    run_id: Optional[str] = None
    status: Optional[str] = None
    status_message: Optional[str] = None
    tags: List[Dict[str, str]] = field(default_factory=list)
    run_link: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    
    @classmethod
    def from_mlflow_response(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Parse a model version from MLflow API response."""
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            creation_timestamp=int(data.get("creation_timestamp", 0)),
            last_updated_timestamp=int(data.get("last_updated_timestamp", 0)),
            current_stage=data.get("current_stage", "None"),
            description=data.get("description"),
            source=data.get("source"),
            run_id=data.get("run_id"),
            status=data.get("status"),
            status_message=data.get("status_message"),
            tags=data.get("tags", []),
            run_link=data.get("run_link"),
            aliases=data.get("aliases", [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "creation_timestamp": self.creation_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
            "current_stage": self.current_stage,
            "description": self.description,
            "source": self.source,
            "run_id": self.run_id,
            "status": self.status,
            "status_message": self.status_message,
            "tags": self.tags,
            "run_link": self.run_link,
            "aliases": self.aliases
        }


@dataclass
class RegisteredModel:
    """Represents a registered model from MLflow registry."""
    name: str
    creation_timestamp: int
    last_updated_timestamp: int
    description: Optional[str] = None
    latest_versions: List[ModelVersion] = field(default_factory=list)
    tags: List[Dict[str, str]] = field(default_factory=list)
    aliases: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_mlflow_response(cls, data: Dict[str, Any]) -> "RegisteredModel":
        """Parse a registered model from MLflow API response."""
        latest_versions = [
            ModelVersion.from_mlflow_response(v) 
            for v in data.get("latest_versions", [])
        ]
        return cls(
            name=data.get("name", ""),
            creation_timestamp=int(data.get("creation_timestamp", 0)),
            last_updated_timestamp=int(data.get("last_updated_timestamp", 0)),
            description=data.get("description"),
            latest_versions=latest_versions,
            tags=data.get("tags", []),
            aliases=data.get("aliases", {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "creation_timestamp": self.creation_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
            "description": self.description,
            "latest_versions": [v.to_dict() for v in self.latest_versions],
            "tags": self.tags,
            "aliases": self.aliases
        }


class RegistrySyncService:
    """
    Registry Sync Service for MLflow Model Registry.
    
    This service is the ONLY authorized source for model registry data.
    It ensures that all model state comes directly from MLflow via
    the official REST API.
    
    CRITICAL CONSTRAINTS:
    - NO permanent caching of models
    - NO database storage of model state
    - ALWAYS fetch fresh data from MLflow
    - TREAT MLflow responses as AUTHORITATIVE
    
    MLflow REST API Reference:
    - POST /api/2.0/mlflow/registered-models/search
    - POST /api/2.0/mlflow/model-versions/search
    - POST /api/2.0/mlflow/registered-models/create
    - POST /api/2.0/mlflow/registered-models/delete
    - POST /api/2.0/mlflow/model-versions/transition-stage
    
    See: https://mlflow.org/docs/latest/rest-api.html
    """
    
    # Short-lived cache TTL (seconds) - only for rate limiting protection
    # This is NOT a permanent cache - it expires quickly
    CACHE_TTL_SECONDS = 5
    
    # Maximum staleness before warning (seconds)
    STALE_THRESHOLD_SECONDS = 30
    
    def __init__(
        self,
        mlflow_base_url: str,
        timeout: float = 30.0,
        auth_token: Optional[str] = None
    ):
        """
        Initialize the Registry Sync Service.
        
        Args:
            mlflow_base_url: Base URL of MLflow tracking server
            timeout: Request timeout in seconds
            auth_token: Optional authentication token
        """
        self.mlflow_base_url = mlflow_base_url.rstrip("/")
        self.timeout = timeout
        self.auth_token = auth_token
        
        # Short-lived cache to prevent hammering MLflow on rapid requests
        # This is NOT permanent storage - cleared every CACHE_TTL_SECONDS
        self._cache: Optional[Tuple[float, List[RegisteredModel]]] = None
        self._last_sync: Optional[SyncMetadata] = None
        
        logger.info(
            f"RegistrySyncService initialized with MLflow URL: {self.mlflow_base_url}"
        )
    
    def _get_headers(self, authorization: Optional[str] = None) -> Dict[str, str]:
        """Build headers for MLflow API requests."""
        headers = {"Content-Type": "application/json"}
        
        if authorization:
            headers["Authorization"] = authorization
        elif self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        return headers
    
    def _compute_state_hash(self, models: List[RegisteredModel]) -> str:
        """
        Compute a hash of the current registry state.
        
        This enables efficient change detection without comparing full state.
        """
        # Create a deterministic representation of the state
        state_repr = json.dumps(
            sorted([m.to_dict() for m in models], key=lambda x: x["name"]),
            sort_keys=True
        )
        return hashlib.sha256(state_repr.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self) -> bool:
        """Check if the short-lived cache is still valid."""
        if self._cache is None:
            return False
        cache_time, _ = self._cache
        return (time.time() - cache_time) < self.CACHE_TTL_SECONDS
    
    async def fetch_registered_models(
        self,
        authorization: Optional[str] = None,
        filter_string: Optional[str] = None,
        max_results: int = 1000,
        force_refresh: bool = False
    ) -> Tuple[List[RegisteredModel], SyncMetadata]:
        """
        Fetch all registered models from MLflow.
        
        This is the PRIMARY method for retrieving model state.
        It ALWAYS queries MLflow (unless cache is valid and force_refresh=False).
        
        MLflow API: POST /api/2.0/mlflow/registered-models/search
        Reference: https://mlflow.org/docs/latest/rest-api.html#search-registered-models
        
        Args:
            authorization: Optional auth header
            filter_string: Optional filter expression
            max_results: Maximum number of models to return
            force_refresh: If True, bypass short-lived cache
            
        Returns:
            Tuple of (models, sync_metadata)
        """
        start_time = time.time()
        
        # Check short-lived cache (rate limiting protection only)
        if not force_refresh and self._is_cache_valid():
            _, cached_models = self._cache
            logger.debug("Returning cached models (cache still valid)")
            return cached_models, self._last_sync
        
        url = f"{self.mlflow_base_url}/api/2.0/mlflow/registered-models/search"
        headers = self._get_headers(authorization)
        
        all_models: List[RegisteredModel] = []
        page_token: Optional[str] = None
        error_message: Optional[str] = None
        mlflow_reachable = True
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                while True:
                    # Build request payload
                    payload: Dict[str, Any] = {"max_results": min(max_results, 1000)}
                    if filter_string:
                        payload["filter"] = filter_string
                    if page_token:
                        payload["page_token"] = page_token
                    
                    logger.debug(f"Fetching models from MLflow: {url}")
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Parse registered models from response
                    for model_data in data.get("registered_models", []):
                        model = RegisteredModel.from_mlflow_response(model_data)
                        all_models.append(model)
                    
                    # Check for pagination
                    page_token = data.get("next_page_token")
                    if not page_token:
                        break
                    
                    # Safety check to prevent infinite loops
                    if len(all_models) >= max_results:
                        break
            
            # Compute state hash for change detection
            state_hash = self._compute_state_hash(all_models)
            
            # Count total versions
            total_versions = sum(len(m.latest_versions) for m in all_models)
            
            # Update short-lived cache
            self._cache = (time.time(), all_models)
            
            sync_duration_ms = (time.time() - start_time) * 1000
            
            self._last_sync = SyncMetadata(
                last_sync_timestamp=datetime.now(timezone.utc),
                last_sync_status=SyncStatus.SUCCESS,
                total_models=len(all_models),
                total_versions=total_versions,
                sync_duration_ms=sync_duration_ms,
                mlflow_reachable=True,
                state_hash=state_hash
            )
            
            logger.info(
                f"Successfully synced {len(all_models)} models, "
                f"{total_versions} versions in {sync_duration_ms:.2f}ms"
            )
            
        except httpx.HTTPStatusError as e:
            mlflow_reachable = True  # Reachable but returned error
            error_message = f"MLflow API error: {e.response.status_code}"
            logger.error(f"MLflow API error: {e.response.status_code} - {e.response.text}")
            
            self._last_sync = SyncMetadata(
                last_sync_timestamp=datetime.now(timezone.utc),
                last_sync_status=SyncStatus.FAILED,
                total_models=0,
                total_versions=0,
                sync_duration_ms=(time.time() - start_time) * 1000,
                mlflow_reachable=mlflow_reachable,
                error_message=error_message
            )
            raise
            
        except httpx.RequestError as e:
            mlflow_reachable = False
            error_message = f"Cannot reach MLflow: {str(e)}"
            logger.error(f"Cannot reach MLflow: {str(e)}")
            
            self._last_sync = SyncMetadata(
                last_sync_timestamp=datetime.now(timezone.utc),
                last_sync_status=SyncStatus.FAILED,
                total_models=0,
                total_versions=0,
                sync_duration_ms=(time.time() - start_time) * 1000,
                mlflow_reachable=mlflow_reachable,
                error_message=error_message
            )
            raise
        
        return all_models, self._last_sync
    
    async def fetch_model_versions(
        self,
        model_name: str,
        authorization: Optional[str] = None,
        max_results: int = 1000
    ) -> List[ModelVersion]:
        """
        Fetch all versions for a specific model.
        
        MLflow API: POST /api/2.0/mlflow/model-versions/search
        Reference: https://mlflow.org/docs/latest/rest-api.html#search-model-versions
        
        Args:
            model_name: Name of the registered model
            authorization: Optional auth header
            max_results: Maximum number of versions to return
            
        Returns:
            List of model versions
        """
        url = f"{self.mlflow_base_url}/api/2.0/mlflow/model-versions/search"
        headers = self._get_headers(authorization)
        
        all_versions: List[ModelVersion] = []
        page_token: Optional[str] = None
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while True:
                payload: Dict[str, Any] = {
                    "filter": f"name='{model_name}'",
                    "max_results": min(max_results, 1000)
                }
                if page_token:
                    payload["page_token"] = page_token
                
                logger.debug(f"Fetching versions for model '{model_name}'")
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                for version_data in data.get("model_versions", []):
                    version = ModelVersion.from_mlflow_response(version_data)
                    all_versions.append(version)
                
                page_token = data.get("next_page_token")
                if not page_token:
                    break
                
                if len(all_versions) >= max_results:
                    break
        
        logger.info(f"Fetched {len(all_versions)} versions for model '{model_name}'")
        return all_versions
    
    async def get_model_version_details(
        self,
        model_name: str,
        version: str,
        authorization: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed metadata for a specific model version.
        
        MLflow API: GET /api/2.0/mlflow/model-versions/get
        Reference: https://mlflow.org/docs/latest/rest-api.html#get-model-version
        
        Args:
            model_name: Name of the registered model
            version: Version number
            authorization: Optional auth header
            
        Returns:
            Model version details including signature and artifacts
        """
        url = f"{self.mlflow_base_url}/api/2.0/mlflow/model-versions/get"
        headers = self._get_headers(authorization)
        params = {"name": model_name, "version": version}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
    
    async def create_registered_model(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[Dict[str, str]]] = None,
        authorization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new registered model in MLflow.
        
        MLflow API: POST /api/2.0/mlflow/registered-models/create
        Reference: https://mlflow.org/docs/latest/rest-api.html#create-registered-model
        
        Args:
            name: Unique model name
            description: Optional description
            tags: Optional list of tags
            authorization: Optional auth header
            
        Returns:
            Created model data from MLflow
        """
        url = f"{self.mlflow_base_url}/api/2.0/mlflow/registered-models/create"
        headers = self._get_headers(authorization)
        
        payload: Dict[str, Any] = {"name": name}
        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            # Invalidate cache after write operation
            self._cache = None
            
            logger.info(f"Created registered model: {name}")
            return response.json()
    
    async def delete_registered_model(
        self,
        name: str,
        authorization: Optional[str] = None
    ) -> None:
        """
        Delete a registered model from MLflow.
        
        MLflow API: DELETE /api/2.0/mlflow/registered-models/delete
        Reference: https://mlflow.org/docs/latest/rest-api.html#delete-registered-model
        
        Args:
            name: Model name to delete
            authorization: Optional auth header
        """
        url = f"{self.mlflow_base_url}/api/2.0/mlflow/registered-models/delete"
        headers = self._get_headers(authorization)
        payload = {"name": name}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url, params=payload, headers=headers)
            
            # MLflow may not support DELETE method, try POST
            if response.status_code == 405:
                response = await client.post(url, json=payload, headers=headers)
            
            response.raise_for_status()
            
            # Invalidate cache after write operation
            self._cache = None
            
            logger.info(f"Deleted registered model: {name}")
    
    async def create_model_version(
        self,
        name: str,
        source: str,
        run_id: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[Dict[str, str]]] = None,
        authorization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new model version in MLflow.
        
        MLflow API: POST /api/2.0/mlflow/model-versions/create
        Reference: https://mlflow.org/docs/latest/rest-api.html#create-model-version
        
        Args:
            name: Registered model name
            source: URI of the model artifact
            run_id: Optional run ID
            description: Optional description
            tags: Optional list of tags
            authorization: Optional auth header
            
        Returns:
            Created version data from MLflow
        """
        url = f"{self.mlflow_base_url}/api/2.0/mlflow/model-versions/create"
        headers = self._get_headers(authorization)
        
        payload: Dict[str, Any] = {"name": name, "source": source}
        if run_id:
            payload["run_id"] = run_id
        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            # Invalidate cache after write operation
            self._cache = None
            
            logger.info(f"Created model version for: {name}")
            return response.json()
    
    async def transition_model_version_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False,
        authorization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transition a model version to a new stage.
        
        MLflow API: POST /api/2.0/mlflow/model-versions/transition-stage
        Reference: https://mlflow.org/docs/latest/rest-api.html#transition-model-version-stage
        
        Args:
            name: Registered model name
            version: Version number
            stage: Target stage (None, Staging, Production, Archived)
            archive_existing_versions: Whether to archive existing versions in target stage
            authorization: Optional auth header
            
        Returns:
            Updated version data from MLflow
        """
        url = f"{self.mlflow_base_url}/api/2.0/mlflow/model-versions/transition-stage"
        headers = self._get_headers(authorization)
        
        payload = {
            "name": name,
            "version": version,
            "stage": stage,
            "archive_existing_versions": archive_existing_versions
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            # Invalidate cache after write operation
            self._cache = None
            
            logger.info(f"Transitioned {name} v{version} to stage: {stage}")
            return response.json()
    
    async def update_registered_model(
        self,
        name: str,
        description: Optional[str] = None,
        authorization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update a registered model's metadata.
        
        MLflow API: PATCH /api/2.0/mlflow/registered-models/update
        Reference: https://mlflow.org/docs/latest/rest-api.html#update-registered-model
        
        Args:
            name: Model name
            description: New description
            authorization: Optional auth header
            
        Returns:
            Updated model data from MLflow
        """
        url = f"{self.mlflow_base_url}/api/2.0/mlflow/registered-models/update"
        headers = self._get_headers(authorization)
        
        payload: Dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(url, json=payload, headers=headers)
            response.raise_for_status()
            
            # Invalidate cache after write operation
            self._cache = None
            
            logger.info(f"Updated registered model: {name}")
            return response.json()
    
    async def update_model_version(
        self,
        name: str,
        version: str,
        description: Optional[str] = None,
        authorization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update a model version's metadata.
        
        MLflow API: PATCH /api/2.0/mlflow/model-versions/update
        Reference: https://mlflow.org/docs/latest/rest-api.html#update-model-version
        
        Args:
            name: Model name
            version: Version number
            description: New description
            authorization: Optional auth header
            
        Returns:
            Updated version data from MLflow
        """
        url = f"{self.mlflow_base_url}/api/2.0/mlflow/model-versions/update"
        headers = self._get_headers(authorization)
        
        payload: Dict[str, Any] = {"name": name, "version": version}
        if description is not None:
            payload["description"] = description
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(url, json=payload, headers=headers)
            response.raise_for_status()
            
            # Invalidate cache after write operation
            self._cache = None
            
            logger.info(f"Updated model version: {name} v{version}")
            return response.json()
    
    def get_sync_status(self) -> Optional[SyncMetadata]:
        """
        Get the current sync status.
        
        Returns:
            Last sync metadata if available
        """
        return self._last_sync
    
    def is_data_stale(self) -> bool:
        """
        Check if the current data is considered stale.
        
        Returns:
            True if data is stale or never synced
        """
        if self._last_sync is None:
            return True
        
        age = (datetime.now(timezone.utc) - self._last_sync.last_sync_timestamp).total_seconds()
        return age > self.STALE_THRESHOLD_SECONDS
    
    def invalidate_cache(self) -> None:
        """
        Explicitly invalidate the short-lived cache.
        
        Call this after any write operation to ensure fresh data.
        """
        self._cache = None
        logger.debug("Cache invalidated")


# Singleton instance for the application
_registry_service: Optional[RegistrySyncService] = None


def get_registry_service(
    mlflow_base_url: str,
    timeout: float = 30.0,
    auth_token: Optional[str] = None
) -> RegistrySyncService:
    """
    Get or create the singleton RegistrySyncService instance.
    
    Args:
        mlflow_base_url: MLflow tracking server URL
        timeout: Request timeout
        auth_token: Optional auth token
        
    Returns:
        The singleton RegistrySyncService instance
    """
    global _registry_service
    
    if _registry_service is None:
        _registry_service = RegistrySyncService(
            mlflow_base_url=mlflow_base_url,
            timeout=timeout,
            auth_token=auth_token
        )
    
    return _registry_service


def reset_registry_service() -> None:
    """Reset the singleton instance (mainly for testing)."""
    global _registry_service
    _registry_service = None
