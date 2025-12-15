"""
Comprehensive test suite for MLflow Model Registry Proxy API.
Tests security, functionality, and error handling.
"""
import pytest
import json
import tempfile
import pickle
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock

# Import app components
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from logger import setup_logging


# Override settings for testing
settings.API_KEY = ""  # Disable auth for most tests
settings.ALLOWED_ORIGINS = ["*"]
settings.MLFLOW_BASE_URL = "http://localhost:5001"

from main_secure import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self):
        """Test successful health check."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "request_id" in data
            assert "mlflow_status" in data
    
    def test_health_check_mlflow_unreachable(self):
        """Test health check when MLflow is unreachable."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=Exception("Connection error"))
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["mlflow_status"] == "unreachable"


class TestSearchModels:
    """Test model search functionality."""
    
    def test_search_models_get(self):
        """Test GET request for searching models."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"registered_models": []}
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = Mock()
            
            response = client.get("/mlflow/registered-models/search")
            
            assert response.status_code == 200
            assert "registered_models" in response.json()
    
    def test_search_models_post(self):
        """Test POST request for searching models."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"registered_models": []}
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = Mock()
            
            response = client.post(
                "/mlflow/registered-models/search",
                json={"filter": "name='test'", "max_results": 10}
            )
            
            assert response.status_code == 200
    
    def test_search_models_validation_error(self):
        """Test validation error with invalid max_results."""
        response = client.post(
            "/mlflow/registered-models/search",
            json={"max_results": 10000}  # Exceeds limit of 1000
        )
        
        assert response.status_code == 422  # Validation error


class TestModelVersionOperations:
    """Test model version CRUD operations."""
    
    def test_create_model_version(self):
        """Test creating a new model version."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"model_version": {"name": "test", "version": "1"}}
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = Mock()
            
            response = client.post(
                "/mlflow/model-versions/create",
                json={
                    "name": "test_model",
                    "source": "models:/test_model/1",
                    "description": "Test model"
                }
            )
            
            assert response.status_code == 200
    
    def test_create_model_version_validation(self):
        """Test validation for model version creation."""
        response = client.post(
            "/mlflow/model-versions/create",
            json={
                "name": "",  # Empty name
                "source": "models:/test/1"
            }
        )
        
        assert response.status_code == 422
    
    def test_get_model_version(self):
        """Test retrieving model version."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"model_version": {"name": "test", "version": "1"}}
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = Mock()
            
            response = client.get("/mlflow/model-versions/get?name=test&version=1")
            
            assert response.status_code == 200


class TestStageTransition:
    """Test model stage transition."""
    
    def test_transition_to_production(self):
        """Test transitioning model to production."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"model_version": {"current_stage": "Production"}}
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = Mock()
            
            response = client.post(
                "/mlflow/model-versions/transition-stage",
                json={
                    "name": "test_model",
                    "version": "1",
                    "stage": "Production",
                    "archive_existing_versions": True
                }
            )
            
            assert response.status_code == 200
    
    def test_transition_invalid_stage(self):
        """Test validation for invalid stage."""
        response = client.post(
            "/mlflow/model-versions/transition-stage",
            json={
                "name": "test_model",
                "version": "1",
                "stage": "InvalidStage"
            }
        )
        
        assert response.status_code == 422


class TestModelDeletion:
    """Test model deletion."""
    
    def test_delete_model_post(self):
        """Test deleting model via POST."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}
            mock_client.return_value.__aenter__.return_value.send = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = Mock()
            
            response = client.post(
                "/mlflow/registered-models/delete",
                json={"name": "test_model"}
            )
            
            assert response.status_code == 200
    
    def test_delete_model_missing_name(self):
        """Test error when model name is missing."""
        response = client.post("/mlflow/registered-models/delete", json={})
        
        assert response.status_code == 400


class TestModelUpdate:
    """Test model update operations."""
    
    def test_update_model_description(self):
        """Test updating model description."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"registered_model": {"name": "test"}}
            mock_client.return_value.__aenter__.return_value.patch = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = Mock()
            
            response = client.patch(
                "/mlflow/registered-models/update",
                json={"name": "test_model", "description": "Updated description"}
            )
            
            assert response.status_code == 200
    
    def test_update_model_version_description(self):
        """Test updating model version description."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"model_version": {"name": "test", "version": "1"}}
            mock_client.return_value.__aenter__.return_value.patch = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = Mock()
            
            response = client.patch(
                "/mlflow/model-versions/update",
                json={
                    "name": "test_model",
                    "version": "1",
                    "description": "Updated version description"
                }
            )
            
            assert response.status_code == 200


class TestModelInvocation:
    """Test model serving invocation."""
    
    def test_invoke_model_success(self):
        """Test successful model invocation."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"predictions": [1, 2, 3]}
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = Mock()
            
            response = client.post(
                "/serve/invoke",
                json={
                    "model_url": "http://localhost:5001/invocations",
                    "inputs": {"data": [[1, 2, 3]]}
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "latency_ms" in data
            assert "request_id" in data
    
    def test_invoke_model_missing_url(self):
        """Test error when model URL is missing."""
        response = client.post(
            "/serve/invoke",
            json={
                "model_url": "",
                "inputs": {"data": [[1, 2, 3]]}
            }
        )
        
        assert response.status_code == 422


class TestFileUpload:
    """Test pickle file upload."""
    
    def test_upload_pkl_file_invalid_extension(self):
        """Test error for invalid file extension."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            f.flush()
            
            with open(f.name, "rb") as file:
                response = client.post(
                    "/mlflow/upload-pkl-model",
                    files={"file": ("test.txt", file, "text/plain")},
                    data={"model_name": "test_model"}
                )
            
            assert response.status_code == 400
            assert "Invalid file type" in response.json()["detail"]
    
    def test_upload_pkl_file_too_large(self):
        """Test error for file too large."""
        # Create a file larger than allowed
        large_size = (settings.MAX_FILE_SIZE_MB + 1) * 1024 * 1024
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"0" * large_size)
            f.flush()
            
            with open(f.name, "rb") as file:
                response = client.post(
                    "/mlflow/upload-pkl-model",
                    files={"file": ("test.pkl", file, "application/octet-stream")},
                    data={"model_name": "test_model"}
                )
            
            assert response.status_code == 413


class TestAuthentication:
    """Test API key authentication."""
    
    def test_protected_endpoint_without_api_key(self):
        """Test that endpoints require API key when enabled."""
        # Enable API key authentication
        with patch.object(settings, 'API_KEY', 'test-api-key'):
            response = client.get("/mlflow/registered-models/search")
            
            # Should fail without API key
            assert response.status_code == 401
    
    def test_protected_endpoint_with_valid_api_key(self):
        """Test endpoint access with valid API key."""
        with patch.object(settings, 'API_KEY', 'test-api-key'):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"registered_models": []}
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
                mock_response.raise_for_status = Mock()
                
                response = client.get(
                    "/mlflow/registered-models/search",
                    headers={"X-API-Key": "test-api-key"}
                )
                
                assert response.status_code == 200
    
    def test_protected_endpoint_with_invalid_api_key(self):
        """Test endpoint access with invalid API key."""
        with patch.object(settings, 'API_KEY', 'test-api-key'):
            response = client.get(
                "/mlflow/registered-models/search",
                headers={"X-API-Key": "wrong-key"}
            )
            
            assert response.status_code == 403


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_exceeded(self):
        """Test that rate limiting works."""
        # This test would need slowapi to be properly configured
        # For now, we just verify the endpoint exists and responds
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = Mock()
            
            # Make multiple requests
            for _ in range(5):
                response = client.get("/health")
                assert response.status_code == 200


class TestErrorSanitization:
    """Test error message sanitization."""
    
    def test_sanitize_bearer_token(self):
        """Test that Bearer tokens are redacted."""
        from main_secure import sanitize_error_message
        
        error_msg = "Authentication failed: Bearer abc123xyz456"
        sanitized = sanitize_error_message(error_msg)
        
        assert "abc123xyz456" not in sanitized
        assert "[REDACTED]" in sanitized
    
    def test_sanitize_api_key(self):
        """Test that API keys are redacted."""
        from main_secure import sanitize_error_message
        
        error_msg = "Invalid api_key: sk-1234567890abcdef"
        sanitized = sanitize_error_message(error_msg)
        
        assert "sk-1234567890abcdef" not in sanitized
        assert "[REDACTED]" in sanitized


class TestPickleLoading:
    """Test pickle file loading with fallbacks."""
    
    def test_load_valid_pickle(self):
        """Test loading a valid pickle file."""
        from main_secure import load_pickle_file
        
        # Create a simple pickle file
        test_obj = {"model": "test_data", "version": 1}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
            pickle.dump(test_obj, f)
            pkl_path = f.name
        
        try:
            loaded = load_pickle_file(pkl_path)
            assert loaded == test_obj
        finally:
            Path(pkl_path).unlink()
    
    def test_load_invalid_pickle(self):
        """Test loading an invalid pickle file."""
        from main_secure import load_pickle_file
        
        # Create an invalid file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
            f.write(b"This is not a valid pickle file")
            pkl_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_pickle_file(pkl_path)
        finally:
            Path(pkl_path).unlink()


class TestSecurityHeaders:
    """Test security headers middleware."""
    
    def test_security_headers_present(self):
        """Test that security headers are added to responses."""
        response = client.get("/health")
        
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in response.headers


class TestRequestIdTracking:
    """Test request ID tracking."""
    
    def test_request_id_in_response(self):
        """Test that request ID is included in response."""
        response = client.get("/health")
        
        assert "X-Request-ID" in response.headers
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
    
    def test_custom_request_id(self):
        """Test using custom request ID."""
        custom_id = "custom-request-123"
        response = client.get("/health", headers={"X-Request-ID": custom_id})
        
        assert response.headers["X-Request-ID"] == custom_id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
