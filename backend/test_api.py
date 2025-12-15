import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_search_models():
    """Search for registered models"""
    response = requests.post(
        f"{BASE_URL}/mlflow/registered-models/search",
        json={"max_results": 10}
    )
    print("\nRegistered Models:", json.dumps(response.json(), indent=2))

def test_create_model_version():
    """Example: Register a new model version"""
    payload = {
        "name": "example-model",
        "source": "runs:/abc123def456/model",
        "description": "Example model version",
        "tags": [
            {"key": "environment", "value": "development"},
            {"key": "framework", "value": "scikit-learn"}
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/mlflow/model-versions/create",
        json=payload
    )
    
    if response.ok:
        print("\nModel Version Created:", json.dumps(response.json(), indent=2))
    else:
        print("\nError:", response.text)

def test_get_model_version():
    """Get specific model version metadata"""
    response = requests.get(
        f"{BASE_URL}/mlflow/model-versions/get",
        params={"name": "example-model", "version": "1"}
    )
    
    if response.ok:
        print("\nModel Version Details:", json.dumps(response.json(), indent=2))
    else:
        print("\nError:", response.text)

def test_invoke_model():
    """Example: Invoke a model"""
    payload = {
        "model_url": "http://localhost:5001/invocations",
        "inputs": {
            "feature1": 1.5,
            "feature2": 2.3,
            "feature3": "category_a"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/serve/invoke",
        json=payload
    )
    
    if response.ok:
        result = response.json()
        print(f"\nModel Response: {json.dumps(result['predictions'], indent=2)}")
        print(f"Latency: {result['latency_ms']}ms")
    else:
        print("\nError:", response.text)

if __name__ == "__main__":
    print("Testing MLflow Model Registry API\n" + "="*50)
    
    # Run tests
    test_health()
    test_search_models()
    
    # Uncomment to test other endpoints:
    # test_create_model_version()
    # test_get_model_version()
    # test_invoke_model()
