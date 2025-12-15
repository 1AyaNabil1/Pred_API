"""
Promote the latest version of a registered model to Production stage
"""
import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Initialize client
client = MlflowClient()

# Model name
MODEL_NAME = "MyModel"

# Get the latest version
latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

if not latest_versions:
    print(f"No versions found for model '{MODEL_NAME}'")
    print("Please run log_model_mlflow.py first to register a model.")
    exit(1)

latest_version = latest_versions[0]
version_number = latest_version.version

print(f"Found {MODEL_NAME} version {version_number}")
print(f"   Current stage: {latest_version.current_stage}")

# Transition to Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version_number,
    stage="Production",
    archive_existing_versions=True  # Archive any existing Production versions
)

print(f"Successfully promoted {MODEL_NAME} version {version_number} to Production!")
print("Your API can now load the model at: models:/MyModel/Production")
print("\nNext steps:")
print("  1. Restart the FastAPI server (it will auto-reload)")
print("  2. Test predictions at http://localhost:8000/docs")
