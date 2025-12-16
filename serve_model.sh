#!/bin/bash
# ================================================================
# Pixonal MLflow Model Serving Script
# ================================================================
# This script starts MLflow model serving with a specified model URI.
#
# Usage:
#   ./serve_model.sh <model_name> [version|stage]
#
# Examples:
#   ./serve_model.sh MyModel 1              # Serve version 1
#   ./serve_model.sh MyModel Production     # Serve Production stage
#   ./serve_model.sh MyModel Staging        # Serve Staging stage
#   ./serve_model.sh MyModel                # Defaults to Production
#
# MLflow Documentation:
# - Model Serving: https://mlflow.org/docs/latest/deployment/deploy-model-locally.html
# - Model URIs: https://mlflow.org/docs/latest/model-registry.html#loading-registered-models
# ================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVING_PORT=${SERVING_PORT:-5002}
SERVING_HOST=${SERVING_HOST:-0.0.0.0}
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5001}

# Parse arguments
MODEL_NAME="${1:-}"
VERSION_OR_STAGE="${2:-Production}"

if [ -z "$MODEL_NAME" ]; then
    echo -e "${RED}Error: Model name is required${NC}"
    echo ""
    echo "Usage: $0 <model_name> [version|stage]"
    echo ""
    echo "Examples:"
    echo "  $0 MyModel 1              # Serve version 1"
    echo "  $0 MyModel Production     # Serve Production stage"
    echo "  $0 MyModel Staging        # Serve Staging stage"
    echo "  $0 MyModel                # Defaults to Production"
    exit 1
fi

# Construct model URI
# Ref: https://mlflow.org/docs/latest/model-registry.html#loading-registered-models
MODEL_URI="models:/${MODEL_NAME}/${VERSION_OR_STAGE}"

echo "================================================================"
echo -e "${BLUE}Pixonal MLflow Model Serving${NC}"
echo "================================================================"
echo ""
echo -e "Model Name:    ${GREEN}${MODEL_NAME}${NC}"
echo -e "Version/Stage: ${GREEN}${VERSION_OR_STAGE}${NC}"
echo -e "Model URI:     ${GREEN}${MODEL_URI}${NC}"
echo ""
echo -e "Tracking URI:  ${MLFLOW_TRACKING_URI}"
echo -e "Serving at:    http://${SERVING_HOST}:${SERVING_PORT}"
echo ""

# Check if port is already in use
if lsof -Pi :${SERVING_PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}Warning: Port ${SERVING_PORT} is already in use.${NC}"
    echo "Stopping existing process..."
    lsof -ti:${SERVING_PORT} | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Export tracking URI for MLflow
export MLFLOW_TRACKING_URI

echo "Starting MLflow Model Serving..."
echo ""

# Start model serving
# Ref: https://mlflow.org/docs/latest/cli.html#mlflow-models-serve
#
# Options:
#   --model-uri: URI of the model to serve
#   --host: Host to bind to
#   --port: Port to bind to
#   --no-conda: Don't use Conda (use local Python environment)
#   --env-manager: How to manage model dependencies (local = use current env)
#
python3 -m mlflow models serve \
    --model-uri "${MODEL_URI}" \
    --host ${SERVING_HOST} \
    --port ${SERVING_PORT} \
    --no-conda \
    --env-manager local &

SERVING_PID=$!

echo -e "Model Serving PID: ${SERVING_PID}"
echo ""

# Wait for the server to start
echo "Waiting for model server to be ready..."
sleep 5

# Check if server is running
if lsof -Pi :${SERVING_PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo ""
    echo "================================================================"
    echo -e "${GREEN}✅ Model Serving Started Successfully${NC}"
    echo "================================================================"
    echo ""
    echo "Model is now being served at:"
    echo -e "  Health:      ${GREEN}http://localhost:${SERVING_PORT}/health${NC}"
    echo -e "  Predictions: ${GREEN}http://localhost:${SERVING_PORT}/invocations${NC}"
    echo ""
    echo "Test with curl:"
    echo '  curl -X POST http://localhost:'${SERVING_PORT}'/invocations \'
    echo '    -H "Content-Type: application/json" \'
    echo '    -d '"'"'{"dataframe_split": {"columns": ["feature1", "feature2"], "data": [[1.0, 2.0]]}}'"'"
    echo ""
    echo "MLflow Payload Formats (per MLflow docs):"
    echo "  - dataframe_split: {columns: [...], data: [[...]]}  (recommended)"
    echo "  - dataframe_records: [{col1: val1, ...}]"
    echo "  - instances: [[val1, val2, ...]]"
    echo "  - inputs: {...} (flexible)"
    echo ""
    echo "Press Ctrl+C to stop the server"
    
    # Wait for the process
    wait $SERVING_PID
else
    echo ""
    echo -e "${RED}Error: Model serving failed to start${NC}"
    echo ""
    echo "Possible causes:"
    echo "  1. Model '${MODEL_NAME}' version '${VERSION_OR_STAGE}' does not exist"
    echo "  2. MLflow tracking server is not running at ${MLFLOW_TRACKING_URI}"
    echo "  3. Model artifacts are not accessible"
    echo ""
    echo "To verify the model exists:"
    echo "  curl -X POST ${MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/get \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"name\": \"${MODEL_NAME}\"}'"
    exit 1
fi
