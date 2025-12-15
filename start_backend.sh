#!/bin/bash

# Start Backend - Runs both MLflow UI and FastAPI concurrently
# This script ensures both services run smoothly together

set -e

echo "🚀 Starting Backend Services..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MLFLOW_PORT=5001
FASTAPI_PORT=8000
MLFLOW_DB="sqlite:///mlflow.db"
MLFLOW_HOST="0.0.0.0"
FASTAPI_HOST="0.0.0.0"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    pkill -P $$ || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if ports are already in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "⚠️  Port $port is already in use. Killing existing process..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# Start MLflow UI
start_mlflow() {
    echo -e "${GREEN}Starting MLflow UI on port $MLFLOW_PORT...${NC}"
    check_port $MLFLOW_PORT
    python3 -m mlflow ui \
        --backend-store-uri $MLFLOW_DB \
        --host $MLFLOW_HOST \
        --port $MLFLOW_PORT &
    MLFLOW_PID=$!
    echo "MLflow UI PID: $MLFLOW_PID"
}

# Start FastAPI
start_fastapi() {
    echo -e "${GREEN}Starting FastAPI on port $FASTAPI_PORT...${NC}"
    check_port $FASTAPI_PORT
    cd backend
    PYTHONPATH=/Users/ayanabil/Documents/Pixonal/Pred_API/backend python3 -m uvicorn main_secure:app \
        --host $FASTAPI_HOST \
        --port $FASTAPI_PORT \
        --reload &
    FASTAPI_PID=$!
    echo "FastAPI PID: $FASTAPI_PID"
    cd ..
}

# Main execution
echo "================================"
echo "  Backend Services Startup"
echo "================================"

start_mlflow
sleep 2  # Give MLflow time to initialize

start_fastapi
sleep 2  # Give FastAPI time to initialize

echo ""
echo "================================"
echo -e "${GREEN}✅ Backend Services Running${NC}"
echo "================================"
echo -e "📊 MLflow UI:  http://localhost:$MLFLOW_PORT"
echo -e "🔌 FastAPI:    http://localhost:$FASTAPI_PORT"
echo -e "📚 API Docs:   http://localhost:$FASTAPI_PORT/docs"
echo "================================"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait
