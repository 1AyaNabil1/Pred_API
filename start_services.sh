#!/bin/bash
# This script starts both MLflow and the FastAPI backend

echo "Starting Pixonal Services..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "Virtual environment not found. Please run:"
    echo "   cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source backend/venv/bin/activate

# Check if MLflow is already running on port 5001
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
    echo "MLflow is already running on port 5001"
else
    echo "Starting MLflow server on port 5001..."
    mlflow server \
        --host 0.0.0.0 \
        --port 5001 \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlruns \
        > mlflow.log 2>&1 &
    
    MLFLOW_PID=$!
    echo "   MLflow PID: $MLFLOW_PID"
    
    # Wait for MLflow to start
    echo "   Waiting for MLflow to start..."
    sleep 3
    
    if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
        echo "   MLflow started successfully"
    else
        echo "   Failed to start MLflow. Check mlflow.log for details."
        exit 1
    fi
fi

echo ""

# Check if FastAPI backend is already running on port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Backend is already running on port 8000"
else
    echo "Starting FastAPI backend on port 8000..."
    cd backend
    uvicorn main_secure:app --host 0.0.0.0 --port 8000 --reload > ../backend.log 2>&1 &
    BACKEND_PID=$!
    cd ..
    echo "   Backend PID: $BACKEND_PID"
    
    # Wait for backend to start
    echo "   Waiting for backend to start..."
    sleep 2
    
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        echo "   Backend started successfully"
    else
        echo "   Failed to start backend. Check backend.log for details."
        exit 1
    fi
fi

echo ""
echo "All services started successfully!"
echo ""
echo "MLflow UI:        http://localhost:5001"
echo "Backend API:      http://localhost:8000"
echo "API Docs:         http://localhost:8000/docs"
echo ""
echo "To start the frontend, run:"
echo "   cd frontend && npm start"
echo ""
echo "To stop services, run:"
echo "   ./stop_services.sh"
echo ""
