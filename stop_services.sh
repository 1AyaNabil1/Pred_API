#!/bin/bash

# Pixonal Studio - Service Stop Script
# This script stops MLflow and FastAPI backend services

echo "Stopping Pixonal Services..."
echo ""

# Stop MLflow (port 5001)
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
    echo "Stopping MLflow server..."
    lsof -ti:5001 | xargs kill -9 2>/dev/null
    echo "   MLflow stopped"
else
    echo "MLflow is not running"
fi

# Stop FastAPI backend (port 8000)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Stopping FastAPI backend..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    echo "   Backend stopped"
else
    echo "Backend is not running"
fi

# Stop frontend if running (port 3000)
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Stopping frontend..."
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    echo "   Frontend stopped"
else
    echo "Frontend is not running"
fi

echo ""
echo "All services stopped successfully!"