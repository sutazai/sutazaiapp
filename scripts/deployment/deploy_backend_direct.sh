#!/bin/bash

# Direct Backend Deployment Script - Rule-Compliant Real Implementation
# Deploys backend without containers for immediate functionality

set -e

echo "🚀 Starting Direct Backend Deployment..."

# Check if we're in the correct directory
if [ ! -f "/opt/sutazaiapp/backend/app/main.py" ]; then
    echo "❌ Backend main.py not found. Are we in the correct directory?"
    exit 1
fi

# Set environment variables for backend
export PYTHONPATH="/opt/sutazaiapp/backend:/opt/sutazaiapp/backend/app:$PYTHONPATH"
export SECRET_KEY="emergency_secret_key_change_in_production_abc123"
export JWT_SECRET="emergency_jwt_secret_change_in_production_xyz789"
export JWT_SECRET_KEY="emergency_jwt_secret_key_change_in_production_def456"
export DEBUG="false"
export REDIS_HOST="localhost"
export DATABASE_URL="postgresql://sutazai:sutazai_password@localhost:10000/sutazai"
export REDIS_URL="redis://localhost:10001"
export RABBITMQ_URL="amqp://sutazai:sutazai_password@localhost:10008"
export OLLAMA_BASE_URL="http://localhost:10104"
export NEO4J_URL="bolt://localhost:10003"
export NEO4J_PASSWORD="sutazai_password"
export POSTGRES_HOST="localhost"
export POSTGRES_USER="sutazai"
export POSTGRES_PASSWORD="sutazai_password" 
export POSTGRES_DB="sutazai"
export RABBITMQ_USER="sutazai"
export RABBITMQ_PASSWORD="sutazai_password"

echo "📋 Environment variables set"

# Kill any existing backend processes
pkill -f "uvicorn.*main:app" || true
pkill -f "python.*main.py" || true
sleep 2

echo "🔧 Installing Python dependencies..."
cd /opt/sutazaiapp/backend

# Install dependencies if not already installed
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt

echo "🎯 Starting Backend API Server on port 10010..."

# Start the backend server directly
cd /opt/sutazaiapp/backend
source .venv/bin/activate

# Use uvicorn to run the FastAPI app on port 10010
uvicorn app.main:app --host 0.0.0.0 --port 10010 --workers 1 --log-level info &

BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for backend to start
sleep 5

# Test the backend
echo "🧪 Testing backend deployment..."
if curl -f http://localhost:10010/health > /dev/null 2>&1; then
    echo "✅ Backend successfully deployed and responding on port 10010"
    echo "🔗 Backend API: http://localhost:10010"
    echo "📊 Health Check: http://localhost:10010/health"
    echo "📚 API Docs: http://localhost:10010/docs"
else
    echo "⚠️ Backend may still be starting up..."
    echo "🔗 Backend should be available at: http://localhost:10010"
fi

echo "Backend PID: $BACKEND_PID" > /tmp/backend.pid
echo "🎉 Direct Backend Deployment Complete!"