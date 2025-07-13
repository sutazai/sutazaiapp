#!/bin/bash
# SutazAI Application Startup Script

set -e

echo "🤖 Starting SutazAI - Advanced AI Agent System"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration before running again."
    exit 1
fi

# Create necessary directories
mkdir -p data logs backups data/uploads data/chromadb

# Export environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the application
echo "🚀 Launching SutazAI application..."
echo "📊 Access the API documentation at: http://localhost:8000/docs"
echo "🩺 Health check available at: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Check if development mode is requested
if [[ "$1" == "--dev" ]]; then
    echo "🔧 Starting in DEVELOPMENT mode..."
    python3 main.py --dev
else
    echo "🏭 Starting in PRODUCTION mode..."
    python3 main.py
fi