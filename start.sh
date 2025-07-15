#!/bin/bash
set -e

echo "🚀 Starting SutazAI System"
echo "=========================="

# Change to project directory
cd /opt/sutazaiapp

# Set environment variables
export PYTHONPATH=/opt/sutazaiapp:$PYTHONPATH
export SUTAZAI_ROOT=/opt/sutazaiapp

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the application
echo "🤖 Starting SutazAI application..."
python3 main.py --dev &

APP_PID=$!

echo "✅ SutazAI started successfully!"
echo "📊 Process ID: $APP_PID"
echo "🌐 Access at: http://localhost:8000"
echo "📚 API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the application"

# Wait for interrupt
trap "echo '🛑 Stopping SutazAI...'; kill $APP_PID; exit 0" INT
wait $APP_PID
