#!/bin/bash
"""
SutazAI v9 System Starter
"""

echo "🚀 Starting SutazAI v9 System"
echo "=========================================="

# Kill any existing processes
pkill -f "simple_backend.py" 2>/dev/null
pkill -f "simple_frontend.py" 2>/dev/null
pkill -f "streamlit.*simple_frontend" 2>/dev/null

sleep 2

# Start backend
echo "🔧 Starting backend server..."
cd /opt/sutazaiapp
python3 simple_backend.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Test backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend started successfully (PID: $BACKEND_PID)"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Start frontend
echo "🌐 Starting frontend server..."
streamlit run simple_frontend.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
FRONTEND_PID=$!

# Wait for frontend
sleep 8

echo ""
echo "=========================================="
echo "🎉 SutazAI v9 System Started!"
echo "=========================================="
echo "🌐 Frontend: http://localhost:8501"
echo "🔧 Backend:  http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "=========================================="
echo ""
echo "Backend PID:  $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "To stop the system:"
echo "kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "🎯 Access your SutazAI system at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop monitoring"

# Monitor processes
while true; do
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "⚠️ Backend process stopped"
        break
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "⚠️ Frontend process stopped"
        break
    fi
    sleep 5
done

echo "🛑 System monitoring stopped"