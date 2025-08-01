#!/bin/bash

echo "🚀 Starting SutazAI Simple Version..."

# Kill any existing Python processes on ports
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8501 | xargs kill -9 2>/dev/null

# Start backend
cd /opt/sutazaiapp/backend
nohup python3 -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
echo "Backend PID: $!"

# Wait for backend to start
sleep 3

# Start frontend
cd /opt/sutazaiapp/frontend
export BACKEND_HOST=172.31.77.193
export BACKEND_PORT=8000
nohup streamlit run fixed_app.py --server.port 8501 --server.address 0.0.0.0 > frontend.log 2>&1 &
echo "Frontend PID: $!"

echo "✅ SutazAI is running!"
echo ""
echo "Access:"
echo "  Frontend: http://172.31.77.193:8501"
echo "  Backend: http://172.31.77.193:8000"
echo ""
echo "Logs:"
echo "  Backend: /opt/sutazaiapp/backend/backend.log"
echo "  Frontend: /opt/sutazaiapp/frontend/frontend.log"
echo ""
echo "To stop: killall python3"