#\!/bin/bash
echo "🚀 SutazAI v8 Enhanced Startup"
echo "============================="
cd /opt/sutazaiapp
source venv/bin/activate

# Start core services
echo "Starting core Python services..."
python3 simple_backend.py &
BACKEND_PID=$\!

streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$\!

# Wait for services to start
sleep 10

# Test services
echo "Testing services..."
curl -s http://localhost:8000/health > /dev/null && echo "✅ Backend healthy" || echo "❌ Backend not ready"
curl -s http://localhost:8501/healthz > /dev/null && echo "✅ Frontend healthy" || echo "❌ Frontend not ready"

echo "🎊 SutazAI v8 is running\!"
echo "Backend PID: "
echo "Frontend PID: "
echo "Access: http://192.168.131.128:8501"
