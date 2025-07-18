#\!/bin/bash
echo '🚀 Starting SutazAI v8 Production System...'
cd /opt/sutazaiapp
source venv/bin/activate
python3 simple_backend.py &
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
echo '✅ SutazAI v8 services started'
