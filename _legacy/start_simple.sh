#\!/bin/bash
cd /opt/sutazaiapp
source venv/bin/activate
export PYTHONPATH=/opt/sutazaiapp
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
sleep 5
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
echo '✅ Services started'
