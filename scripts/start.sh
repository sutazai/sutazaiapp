#!/bin/bash
#
# Activate the virtual environment
source /opt/sutazaiapp/venv-sutazaiapp/bin/activate

# Start the FastAPI service (assuming main.py initializes the FastAPI app)
nohup flask run --host=0.0.0.0 --port=5000 > /opt/sutazaiapp/logs/flask.log 2>&1 &

# Start the Streamlit UI (assuming the UI entry point is at ui/app.py)
nohup streamlit run /opt/sutazaiapp/ui/app.py --server.port=8501 > /opt/sutazaiapp/logs/streamlit.log 2>&1 &

echo "All services started."