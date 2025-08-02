#!/bin/bash

# Stop FastAPI service
pkill -f "flask run"

# Stop Streamlit service
pkill -f "streamlit run"

echo "All services stopped."