#!/bin/bash
echo "Starting cache cleanup..."
find /opt/sutazaiapp -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /opt/sutazaiapp -name "*.pyc" -delete
find /opt/sutazaiapp -name "*.pyo" -delete
rm -rf /opt/sutazaiapp/.pytest_cache
rm -rf /opt/sutazaiapp/.ruff_cache
if [ -d "/opt/sutazaiapp/data/models/.cache" ]; then rm -rf /opt/sutazaiapp/data/models/.cache/*; fi
echo "Cache cleanup completed!"
