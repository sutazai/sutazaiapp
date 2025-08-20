#!/bin/bash
export PYTHONPATH=/opt/sutazaiapp:$PYTHONPATH
cd /opt/sutazaiapp/.mcp/UltimateCoderMCP
exec ./.venv/bin/python main.py "$@"
