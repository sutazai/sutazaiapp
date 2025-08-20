#!/bin/bash
export PYTHONPATH=/opt/sutazaiapp:$PYTHONPATH
cd /opt/sutazaiapp/.venvs/extended-memory
exec ./bin/python main.py "$@"
