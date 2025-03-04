#!/bin/bash
sshpass -p "1988" rsync -av --exclude=venv --exclude=__pycache__ --exclude=.git --exclude=.pytest_cache /opt/sutazaiapp/core_system /opt/sutazaiapp/ai_agents /opt/sutazaiapp/scripts /opt/sutazaiapp/backend /opt/sutazaiapp/tests root@192.168.100.100:/opt/sutazaiapp/
