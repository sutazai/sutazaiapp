#!/bin/bash
expect -c "spawn rsync -av --exclude=venv --exclude=__pycache__ /opt/sutazaiapp/ sutazaiapp_dev@192.168.100.100:/opt/sutazaiapp/; expect \"password:\"; send \"1988\r\"; expect eof"