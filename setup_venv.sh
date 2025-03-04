#!/bin/bash
sshpass -p "1988" ssh root@192.168.100.100 "cd /opt/sutazaiapp && python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
