#!/bin/bash
sshpass -p "1988" ssh root@192.168.100.100 "cd /opt/sutazaiapp && source venv/bin/activate && ./scripts/run_tests.sh"
