#!/bin/bash
# Automatically switch to backup server if primary fails
if ! curl -sSf http://primary-server/health; then
    ./switch_to_backup.sh
    echo "Failover to backup server completed"
fi 