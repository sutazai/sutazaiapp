#!/bin/bash
# Report incidents to monitoring system
INCIDENT=$1
curl -X POST -H "Content-Type: application/json" -d "{\"incident\": \"$INCIDENT\"}" http://monitoring-system/api/incidents
echo "Incident reported: $INCIDENT" 