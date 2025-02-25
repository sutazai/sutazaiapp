#!/bin/bash
# Send alerts to monitoring system
MESSAGE=$1
WEBHOOK_URL="https://hooks.example.com/alerts"

curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"$MESSAGE\"}" \
  $WEBHOOK_URL 