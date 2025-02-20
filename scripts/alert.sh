#!/bin/bash
# New alert script
MESSAGE=$1
curl -X POST -H 'Content-Type: application/json' \
  -d "{\"text\":\"$MESSAGE\"}" \
  $WEBHOOK_URL 