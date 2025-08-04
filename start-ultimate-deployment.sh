#!/bin/bash
#
# Quick Start Script for Ultimate Deployment System
#
# Usage: ./start-ultimate-deployment.sh [environment]
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENT="${1:-local}"

echo "🚀 Starting Ultimate Deployment System"
echo "Environment: $ENVIRONMENT"
echo "Dashboard: http://localhost:7777"
echo "WebSocket: ws://localhost:7778"
echo "API: http://localhost:7779"
echo ""

cd "$PROJECT_ROOT"
python3 scripts/ultimate-deployment-master.py deploy --environment "$ENVIRONMENT"
