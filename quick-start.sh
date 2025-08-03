#!/bin/bash

echo "🚀 Starting Hygiene Enforcement System..."

# Kill any existing processes
pkill -f rule-control-manager
pkill -f "python3 -m http.server"

# Start API
echo "Starting API..."
cd /opt/sutazaiapp
python scripts/agents/rule-control-manager.py --port 8100 &
sleep 3

# Start Dashboard
echo "Starting Dashboard..."
cd dashboard/hygiene-monitor
python3 -m http.server 8080 &
sleep 2

echo "✅ System Started!"
echo "📊 Dashboard: http://localhost:8080"
echo "🔧 API: http://localhost:8100"
echo ""
echo "Test it with: curl http://localhost:8100/api/rules | jq"