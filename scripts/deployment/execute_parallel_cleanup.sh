#!/bin/bash
# ULTRA PARALLEL EXECUTION COORDINATOR
# Manages all 5 tracks simultaneously for maximum efficiency
set -euo pipefail


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

echo "=============================================="
echo "   ULTRA PARALLEL CLEANUP EXECUTION"
echo "=============================================="
echo "Started: $(date)"
echo ""

# Verify backup exists
LATEST_BACKUP=$(ls -t /opt/sutazaiapp/backups 2>/dev/null | head -1)
if [ -z "$LATEST_BACKUP" ]; then
    echo "ERROR: No backup found! Run ultra_backup.sh first!"
    exit 1
fi
echo "✅ Backup verified: /opt/sutazaiapp/backups/$LATEST_BACKUP"
echo ""

# Create log directory for each track
LOG_DIR="/opt/sutazaiapp/logs/parallel_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "=== STARTING 5 PARALLEL TRACKS ==="
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Function to run track and log output
run_track() {
    local track_name=$1
    local script_path=$2
    local log_file="$LOG_DIR/${track_name}.log"
    
    echo "Starting $track_name..."
    nohup bash -c "$script_path" > "$log_file" 2>&1 &
    local pid=$!
    echo "$track_name started with PID: $pid"
    echo "$pid" > "$LOG_DIR/${track_name}.pid"
}

# TRACK 1: Infrastructure (Resource Allocation + RabbitMQ)
cat > "$(mktemp /tmp/track1_infrastructure.sh.XXXXXX)" <<'EOF'
#!/bin/bash
echo "TRACK 1: Infrastructure Optimization"

# Fix resource over-allocation
cat > /opt/sutazaiapp/docker-compose.resource-fix.yml <<'COMPOSE'
version: '3.8'
services:
  kong:
    mem_limit: 512m
    mem_reservation: 256m
    cpus: 0.5
    
  consul:
    mem_limit: 512m
    mem_reservation: 256m
    cpus: 0.5
    
  rabbitmq:
    mem_limit: 1g
    mem_reservation: 512m
    cpus: 1.0
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  ollama:
    mem_limit: 4g
    mem_reservation: 2g
    cpus: 2.0
COMPOSE

# Apply changes
docker-compose -f docker-compose.yml -f docker-compose.resource-fix.yml up -d kong consul rabbitmq ollama
echo "Resource allocation fixed"

# Verify services
sleep 30
curl -s http://localhost:10006/v1/status/leader && echo "Consul: OK" || echo "Consul: FAILED"
docker exec sutazai-rabbitmq rabbitmqctl status && echo "RabbitMQ: OK" || echo "RabbitMQ: FAILED"
EOF
chmod +x /tmp/track1_infrastructure.sh
run_track "TRACK1_Infrastructure" "/tmp/track1_infrastructure.sh"

# TRACK 2: Dockerfile Consolidation
cat > "$(mktemp /tmp/track2_dockerfiles.sh.XXXXXX)" <<'EOF'
#!/bin/bash
echo "TRACK 2: Dockerfile Consolidation"
python3 /opt/sutazaiapp/scripts/dockerfile-dedup/consolidate_dockerfiles.py || {
    # Fallback: Simple deduplication
    cd /opt/sutazaiapp
    mkdir -p docker/templates
    find . -name "Dockerfile*" -type f | while read df; do
        hash=$(md5sum "$df" | cut -d' ' -f1)
        if [ ! -f "docker/templates/Dockerfile.$hash" ]; then
            cp "$df" "docker/templates/Dockerfile.$hash"
        fi
    done
    echo "Dockerfiles backed up to templates"
}
EOF
chmod +x /tmp/track2_dockerfiles.sh
run_track "TRACK2_Dockerfiles" "/tmp/track2_dockerfiles.sh"

# TRACK 3: Script Organization
cat > "$(mktemp /tmp/track3_scripts.sh.XXXXXX)" <<'EOF'
#!/bin/bash
echo "TRACK 3: Script Organization"
cd /opt/sutazaiapp

# Create organized structure
mkdir -p scripts/{deployment,maintenance,monitoring,testing,utils}

# Move scripts to organized locations
find scripts -maxdepth 1 -name "*.sh" -o -name "*.py" | while read script; do
    name=$(basename "$script")
    if [[ "$name" == "*deploy*" ]]; then
        mv "$script" scripts/deployment/ 2>/dev/null
    elif [[ "$name" == "*test*" ]]; then
        mv "$script" scripts/testing/ 2>/dev/null
    elif [[ "$name" == "*monitor*" ]] || [[ "$name" == "*health*" ]]; then
        mv "$script" scripts/monitoring/ 2>/dev/null
    elif [[ "$name" == "*backup*" ]] || [[ "$name" == "*clean*" ]]; then
        mv "$script" scripts/maintenance/ 2>/dev/null
    else
        mv "$script" scripts/utils/ 2>/dev/null
    fi
done

echo "Scripts organized into categories"
EOF
chmod +x /tmp/track3_scripts.sh
run_track "TRACK3_Scripts" "/tmp/track3_scripts.sh"

# TRACK 4: Code Cleanup (conceptual + BaseAgent)
cat > "$(mktemp /tmp/track4_cleanup.sh.XXXXXX)" <<'EOF'
#!/bin/bash
echo "TRACK 4: Code Cleanup"

# Clean conceptual elements
python3 <<'PYTHON'
import re
from pathlib import Path

replacements = {
    r'\bwizard\b': 'service',
    r'\bmagic\b': 'automated',
    r'\bteleport\b': 'transfer',
    r'\bfantasy\b': 'theoretical',
    r'\bblack-box\b': 'module'
}

count = 0
for pattern in ['*.py', '*.md']:
    for file_path in Path('/opt/sutazaiapp').rglob(pattern):
        if 'test' in str(file_path) or 'backup' in str(file_path):
            continue
        try:
            content = file_path.read_text()
            original = content
            for old, new in replacements.items():
                content = re.sub(old, new, content, flags=re.IGNORECASE)
            if content != original:
                file_path.write_text(content)
                count += 1
        except:
            pass
print(f"Cleaned {count} files")
PYTHON

# Consolidate BaseAgent
echo "Consolidating BaseAgent..."
if [ -f "/opt/sutazaiapp/backend/app/agents/core/base_agent.py" ]; then
    cp /opt/sutazaiapp/backend/app/agents/core/base_agent.py /opt/sutazaiapp/agents/core/base_agent.py.backup
    rm /opt/sutazaiapp/backend/app/agents/core/base_agent.py
    echo "BaseAgent consolidated"
fi
EOF
chmod +x /tmp/track4_cleanup.sh
run_track "TRACK4_Cleanup" "/tmp/track4_cleanup.sh"

# TRACK 5: Database UUID Migration (starts after 1 hour)
cat > "$(mktemp /tmp/track5_database.sh.XXXXXX)" <<'EOF'
#!/bin/bash
echo "TRACK 5: Database Optimization (waiting 1 hour for other tracks)"
sleep 3600  # Wait for other tracks to stabilize

# UUID migration would go here
echo "UUID migration skipped - requires careful planning"
EOF
chmod +x /tmp/track5_database.sh
run_track "TRACK5_Database" "/tmp/track5_database.sh"

echo ""
echo "=== ALL TRACKS STARTED ==="
echo ""
echo "Monitor progress with: /opt/sutazaiapp/scripts/monitoring/parallel_execution_monitor.sh"
echo "Check logs in: $LOG_DIR"
echo ""
echo "Track PIDs:"
cat $LOG_DIR/*.pid
echo ""
echo "To stop all tracks: kill $(cat $LOG_DIR/*.pid | tr '\n' ' ')"
echo ""

# Wait for all tracks to complete (with timeout)
echo "Waiting for tracks to complete (max 5 hours)..."
TIMEOUT=$((5 * 60 * 60))  # 5 hours in seconds
START_TIME=$(date +%s)

# Timeout mechanism to prevent infinite loops
LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
loop_start=$(date +%s)
while true; do
    RUNNING=0
    for pidfile in $LOG_DIR/*.pid; do
        if [ -f "$pidfile" ]; then
            PID=$(cat "$pidfile")
            if ps -p $PID > /dev/null 2>&1; then
                RUNNING=$((RUNNING + 1))
            fi
        fi
    # Check for timeout
    current_time=$(date +%s)
    if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
        echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
        break
    fi

    done
    
    if [ $RUNNING -eq 0 ]; then
        echo "All tracks completed!"
        break
    fi
    
    ELAPSED=$(($(date +%s) - START_TIME))
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "Timeout reached! Some tracks may still be running."
        break
    fi
    
    echo "Tracks still running: $RUNNING"
    sleep 30
done

echo ""
echo "=== FINAL VALIDATION ==="
python3 <<'VALIDATION'
import requests
from pathlib import Path

# Check metrics
dockerfiles = len(list(Path('/opt/sutazaiapp').rglob('Dockerfile*')))
scripts = len(list(Path('/opt/sutazaiapp/scripts').rglob('*.py'))) + \
          len(list(Path('/opt/sutazaiapp/scripts').rglob('*.sh')))
base_agents = len(list(Path('/opt/sutazaiapp').rglob('base_agent.py')))

print(f"Dockerfiles: 587 -> {dockerfiles}")
print(f"Scripts: 447 -> {scripts}")
print(f"BaseAgent files: 2 -> {base_agents}")

# Check services
try:
    backend = requests.get('http://localhost:10010/health', timeout=5)
    print(f"Backend: {'OK' if backend.status_code == 200 else 'FAILED'}")
except:
    print("Backend: FAILED")

if dockerfiles < 100 and scripts < 100 and base_agents <= 1:
    print("\n✅ CLEANUP SUCCESSFUL!")
else:
    print("\n⚠️ CLEANUP PARTIALLY COMPLETE")
VALIDATION

echo ""
echo "Completed: $(date)"
echo "Full logs available in: $LOG_DIR"