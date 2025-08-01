#!/bin/bash
#
# start_node_exporter.sh - Script to start Node Exporter for system metrics
#

set -e

# Source environment variables
if [ -f /opt/sutazaiapp/.env ]; then
    source <(grep -v '^#' /opt/sutazaiapp/.env | sed -E 's/(.*)=(.*)/export \1="\2"/')
fi

# Configuration
NODE_EXPORTER_DIR="${NODE_EXPORTER_DIR:-/opt/sutazaiapp/monitoring/node_exporter}"
NODE_EXPORTER_VERSION="${NODE_EXPORTER_VERSION:-1.7.0}"
NODE_EXPORTER_PORT="${NODE_EXPORTER_PORT:-9100}"
LOG_DIR="${LOGS_DIR:-/opt/sutazaiapp/logs}/node_exporter"

# Create directories if they don't exist
mkdir -p "$NODE_EXPORTER_DIR"
mkdir -p "$LOG_DIR"

echo "Starting Node Exporter on port $NODE_EXPORTER_PORT..."

# Check if Node Exporter binary exists, download if not
NODE_EXPORTER_BIN="$NODE_EXPORTER_DIR/node_exporter"
if [ ! -f "$NODE_EXPORTER_BIN" ]; then
    echo "Node Exporter binary not found. Downloading..."
    
    # Determine system architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        NE_ARCH="amd64"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        NE_ARCH="arm64"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
    
    # Download and extract Node Exporter
    TEMP_DIR=$(mktemp -d)
    pushd "$TEMP_DIR"
    wget "https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-${NE_ARCH}.tar.gz" -O node_exporter.tar.gz
    tar -xzf node_exporter.tar.gz
    mv node_exporter-*/node_exporter "$NODE_EXPORTER_BIN"
    popd
    rm -rf "$TEMP_DIR"
    
    # Set permissions
    chmod +x "$NODE_EXPORTER_BIN"
    echo "Node Exporter binary downloaded and installed."
fi

# Start Node Exporter
exec "$NODE_EXPORTER_BIN" \
    --web.listen-address=:$NODE_EXPORTER_PORT \
    --collector.filesystem.ignored-mount-points="^/(dev|proc|sys|var/lib/docker/.+|run/user)($|/)" \
    --collector.textfile.directory=/var/lib/node_exporter/textfile_collector \
    --collector.filesystem \
    --collector.diskstats \
    --collector.meminfo \
    --collector.netdev \
    --collector.cpu \
    --collector.loadavg \
    2>&1 | tee -a "$LOG_DIR/node_exporter.log"
