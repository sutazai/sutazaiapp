#!/bin/bash
#
# start_prometheus.sh - Script to start Prometheus monitoring service
#

set -e

# Source environment variables
if [ -f /opt/sutazaiapp/.env ]; then
    source <(grep -v '^#' /opt/sutazaiapp/.env | sed -E 's/(.*)=(.*)/export \1="\2"/')
fi

# Configuration
PROMETHEUS_DIR="${PROMETHEUS_DIR:-/opt/sutazaiapp/monitoring/prometheus}"
PROMETHEUS_DATA_DIR="${PROMETHEUS_DATA_DIR:-/opt/sutazaiapp/data/prometheus}"
PROMETHEUS_CONFIG="${PROMETHEUS_CONFIG:-/opt/sutazaiapp/config/prometheus/prometheus.yml}"
PROMETHEUS_VERSION="${PROMETHEUS_VERSION:-2.48.1}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
LOG_DIR="${LOGS_DIR:-/opt/sutazaiapp/logs}/prometheus"

# Create directories if they don't exist
mkdir -p "$PROMETHEUS_DIR"
mkdir -p "$PROMETHEUS_DATA_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$PROMETHEUS_CONFIG")"

echo "Starting Prometheus monitoring service on port $PROMETHEUS_PORT..."

# Check if Prometheus binary exists, download if not
PROMETHEUS_BIN="$PROMETHEUS_DIR/prometheus"
if [ ! -f "$PROMETHEUS_BIN" ]; then
    echo "Prometheus binary not found. Downloading..."
    
    # Determine system architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        PROM_ARCH="amd64"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        PROM_ARCH="arm64"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
    
    # Download and extract Prometheus
    TEMP_DIR=$(mktemp -d)
    pushd "$TEMP_DIR"
    wget "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${PROM_ARCH}.tar.gz" -O prometheus.tar.gz
    tar -xzf prometheus.tar.gz
    mv prometheus-*/prometheus "$PROMETHEUS_BIN"
    mv prometheus-*/promtool "$PROMETHEUS_DIR/"
    mkdir -p "$PROMETHEUS_DIR/consoles" "$PROMETHEUS_DIR/console_libraries"
    cp -r prometheus-*/consoles/* "$PROMETHEUS_DIR/consoles/"
    cp -r prometheus-*/console_libraries/* "$PROMETHEUS_DIR/console_libraries/"
    popd
    rm -rf "$TEMP_DIR"
    
    # Set permissions
    chmod +x "$PROMETHEUS_BIN"
    echo "Prometheus binary downloaded and installed."
fi

# Create default config if it doesn't exist
if [ ! -f "$PROMETHEUS_CONFIG" ]; then
    cat > "$PROMETHEUS_CONFIG" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "sutazai-api"
    metrics_path: /metrics
    static_configs:
      - targets: ["localhost:8000"]

  - job_name: "node"
    static_configs:
      - targets: ["localhost:9100"]
EOF
    echo "Created default Prometheus configuration at $PROMETHEUS_CONFIG"
fi

# Start Prometheus with the configuration
exec "$PROMETHEUS_BIN" \
    --config.file="$PROMETHEUS_CONFIG" \
    --storage.tsdb.path="$PROMETHEUS_DATA_DIR" \
    --web.console.templates="$PROMETHEUS_DIR/consoles" \
    --web.console.libraries="$PROMETHEUS_DIR/console_libraries" \
    --web.listen-address=0.0.0.0:$PROMETHEUS_PORT \
    2>&1 | tee -a "$LOG_DIR/prometheus.log"
