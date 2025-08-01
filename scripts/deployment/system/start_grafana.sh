#!/bin/bash
#
# start_grafana.sh - Script to start Grafana for metrics visualization
#

set -e

# Source environment variables
if [ -f /opt/sutazaiapp/.env ]; then
    source <(grep -v '^#' /opt/sutazaiapp/.env | sed -E 's/(.*)=(.*)/export \1="\2"/')
fi

# Configuration
GRAFANA_DIR="${GRAFANA_DIR:-/opt/sutazaiapp/monitoring/grafana}"
GRAFANA_DATA_DIR="${GRAFANA_DATA_DIR:-/opt/sutazaiapp/data/grafana}"
GRAFANA_CONFIG_DIR="${GRAFANA_CONFIG_DIR:-/opt/sutazaiapp/config/grafana}"
GRAFANA_VERSION="${GRAFANA_VERSION:-10.2.3}"
GRAFANA_PORT="${GRAFANA_PORT:-3001}"  # Use 3001 to avoid conflict with Web UI
LOG_DIR="${LOGS_DIR:-/opt/sutazaiapp/logs}/grafana"

# Create directories if they don't exist
mkdir -p "$GRAFANA_DIR"
mkdir -p "$GRAFANA_DATA_DIR"
mkdir -p "$GRAFANA_CONFIG_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$GRAFANA_DATA_DIR/plugins"
mkdir -p "$GRAFANA_DATA_DIR/provisioning/datasources"
mkdir -p "$GRAFANA_DATA_DIR/provisioning/dashboards"

echo "Starting Grafana on port $GRAFANA_PORT..."

# Check if Grafana binary exists, download if not
GRAFANA_BIN="$GRAFANA_DIR/bin/grafana"
if [ ! -f "$GRAFANA_BIN" ]; then
    echo "Grafana binary not found. Downloading..."
    
    # Determine system architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        GRAFANA_ARCH="amd64"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        GRAFANA_ARCH="arm64"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
    
    # Download and extract Grafana
    TEMP_DIR=$(mktemp -d)
    pushd "$TEMP_DIR"
    wget "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-${GRAFANA_ARCH}.tar.gz" -O grafana.tar.gz
    tar -xzf grafana.tar.gz
    mkdir -p "$GRAFANA_DIR/bin"
    cp -r grafana-*/bin/* "$GRAFANA_DIR/bin/"
    cp -r grafana-*/public "$GRAFANA_DIR/"
    cp -r grafana-*/conf "$GRAFANA_DIR/"
    popd
    rm -rf "$TEMP_DIR"
    
    echo "Grafana downloaded and installed."
fi

# Create default datasource configuration if it doesn't exist
DATASOURCE_CONFIG="$GRAFANA_DATA_DIR/provisioning/datasources/prometheus.yaml"
if [ ! -f "$DATASOURCE_CONFIG" ]; then
    cat > "$DATASOURCE_CONFIG" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    version: 1
    editable: true
EOF
    echo "Created default Prometheus datasource configuration."
fi

# Create default dashboard configuration if it doesn't exist
DASHBOARD_CONFIG="$GRAFANA_DATA_DIR/provisioning/dashboards/sutazai.yaml"
if [ ! -f "$DASHBOARD_CONFIG" ]; then
    cat > "$DASHBOARD_CONFIG" << EOF
apiVersion: 1

providers:
  - name: 'SutazAI'
    orgId: 1
    folder: 'SutazAI'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: $GRAFANA_DATA_DIR/dashboards
EOF
    echo "Created default dashboard provider configuration."
fi

# Create default Grafana custom configuration if it doesn't exist
GRAFANA_CUSTOM_CONFIG="$GRAFANA_CONFIG_DIR/custom.ini"
if [ ! -f "$GRAFANA_CUSTOM_CONFIG" ]; then
    mkdir -p "$(dirname "$GRAFANA_CUSTOM_CONFIG")"
    cat > "$GRAFANA_CUSTOM_CONFIG" << EOF
[server]
http_port = $GRAFANA_PORT
domain = localhost
root_url = %(protocol)s://%(domain)s:%(http_port)s/

[paths]
data = $GRAFANA_DATA_DIR
logs = $LOG_DIR
plugins = $GRAFANA_DATA_DIR/plugins

[security]
admin_user = admin
admin_password = admin
EOF
    echo "Created custom Grafana configuration at $GRAFANA_CUSTOM_CONFIG"
fi

# Start Grafana with the configuration
exec "$GRAFANA_BIN" \
    --homepath="$GRAFANA_DIR" \
    --config="$GRAFANA_CUSTOM_CONFIG" \
    cfg:default.paths.provisioning="$GRAFANA_DATA_DIR/provisioning" \
    2>&1 | tee -a "$LOG_DIR/grafana.log"
