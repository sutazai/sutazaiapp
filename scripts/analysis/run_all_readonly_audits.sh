#!/usr/bin/env bash
# Purpose: Run all read-only audits and generate reports under /reports/cleanup
# Safety: Does not modify MCP or runtime configs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/docker_inventory.sh"
bash "${SCRIPT_DIR}/reconcile_portregistry_vs_compose.sh"
bash "${SCRIPT_DIR}/mcp_readonly_selfcheck.sh"
bash "${SCRIPT_DIR}/playwright_discovery.sh"
bash "${SCRIPT_DIR}/live_logs_sanity.sh"
bash "${SCRIPT_DIR}/compliance_scan.sh"

echo "All read-only audits executed. See /opt/sutazaiapp/reports/cleanup for outputs."