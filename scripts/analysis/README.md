# Read-only Analysis Scripts

These scripts provide investigation scaffolding only. They do not change runtime state, MCP configuration, or containers.

- `docker_inventory.sh`: lists docker-related files (Dockerfiles/compose) statically.
- `reconcile_portregistry_vs_compose.sh`: compares IMPORTANT/diagrams/PortRegistry.md with compose files (heuristic). Outputs a reconciliation report.
- `mcp_readonly_selfcheck.sh`: Rule 20-safe MCP audit (presence of .mcp.json, wrappers, basic deps).
- `playwright_discovery.sh`: finds Playwright configs and candidate tests.
- `live_logs_sanity.sh`: validates live_logs.sh via dry-run status.
- `compliance_scan.sh`: CHANGELOG.md presence scan (informational).
- `run_all_readonly_audits.sh`: executes all the above in a safe sequence.

Outputs are written to `/opt/sutazaiapp/reports/cleanup` with timestamped filenames.
