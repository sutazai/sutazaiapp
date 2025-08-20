---
document_id: "DOC-2025-OPS-MCP-0001"
title: "MCP Audit Procedure (Rule 20: Read-only)"
created_date: "2025-08-19 21:10:00 UTC"
created_by: "security-manager.md"
last_modified: "2025-08-19 21:10:00 UTC"
status: "active"
owner: "security.team@sutazaiapp.local"
category: "process"
---

# MCP Audit Procedure (Read-only)

- Absolute rule: Do not modify `.mcp.json`, wrappers, or MCP infra without explicit authorization.
- Audits must be read-only and evidence-based.

## Steps
1. Verify presence of `.mcp.json` and `/scripts/mcp/wrappers`.
2. List declared STDIO servers and associated wrapper commands.
3. Confirm wrapper scripts exist and have execute bit.
4. Optional: check for required binaries (npx, node, python3, jq), report missing.
5. Do not attempt to start/stop MCP servers during audit.

## Tooling
- `/scripts/analysis/mcp_readonly_selfcheck.sh`: generates a timestamped report in `/reports/cleanup`.

## Reporting
- All findings must be logged into `/reports/cleanup` with timestamps.
- Any deviations between docs and config must be flagged for architect review.
