---
document_id: "DOC-2025-OPS-CLN-0002"
title: "500-Agent Role Mapping (Read-only investigation kickoff)"
created_date: "2025-08-19 21:10:00 UTC"
created_by: "system-architect.md"
last_modified: "2025-08-19 21:10:00 UTC"
status: "active"
owner: "architecture.team@sutazaiapp.local"
category: "process"
---

# 500-Agent Role Mapping

Investigation-first teams; execution steps gated by reports and approvals. Traceability via /reports/cleanup ledger.

- Architecture (90)
  - System (20), Backend (15), Frontend (15), API (15), Mesh/Network (25)
- Refactoring squads (160)
  - Backend (60), Frontend (50), Shared/utils (30), Docker/infra (20)
- MCP auditing (Rule 20, read-only) (30)
  - Config integrity (10), Wrapper presence/health read checks (10), DIND audits (10)
- Testing (120)
  - Unit (40), Integration (40), E2E/Playwright (30), Performance (10)
- Documentation & compliance (50)
  - Authority sync (20), CHANGELOG hygiene (15), Diagrams alignment (15)
- Observability & perf (30)
- Change intelligence & reporting (20)

All outputs must:
- Reference authoritative docs under /IMPORTANT
- Update /reports/cleanup/* with evidence artifacts
- Keep MCP untouched unless explicit authorization is provided
