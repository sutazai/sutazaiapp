# Agent Role Assignments (200 Claude Agents)

Source of truth: coordination_bus/agents.csv

- Architecture Analysis: Claude-001 – Claude-020
- Dependency & Conflict Mapping: Claude-021 – Claude-040
- Backend Refactoring: Claude-041 – Claude-070
- Services & Infrastructure Cleanup: Claude-071 – Claude-100
- Agents Protocol & Orchestration: Claude-101 – Claude-130
- Testing & Integration: Claude-131 – Claude-160
- Performance & Security: Claude-161 – Claude-180
- Documentation & Reporting: Claude-181 – Claude-200

Each agent reports via `coordination_bus/messages/status.jsonl`. Heartbeats go to `coordination_bus/messages/heartbeats.jsonl`. Directives originate from Architect in `coordination_bus/messages/directives.jsonl`.

Trace all changes with task IDs (CLN-YYYYMMDD-XXXX) and update the ledger at `reports/cleanup/ledger.jsonl`.

