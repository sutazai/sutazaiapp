# Agent Implementation Guide (Claude local profiles)

Scope
- Approved local profiles only (R16). No external AI APIs.
- Orchestration aligns with ASoT module boundaries.

Key Topics
- Role templates and capabilities
- Task routing, idempotency, and retries
- Safety, guardrails, and audit logging

Operational Rules
- Tag every change with `TID-<epic>-<seq>` and write per-file changelogs.
- Report hourly on the coordination bus; escalate blockers immediately.

Citations
- Cleanup program: /opt/sutazaiapp/IMPORTANT/10_canonical/operations/agent_cleanup_200_claude.md#L1-L200

