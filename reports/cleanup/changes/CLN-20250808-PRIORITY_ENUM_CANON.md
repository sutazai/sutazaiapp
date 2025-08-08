# CLN-20250808-PRIORITY-ENUM-CANON

Change Type: Canonicalization and de-duplication
Date: 2025-08-08
Author: System Architect

Summary:
- Introduced canonical `TaskPriority` and `AlertSeverity` in `backend/app/schemas/message_types.py`.
- Replaced duplicate enum definitions and adjusted call sites to use the canonical types.
- Ensured numeric scheduling logic uses `TaskPriority.rank` to avoid reliance on integer `.value`.

Files Changed:
- backend/app/schemas/message_types.py (+TaskPriority, +AlertSeverity)
- backend/app/models/agent.py (use canonical TaskPriority)
- backend/app/orchestration/task_router.py (use canonical TaskPriority; numeric rank)
- backend/app/orchestration/agent_orchestrator.py (use canonical TaskPriority)
- backend/app/orchestration/monitoring.py (use canonical AlertSeverity)
- backend/oversight/alert_notification_system.py (use canonical AlertSeverity)
- backend/edge_inference/telemetry.py (use canonical AlertSeverity)
- backend/federated_learning/monitoring.py (use canonical AlertSeverity)
- backend/ai_agents/core/orchestration_controller.py (use canonical TaskPriority; preserve numeric output)
- backend/ai_agents/specialized/orchestrator.py (use canonical TaskPriority)
- backend/ai_agents/orchestration/master_agent_orchestrator.py (use canonical TaskPriority)

Backward Compatibility:
- `TaskPriority` remains a string-valued Enum for API stability; numeric logic should call `.rank`.
- `AlertSeverity` values unchanged (`info|warning|error|critical`).

Traceability:
- Task ID: CLN-20250808-PRIORITY-ENUM-CANON
- Ledger entries emitted via `scripts/coord_bus.py`.

Validation Plan:
- Run `pytest -q` for affected modules where tests exist.
- Smoke-import updated modules.
- Linters: `flake8 backend/` and mypy spot-checks for changed files.

