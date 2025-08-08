# CLN-20250808-EVENT-HANDLER-DEDUPE

Change Type: Deduplication (shared event handler registration)
Date: 2025-08-08
Author: System Architect

Summary:
- Introduced `backend/app/orchestration/event_utils.py` with `register_event_handler()`.
- Replaced duplicated `register_event_handler` implementations in:
  - `backend/ai_agents/core/agent_registry.py`
  - `backend/ai_agents/orchestration/agent_registry_service.py`
  - `backend/app/orchestration/workflow_engine.py` (async wrapper calls util)
- Standardizes handler de-duplication and lazy list initialization.

Backward Compatibility:
- Method signatures unchanged; behavior consistent, with added duplicate protection.

Traceability:
- Task ID: CLN-20250808-EVENT-HANDLER-DEDUPE
- Logged via coordination bus ledger.

Validation Plan:
- Smoke import modules with `PYTHONPATH=backend`.
- Re-run static discovery to confirm duplicate group not increased.

