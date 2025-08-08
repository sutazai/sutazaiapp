# CLN-20250808-AGENTNAME-CHECKSUM-VALIDATION

Change Type: Deduplication and API validation centralization
Date: 2025-08-08
Author: System Architect

Summary:
- Centralized `get_agent_name()` into `agents/core/utils.py` and refactored:
  - `agents/fastapi_wrapper.py`, `agents/container_startup.py`, `agents/universal_startup.py`,
    `agents/standalone_main.py`, `container_startup.py`, and `need to be sorted/sutazaiapp/jarvis/container_startup.py`
- Eliminated duplicate `calculate_checksum` method bodies in backup automation scripts; use shared `scripts/lib/file_utils.py`.
- Unified model-name validation via `backend/app/utils/validation.py` and applied in chat/streaming/main request models.

Outcome:
- Duplicate groups reduced to 41 in `reports/cleanup/conflict_map.json`.

Traceability:
- Task ID: CLN-20250808-AGENTNAME-CHECKSUM-VALIDATION

