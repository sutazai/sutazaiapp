# CLN-20250808-CHECKSUM-AND-JARVIS-SCHEMAS

Change Type: Deduplication and centralization
Date: 2025-08-08
Author: System Architect

Summary:
- Centralized file checksums for backup automation scripts in `scripts/lib/file_utils.py` and refactored 6 modules to reuse it.
- Centralized Jarvis service request/response schemas in `services/jarvis/schemas.py` and refactored `main.py`, `main_simple.py`, `main_basic.py` to import.
- Centralized backend model-name validation in `backend/app/utils/validation.py` and updated validators in API models to call it.

Outcome:
- Duplicate groups reduced further (→ 43). Less surface area for drift.

Traceability:
- Task IDs: CLN-20250808-CHECKSUM-AND-JARVIS-SCHEMAS

Validation Plan:
- Static discovery updated (conflict_map.json).
- Manual smoke import for changed modules under appropriate PYTHONPATH.

