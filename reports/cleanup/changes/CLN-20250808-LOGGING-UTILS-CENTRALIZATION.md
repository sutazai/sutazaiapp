# CLN-20250808-LOGGING-UTILS-CENTRALIZATION

Change Type: Deduplication (script logging helpers)
Date: 2025-08-08
Author: System Architect

Summary:
- Introduced `scripts/lib/logging_utils.py` with a single `setup_logging()` that supports either `verbose` or `level` usage.
- Updated DevOps health-check scripts to import shared helper instead of local duplicates.
- Updated `backend/scripts/deploy-federated-learning.py` to reuse shared helper.

Files Updated:
- scripts/devops/health_check_gateway.py
- scripts/devops/health_check_ollama.py
- scripts/devops/health_check_dataservices.py
- scripts/devops/infrastructure_health_check.py
- scripts/devops/health_check_monitoring.py
- scripts/devops/health_check_vectordb.py
- backend/scripts/deploy-federated-learning.py

Outcome:
- Duplicate group count reduced further (46 → 45) in conflict map.

Traceability:
- Task ID: CLN-20250808-LOGGING-UTILS-CENTRALIZATION

