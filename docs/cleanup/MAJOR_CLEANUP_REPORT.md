# Major Cleanup Report — Phase 1 (Non-Destructive)

Date: {{commit_date}}

Scope: Organize backups/strays, add standards and PR hygiene, no functional changes.

## Actions Performed

Moved backup/stray files into `backups/` for safety and clarity:

- backups/archive/CLAUDE.md.backup-20250814_005228
- backups/archive/docker-compose.yml.backup
- backups/archive/docker-compose.yml.backup_20250811_164252
- backups/archive/docker-compose.yml.backup.20250809_114705
- backups/archive/docker-compose.yml.backup.20250810_155642
- backups/archive/docker-compose.yml.backup.20250813_092940
- backups/archive/Makefile.bak-2025-08-11_214515
- backups/env/.env.backup.20250811_193726
- backups/env/.env.secure.backup.20250811_193052
- backups/env/.env.secure.backup.20250813_092537
- backups/misc/=5.27.0

Documentation and hygiene added:

- docs/codebase-standards.md — authoritative standards guide
- .github/pull_request_template.md — standards checklist for PRs
- .github/ISSUE_TEMPLATE/{bug_report.md,feature_request.md}
- .editorconfig — cross‑editor formatting baseline

## Proposed Phase 2 (Pending Approval)

- Docker compose consolidation: designate a canonical `docker-compose.yml` profile set; deprecate variants in docs; archive duplicates per DOCKER_CONSOLIDATION_PLAN.json
- Env/config unification: keep `.env.example` and environment‑specific files under `environments/`; move historical backups into `backups/env/`
- Secrets tightening: verify `.gitignore` covers secrets; rotate any accidentally committed secrets; ensure detect‑secrets baseline is current
- Test/report hygiene: route logs and artifacts to `logs/` and `reports/` consistently; ignore in VCS
- Script deduplication: merge overlapping scripts under `scripts/` per `script_consolidation_cleanup.json`

No code, API, or build behavior changes were made in Phase 1.

