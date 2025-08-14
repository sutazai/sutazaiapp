# Deletion Index — Candidates and Sources

This index consolidates existing lists of duplicate/obsolete files. No deletions were executed in Phase 1.

## Sources

- `phase1_archive_list_comprehensive.txt` — 55 Dockerfiles to archive
- `phase2_removal_list.txt` — 25 exact duplicate Dockerfiles to remove
- `SCRIPT_DELETION_LIST.txt` — 800+ scripts flagged (requires careful review)
- `DOCKER_CONSOLIDATION_PLAN.json` — exhaustive Dockerfile consolidation map

## Policy

- Investigate first; confirm consumers and replacements
- Prefer archiving to `backups/` over deletion in initial pass
- Remove only exact MD5 duplicates or unreachable code with sign‑off

## Next Steps (Proposed)

- Validate MD5 groups in `phase2_removal_list.txt` and remove confirmed duplicates
- Move archive candidates to `backups/dockerfiles/` keeping directory structure
- Deduplicate scripts per `script_consolidation_cleanup.json`
- Open PR with full diff and sign‑offs from CODEOWNERS

