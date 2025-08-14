# Docker Compose Consolidation

Goal: Reduce variant sprawl and designate a canonical compose definition.

## Canonical

- Primary: `docker-compose.yml`
- Profiles: use Compose profiles and Makefile targets for performance/security variations instead of separate files.

## Variants Present (examples)

- `docker-compose.optimized.yml`, `docker-compose.performance.yml`, `docker-compose.secure.yml`, `docker-compose.standard.yml`, `docker-compose.ultra-performance.yml`, overrides under `*.override.yml`

## Plan

1) Map services from variants into profiles within `docker-compose.yml`
2) Migrate overrides to well‑named profiles (e.g., `performance`, `secure`)
3) Archive superseded files under `backups/compose/` (do not delete)
4) Update docs and Makefile to reference profiles instead of file variants

Reference: `DOCKER_CONSOLIDATION_PLAN.json` for detailed file lists.

