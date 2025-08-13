# ADR: Remove Service Mesh Compose Stack (Kong/Consul/RabbitMQ)

- Status: Accepted
- Date: 2025-08-07
- Owner: Codex CLI (AI)

## Context

CLAUDE.md reflects the current reality: Kong/Consul/RabbitMQ are running but not configured or integrated with application flows. They add operational complexity without delivering value, and they contradict Rule 2 (don’t break) and Rule 1 (no conceptual) when advertised capabilities aren’t actually wired up. The mesh lived in a separate Compose file: `docker-compose.service-mesh.yml`.

## Decision

- Remove `docker-compose.service-mesh.yml` from the repository to prevent accidental deployment of unused/unfinished mesh components.
- Mark scripts under `scripts/service-mesh/` as deprecated without breaking changes.
- Keep config references (e.g., port registry) for historical context; these can be pruned in a follow-up once consumers are audited.

## Alternatives Considered

- Gate under profiles: Not necessary because mesh was already isolated in a separate file; deletion avoids confusion.
- Keep file but document as disabled: Risk of accidental use remains.

## Consequences

- Simpler, truthful Compose footprint aligned with CLAUDE.md reality.
- No impact to core services (backend, frontend, databases, Ollama, monitoring) since they live in the main Compose files.
- Mesh-related scripts remain for future reintroduction, but are clearly marked as deprecated.

## How to Reintroduce Later (If Needed)

- Restore the file in a feature branch and add  , verified routes and health for a single service end-to-end.
- Add tests verifying Kong routes and Consul service registry functionality before merging.

