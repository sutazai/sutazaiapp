# ISSUE-0004: Service Mesh Routes Not Defined

- Impacted: API Gateway, Service Discovery
- Options:
  - A: Define Kong services/routes for all internal APIs (recommended)
  - B: Bypass mesh temporarily (localhost links) [violates standards]
  - C: Replace with Traefik (re-evaluate)
- Recommendation: A
- Consequences: Requires route definitions, health checks, CI validation
- Sources: `/workspace/IMPORTANT/SUTAZAI_PRD.md`